import random
from pathlib import Path
from random import shuffle
from typing import Optional, Union, List, Dict

import pandas as pd
import PIL
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from asr.base import BaseTrainer
from asr.base import BaseModel, BaseMetric
from asr.base.base_text_encoder import BaseTextEncoder
from asr.data.batch import ASRBatch
from asr.logger.utils import plot_spectrogram_to_buf
from asr.utils import inf_loop
from asr.metric import WER, calc_wer, calc_cer
from asr.logger import asr_logger


class Trainer(BaseTrainer):
    """Trainer class."""
    def __init__(
            self,
            model: BaseModel,
            criterion: torch.nn.modules.loss._Loss,
            optimizer: torch.optim.Optimizer,
            cfg: DictConfig,
            train_dataloader: DataLoader,
            val_dataloaders: Union[DataLoader, List[DataLoader]],
            text_encoder:  BaseTextEncoder,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            len_epoch: Optional[int] = None,
    ) -> None:

        super().__init__(model, criterion, optimizer, cfg, lr_scheduler)

        self.text_encoder = text_encoder

        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloaders: Dict[str, DataLoader] = val_dataloaders

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch

        self.wer = WER()

    def _clip_grad_norm(self):
        if self.cfg.get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_norm_clip
            )

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training logic for an epoch."""
        self.writer.add_scalar("epoch", epoch)

        self.model.train()

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            batch = self.process_batch(batch, is_train=True)
            self.wer.update(batch, text_encoder=self.text_encoder)

            if (
                self.cfg.get('log_step', None) is not None and batch_idx % self.cfg.log_step == 0
            ):
                self.writer.set_step(epoch * self.len_epoch + batch_idx)
                self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])

                self._log_predictions(batch)
                self._log_spectrogram(batch)

                asr_logger.debug(
                    f"train epoch: {epoch} {self._progress(batch_idx)} |"
                    f"loss: {batch.batch_loss.item()}"
                )

            if batch_idx >= self.len_epoch:
                break

        train_result = self.wer.compute()
        train_result = {
            f"training_{key}": value for key, value in train_result.items()
        }
        self.wer.reset()

        return train_result

    def process_batch(self, batch: ASRBatch, is_train: bool) -> ASRBatch:
        """
        Move batch to device. Procces it and save model outputs.
        """
        batch.to(self.device)

        if is_train:
            self.optimizer.zero_grad()

        batch = self.model(batch)
        batch.log_probs = F.log_softmax(batch.logits, dim=-1)
        batch.log_probs_lengths = self.model.transform_input_lengths(
            batch.spectrograms_lengths
        )
        batch.batch_loss = self.criterion(batch)

        if is_train:
            batch.batch_loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return batch

    def _validation_epoch(self, epoch: int) -> Dict[str, float]:

        self.model.eval()

        validation_result = {}

        for part, dataloader in self.val_dataloaders.items():
            with torch.no_grad():
                for _, batch in tqdm(
                    enumerate(dataloader), desc=part, total=len(dataloader),
                ):
                    batch = self.process_batch(batch, is_train=False)
                    self.wer.update(batch, text_encoder=self.text_encoder)

                self.writer.set_step(epoch * self.len_epoch, part)

                self._log_predictions(batch)
                self._log_spectrogram(batch)

                part_result = self.wer.compute()
                validation_result.update(
                    {f'{part}_{key}': value for key, value in part_result.items()}
                )
                self.wer.reset()

        return validation_result

    def _progress(self, batch_idx: int) -> str:
        """
        Logging training or validation progress.
        """
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return f'[{current}/{total} ({round(100.0 * current / total, 2)})]'

    def _log_predictions(
            self, batch: ASRBatch, examples_to_log: int = 5,
    ) -> None:
        """
        Logging predictions during training or validation.
        """

        # TODO: implement logging of beam search results
        if self.writer is None:
            return

        argmax_inds = batch.log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, batch.log_probs_lengths.numpy())
        ]

        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        tuples = list(
            zip(argmax_texts, batch.transcriptions, argmax_texts_raw, batch.audio_paths)
        )
        shuffle(tuples)

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = BaseTextEncoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }

        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, batch: ASRBatch) -> None:
        """
        Logging randomly selected spectrogramm.
        """
        spectrogram = random.choice(batch.spectrograms.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
