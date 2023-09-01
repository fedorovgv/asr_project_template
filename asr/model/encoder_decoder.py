import typing as tp

import torch
from omegaconf import DictConfig

from asr.core import CoreModule, Serialization
from asr.submodules import CoreEncoder, CoreDecoder
from asr.text_encoder import CoreTextEncoder
from asr.data import ASRBatch, collate_fn, get_asr_dataset
from asr.metric import WER
from asr.logger import logger


class EncDecModel(CoreModule):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cfg = cfg

        self.log_every_n_steps = cfg.get('log_every_n_steps', 1)

        self.encoder = CoreEncoder.from_config(cfg.encoder)
        self.decoder = CoreDecoder.from_config(cfg.decoder)

        self.text_encoder = CoreTextEncoder.from_config(cfg.text_encoder)

        self.loss = Serialization.from_config(cfg.loss)

        self.wer = WER()

    def forward(
        self, features: torch.Tensor, features_len: torch.Tensor
    ) -> torch.Tensor:

        encoded, encoded_len = self.encoder(features, features_len)

        logprobs = self.decoder(encoded)
        preds = logprobs.argmax(dim=-1)

        return logprobs, encoded_len, preds

    def training_step(self, batch: ASRBatch, batch_nb: int):
        # process batch
        features, features_len = batch.features, batch.features_len
        targets, targets_len = batch.targets, batch.targets_len
        texts = batch.texts

        # forward pass 
        logprobs, encoded_len, preds = self.forward(features, features_len)

        # loss calculating
        loss = self.loss(
            logprobs.transpose(1, 0), targets, encoded_len, targets_len
        ).mean()

        log = {"train_loss": loss, "lr": self.optimizers().param_groups[0]["lr"]}

        if (batch_nb + 1) % self.log_every_n_steps == 0:

            for (logits, logit_len, target_text) in zip(preds.numpy(), features_len.numpy(), texts):
                if hasattr(self.text_encoder, "ctc_decode"):
                    pred_text = self.text_encoder.ctc_decode(logits[:logit_len])
                else:
                    pred_text = self.text_encoder.decode(logits[:logit_len])

                self.wer.update(pred_text, target_text)
    
            wer, _, _, cer, _, _ = self.wer.compute()
            self.wer.reset()

            log["train_wer"] = wer
            log["train_cer"] = cer

            logger.info(f"reference  : {target_text}")
            logger.info(f"prediction : {pred_text}")

        self.log_dict(log)

        return {"loss": loss}

    def validation_step(self, batch: ASRBatch, batch_nb: int):

        features, features_len = batch.features, batch.features_len
        targets, targets_len = batch.targets, batch.targets_len
        texts = batch.texts

        logprobs, encoded_len, preds = self.forward(features, features_len)

        loss = self.loss(
            logprobs.transpose(1, 0), targets, encoded_len, targets_len,
        )

        for logits, logit_len, target_text in zip(preds.numpy(), features_len.numpy(), texts):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(logits[:logit_len])
            else:
                pred_text = self.text_encoder.decode(logits[:logit_len])

            self.wer.update(pred_text, target_text)

        logger.info(f"reference  : {target_text}")
        logger.info(f"prediction : {pred_text}")

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):

        val_loss = torch.cat([x["val_loss"].reshape(1) for x in outputs]).mean()

        wer, _, wer_den, cer, _, cer_den = self.wer.compute()
        self.wer.reset()

        self.log_dict(
            {
                "val_wer": wer if wer_den != 0 else 1,
                "val_cer": cer if cer_den != 0 else 1,
                "val_loss": val_loss,
            }
        )

    def train_dataloader(self):
        train_asr_dataset = get_asr_dataset(
            self._cfg.train_dataloader.dataset, self.text_encoder
        )

        return torch.utils.data.DataLoader(
            dataset=train_asr_dataset,
            batch_size=self._cfg.train_dataloader.batch_size,
            num_workers=self._cfg.train_dataloader.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        val_asr_dataset = get_asr_dataset(
            self._cfg.val_dataloader.dataset, self.text_encoder
        )

        return torch.utils.data.DataLoader(
            dataset=val_asr_dataset,
            batch_size=self._cfg.val_dataloader.batch_size,
            num_workers=self._cfg.val_dataloader.num_workers,
            collate_fn=collate_fn,
        )

    def configure_optimizers(self):
        optimizer = Serialization.from_config(
            self._cfg.optim, params=self.parameters(),
        )
        lr_scheduler = Serialization.from_config(
            self._cfg.lr_scheduler, optimizer=optimizer,
        )
        lr_scheduler = {
            "scheduler": lr_scheduler, "interval": "step",
        }
        return {
            "optimizer": optimizer, "lr_scheduler": lr_scheduler
        }
