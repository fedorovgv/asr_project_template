from abc import abstractmethod
from pathlib import Path
from typing import Union, Optional

import torch
from numpy import inf
from omegaconf import DictConfig

from asr.base.base_model import BaseModel
from asr.base.base_metric import BaseMetric
from asr.logger import get_visualizer
from asr.logger import asr_logger


class BaseTrainer:
    """Base class for all trainers."""
    def __init__(
        self,
        model: BaseModel,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        cfg: DictConfig,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:

        self.cfg = cfg

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = cfg.device

        # all about epochs
        self._last_epoch = 0
        self.start_epoch = 0
        self.epochs = cfg.epochs

        # configuration to monitor model performance and save best
        self.monitor = cfg.get("monitor", "off")

        assert self.monitor in ['on', 'off']

        if self.monitor == "off":
            self.monitor_mode = "off"
            self.monitor_best = 0
        else:
            self.monitor_mode = cfg.monitor_mode
            self.monitor_metric = cfg.monitor

            assert self.monitor_mode in ["min", "max"]

            self.monitor_best = inf if self.monitor_mode == "min" else -inf
            self.early_stop = cfg.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # checkpoint params
        self.save_period = cfg.save_period
        self.checkpoint_dir = Path(cfg.save_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # setup visualization writer instance
        self.writer = get_visualizer(
            cfg, asr_logger, cfg.visualize
        )

        # resume from checkpoint if necessary
        if cfg.get('resume', None) is not None:
            self._resume_checkpoint(cfg.resume)

    @abstractmethod
    def _train_epoch(self, epoch: int) -> None:
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    @abstractmethod
    def _validation_epoch(self, epoch: int) -> None:
        """
        Validation logic for an epoch.
        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            asr_logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs):
            self._last_epoch = epoch

            train_result = self._train_epoch(epoch)
            val_result = self._validation_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(train_result)
            log.update(val_result)

            # print logged informations to the screen
            for key, value in log.items():
                asr_logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.monitor_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.monitor_mode == "min":
                        improved = log[self.monitor_metric] <= self.monitor_best
                    elif self.monitor_mode == "max":
                        improved = log[self.monitor_metric] >= self.monitor_best
                    else:
                        improved = False
                except KeyError:
                    asr_logger.warning(
                        f"Warning: Metric {self.monitor_metric} is not found. "
                        "Model performance monitoring is disabled."
                    )
                    self.monitor_mode = "off"
                    improved = False

                if improved:
                    self.monitor_best = log[self.monitor_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    asr_logger.info(
                        f"Validation performance didn't improve for {self.early_stop} epochs. "
                        "Training stops."
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(
        self, epoch: int, save_best: bool = False, only_best: bool = False,
    ) -> None:
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.monitor_best,
            "config": self.cfg,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch_{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            asr_logger.info(f"Saving checkpoint: {filename}")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            asr_logger.info("Saving current best: model_best.pth")

    def _resume_checkpoint(self, resume_path: Union[str, Path]) -> None:
        resume_path = str(resume_path)
        asr_logger.info(f"Loading checkpoint: {resume_path}")

        checkpoint = torch.load(resume_path, self.device)

        self.start_epoch = checkpoint["epoch"]
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            asr_logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"] or
            checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            asr_logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        asr_logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
