import logging
from datetime import datetime
from typing import Optional, Any

import torch
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig


class WanDBWriter:
    def __init__(self, cfg: DictConfig, logger: logging.Logger) -> None:
        self.writer = None
        self.selected_module = ""

        try:
            import wandb
            wandb.login()

            if cfg.get('wandb_project', None) is None:
                raise ValueError("please specify project name for wandb")

            wandb.init(
                project=cfg.wandb_project,
                config=cfg.get('config', None),
            )
            self.wandb = wandb

        except ImportError:
            logger.warning("For use wandb install it via \n\t pip install wandb")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step: int, mode: str = "train") -> None:
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def _scalar_name(self, scalar_name: str) -> str:
        return f"{scalar_name}_{self.mode}"

    def add_scalar(self, scalar_name: str, scalar: int) -> None:
        self.wandb.log({
            self._scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag: str, scalars: list) -> None:
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in
               scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name: str, image: torch.tensor) -> None:
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_audio(self, scalar_name: str, audio: torch.tensor, sample_rate: Optional[int] = None):
        audio = audio.detach().cpu().numpy().T
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)

    def add_text(self, scalar_name: str, text: str) -> None:
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_histogram(self, scalar_name: str, hist: torch.tensor, bins: Optional[int] = None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self._scalar_name(scalar_name): hist
        }, step=self.step)

    def add_table(self, table_name: str, table: pd.DataFrame) -> None:
        self.wandb.log({self._scalar_name(table_name): wandb.Table(dataframe=table)},
                       step=self.step)

    def add_images(self, scalar_name: str, images: Any) -> None:
        raise NotImplementedError()

    def add_pr_curve(self, scalar_name: str, scalar: float) -> None:
        raise NotImplementedError()

    def add_embedding(self, scalar_name: str, scalar: float) -> None:
        raise NotImplementedError()
