import io
from enum import Enum
import logging
from typing import Optional

from .wandb import WanDBWriter

import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig


class VisualizerBackendType(str, Enum):
    wandb: str = "wandb"


def get_visualizer(
    cfg: DictConfig, logger: logging.Logger, backend: VisualizerBackendType
) -> WanDBWriter:

    if backend == VisualizerBackendType.wandb:
        return WanDBWriter(cfg, logger)

    return None


def plot_spectrogram_to_buf(
    spectrogram_tensor: torch.tensor, name: Optional[str] = None
) -> io.BytesIO:
    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram_tensor)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
