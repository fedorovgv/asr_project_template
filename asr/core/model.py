from abc import abstractmethod
from typing import Any

import pytorch_lightning as pl

from asr.data.batch import ASRBatch
from .serialization import Serialization


class CoreModule(pl.LightningModule, Serialization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, batch: ASRBatch) -> ASRBatch:
        raise NotImplementedError()

    def transform_input_lengths(self, batch: ASRBatch) -> ASRBatch:
        raise NotImplementedError()
