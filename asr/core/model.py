from abc import abstractmethod
from typing import Any

import pytorch_lightning as pl

from .serialization import Serialization


class CoreModule(pl.LightningModule, Serialization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
