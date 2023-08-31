import typing as tp
from abc import abstractmethod

import torch
import torch.nn as nn

from asr.core import CoreModule


class CoreDecoder(CoreModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> tp.Any:
        raise NotImplementedError()


class BaseLineDecoder(CoreDecoder):
    def __init__(
        self, fc_hidden: int, n_class: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.proj = nn.Linear(
            in_features=fc_hidden, out_features=n_class,
        )

    def forward(self, encoded_features: torch.tensor) -> torch.tensor:

        logits = self.proj(encoded_features)

        return logits
