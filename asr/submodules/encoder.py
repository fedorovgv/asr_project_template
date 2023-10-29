import typing as tp
from abc import abstractmethod

import torch
import torch.nn as nn

from asr.core import CoreModule


class CoreEncoder(CoreModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> tp.Any:
        raise NotImplementedError()


class BaseLineEncoder(CoreEncoder):
    def __init__(
        self, n_mels: int, fc_hidden: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_mels, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden)
        )

    def forward(
        self, features: torch.tensor, features_len: torch.tensor,
    ) -> tp.Tuple[torch.tensor]:

        encoded = self.encoder(features)
        encoded_len = features_len

        return encoded, encoded_len
