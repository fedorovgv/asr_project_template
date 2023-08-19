import re
from typing import List, Union

import numpy as np
from torch import Tensor
from omegaconf import OmegaConf, DictConfig

from .serialization import Serialization


class BaseTextEncoder(Serialization):
    """
    Base class for all encoders.
    """

    def encode(self, text) -> Tensor:
        raise NotImplementedError()

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item: int) -> str:
        raise NotImplementedError()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
