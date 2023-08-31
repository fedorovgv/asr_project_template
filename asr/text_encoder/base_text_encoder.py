import re
import typing as tp

import numpy as np
from torch import Tensor

from asr.core.serialization import Serialization


class CoreTextEncoder(Serialization):
    """
    Base class for all encoders.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__ini__(*args, **kwargs)

    def encode(self, text) -> Tensor:
        raise NotImplementedError()

    def decode(self, vector: tp.Union[Tensor, np.ndarray, tp.List[int]]):
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
