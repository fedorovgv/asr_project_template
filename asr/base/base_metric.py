from abc import abstractmethod
from typing import Any

from asr.data.batch import ASRBatch


class BaseMetric:
    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, batch: ASRBatch) -> Any:
        raise NotImplementedError()
