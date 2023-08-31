from typing import Optional, List, Union
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ASRSample:
    audio_path: Union[Path, str]
    audio_wave: torch.tensor

    duration: float

    spec: torch.tensor

    text: str
    target: torch.tensor


@dataclass
class ASRBatch:
    """
        B - batch size
        T - max time of mel spectrograms accross batch
        Mf - mel frequences nums
        L - max lengths of text sequence
    """
    features: torch.tensor  # (B, T, Mf)
    features_len: torch.tensor  # (B,)

    targets: torch.tensor  # (B, L)
    targets_len: torch.tensor  # (B,)

    audio_paths: List[Union[Path, str]]
    audio_waves: List[torch.tensor]
    waves_dur: List[float]

    texts: Optional[List[str]] = None

    log_probs: Optional[torch.tensor] = None
    log_probs_lengths: Optional[torch.tensor] = None

    def to(self, device: str) -> None:
        self.features = self.features.to(device)
        self.features_len = self.features_len.to(device)
