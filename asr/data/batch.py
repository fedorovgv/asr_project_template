from typing import Optional, List, Union
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ASRBatch:
    """
        B - batch size
        T - max time of mel spectrograms accross batch
        Mf - mel frequences nums
        L - max lengths of text sequence
    """
    spectrograms: torch.tensor  # (B, T, Mf)
    spectrograms_lengths: torch.tensor  # (B,)

    transcriptions_encoded: torch.tensor  # (B, L)
    transcriptions_encoded_lengths: torch.tensor  # (B,)

    audio_paths: List[Union[Path, str]]
    audio_waves: List[torch.tensor]
    wave_durations: List[float]

    batch_loss: Optional[float] = None  # may be useful for debug

    transcriptions: Optional[List[str]] = None

    log_probs: Optional[torch.tensor] = None
    log_probs_lengths: Optional[torch.tensor] = None

    def to(self, device: str) -> None:
        self.spectrograms_lengths = self.spectrograms_lengths.to(device)
        self.spectrograms = self.spectrograms.to(device)
