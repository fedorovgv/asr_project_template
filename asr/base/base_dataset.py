from typing import Union, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from .base_text_encoder import BaseTextEncoder
from .serialization import Serialization


@dataclass
class ASRSample:
    audio_path: Union[Path, str]
    audio_wave: torch.tensor

    duration: float

    spectrogram: torch.tensor

    transcriprion: str
    transcriprion_encoded: torch.tensor


class BaseDataset(Dataset, Serialization):
    def __init__(
        self,
        index_path: Union[Path, str],
        text_encoder: BaseTextEncoder,
        wave_aug,
        spec_aug,
        wave2spec,
        log_spec: bool = True,
        max_audio_length: Optional[int] = None, 
        max_text_length: Optional[int] = None, 
    ) -> None:
        """
        Base class for all datasets.
        args:
            index_path: path for index in csv format
            text_encoder: text encoder object
            wave_augs: wave augs
            spec_augs: spec augs
            max_audio_length: max audio length for wav filtering by their durations
            max_text_length: max text length for wav filtering by transcriprion lengths
        """
        super().__init__()

        self.text_encoder = text_encoder
        self.wave_aug = wave_aug
        self.spec_aug = spec_aug
        self.log_spec = log_spec
        self.wave2spec = wave2spec

        self.dataset_name = self._parse_dataset_name(index_path)

        index = pd.read_csv(index_path, sep='\t')

        self._index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length,
        )
        self._index = index

        del index

    @staticmethod
    def _parse_dataset_name(index_path: Union[str, Path]) -> str:
        return index_path.split('/')[-1]

    def __getitem__(self, index: int) -> ASRSample:

        data_row = self._index.iloc[index]
        audio_path = data_row.audio_path
        audio_wave = self._load_audio(audio_path)
        audio_wave, spectrogram = self._process_wave(audio_wave)

        return ASRSample(
            audio_path=audio_path,
            audio_wave=audio_wave,
            duration=data_row.duration,
            spectrogram=spectrogram,
            transcriprion=data_row.transcriprion,
            transcriprion_encoded=self.text_encoder.encode(data_row.transcriprion),
        )

    def __len__(self) -> int:
        return len(self._index)

    def _load_audio(self, path: Union[Path, str]) -> torch.tensor:
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        return audio_tensor

    def _process_wave(self, audio_tensor_wave: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:

        with torch.no_grad():
            if self.wave_aug is not None:
                audio_tensor_wave = self.wave_aug(audio_tensor_wave)
            spectrogram = self.wave2spec(audio_tensor_wave)
            if self.spec_aug is not None:
                spectrogram = self.spec_aug(spectrogram)
            if self.log_spec:
                spectrogram = torch.log(spectrogram + 1e-5)
            return audio_tensor_wave, spectrogram

    @staticmethod
    def _filter_records_from_dataset(
            index: pd.DataFrame, max_audio_length: int, max_text_length: int,
    ) -> pd.DataFrame:
        return index
