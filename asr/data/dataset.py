from typing import Union, Optional, Tuple
from pathlib import Path

import torch
import torchaudio
import pandas as pd
from omegaconf import DictConfig

from asr.text_encoder import CoreTextEncoder
from asr.core.serialization import Serialization
from .batch import ASRSample


class ASRDataset(torch.utils.data.Dataset, Serialization):

    def __init__(
        self,
        manifest_path: Union[Path, str],
        text_encoder: CoreTextEncoder,
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
            manifest_path: path for index in csv format
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

        self.dataset_name = self._parse_dataset_name(manifest_path)

        index = pd.read_csv(manifest_path, sep='\t')

        self._index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length,
        )
        self._index = index

        del index

    @staticmethod
    def _parse_dataset_name(manifest_path: Union[str, Path]) -> str:
        return manifest_path.split('/')[-1]

    def __getitem__(self, index: int) -> ASRSample:

        data_row = self._index.iloc[index]

        audio_path = data_row.audio_path
        audio_wave = self._load_audio(audio_path)
        audio_wave, spectrogram = self._process_wave(audio_wave)

        transcriprion = CoreTextEncoder.normalize_text(
            data_row.transcriprion,
        )
        target = self.text_encoder.encode(transcriprion)

        return ASRSample(
            audio_path=audio_path,
            audio_wave=audio_wave,
            duration=data_row.duration,
            spec=spectrogram,
            text=transcriprion,
            target=target,
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


def get_asr_dataset(cfg: DictConfig, text_encoder: CoreTextEncoder) -> ASRDataset:
    """
    Returns dataset according to config.
    """
    wave_aug = (
        Serialization.from_config(cfg.wave_aug) if cfg.get('wave_aug', False) else None
    )
    spec_aug = (
        Serialization.from_config(cfg.spec_aug) if cfg.get('spec_aug', False) else None
    )
    wave2spec = Serialization.from_config(cfg.wave2spec)

    dataset_cfg = cfg.copy()

    if dataset_cfg.get('wave_aug', False):
        del dataset_cfg.wave_aug

    if dataset_cfg.get('spec_aug', False):
        del dataset_cfg.spec_aug

    dataset_kwargs = {
        'text_encoder': text_encoder,
        'wave_aug': wave_aug,
        'spec_aug': spec_aug,
        'wave2spec': wave2spec,
    }
    dataset = Serialization.from_config(dataset_cfg, **dataset_kwargs)

    return dataset
