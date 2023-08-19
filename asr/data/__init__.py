from typing import List, Dict

import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from asr.base import BaseTextEncoder, BaseDataset
from asr.base.serialization import Serialization
from asr.data.batch import ASRBatch
from asr.base.base_dataset import ASRSample


def collate_fn(asr_samples: List[ASRSample]) -> ASRBatch:
    """
    Collate and pad fields in dataset items.
    """
    # spectrogram
    spectrograms = [
        asr_sample.spectrogram.squeeze(0).transpose(1, 0) for asr_sample in asr_samples
    ]
    spectrograms_lengths = torch.tensor([item.size(0) for item in spectrograms])
    spectrograms = pad_sequence(spectrograms, batch_first=True).transpose(2, 1)

    # transcription
    transcriptions_encoded = [
        torch.transpose(asr_sample.transcriprion_encoded, 1, 0) for asr_sample in asr_samples
    ]
    transcriptions_encoded_lengths = torch.tensor(
        [asr_sample.transcriprion_encoded.size(0) for asr_sample in asr_samples]
    )
    transcriptions_encoded = pad_sequence(transcriptions_encoded, batch_first=True).squeeze(-1)

    # transcriprions for metric calculation
    transcriprions = [asr_sample.transcriprion for asr_sample in asr_samples]

    # audio for debug or logging
    audio_paths = [asr_sample.audio_path for asr_sample in asr_samples]
    audio_waves = [asr_sample.audio_wave for asr_sample in asr_samples]
    wave_durations = [asr_sample.duration for asr_sample in asr_samples]

    return ASRBatch(
        spectrograms_lengths=spectrograms_lengths,
        spectrograms=spectrograms,
        transcriptions_encoded=transcriptions_encoded,
        transcriptions_encoded_lengths=transcriptions_encoded_lengths,
        audio_paths=audio_paths,
        audio_waves=audio_waves,
        wave_durations=wave_durations,
        transcriptions=transcriprions,
    )


def _get_dataset(cfg: DictConfig, text_encoder: BaseTextEncoder) -> BaseDataset:
    """
    Returns dataset according to config.
    """
    wave_aug = Serialization.from_config(cfg.wave_aug) if cfg.get('wave_aug', False) else None
    spec_aug = Serialization.from_config(cfg.spec_aug) if cfg.get('spec_aug', False) else None
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


def get_dataloader(
    cfg: DictConfig, text_encoder: BaseTextEncoder
) -> torch.utils.data.DataLoader:
    dataset_cfg = cfg.dataset
    dataset = _get_dataset(dataset_cfg, text_encoder)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.get('shuffle', False),
        num_workers=cfg.get('num_workers', 0),
        collate_fn=collate_fn,
        pin_memory=cfg.get('pin_memory', False),
        drop_last=cfg.get('drop_last', False),
    )
    return dataloader


def get_dataloaders(
    cfg: DictConfig, text_encoder: BaseTextEncoder
) -> Dict[str, torch.utils.data.DataLoader]:

    index_paths = cfg.dataset.index_path
    if isinstance(index_paths, str):
        index_paths = [index_paths]

    dataloaders = []
    for index_path in index_paths:
        dataloder_cfg = cfg.copy()
        dataloder_cfg.dataset.index_path = index_path
        dataloader = get_dataloader(
            dataloder_cfg, text_encoder
        )
        dataloaders.append(
            (dataloader.dataset.dataset_name, dataloader)
        )

    return dict(dataloaders)
