from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from .batch import ASRBatch
from .dataset import ASRSample


def collate_fn(samples: List[ASRSample]) -> ASRBatch:
    """
    Collate and pad fields in dataset items.
    """
    # spectrogram
    features = [
        sample.spec.squeeze(0).transpose(1, 0) for sample in samples
    ]
    features_len = torch.tensor([item.size(0) for item in features])
    features = pad_sequence(features, batch_first=True)

    # transcription
    targets = [
        torch.transpose(sample.target, 1, 0) for sample in samples
    ]
    target_len = torch.tensor(
        [sample.target.size(0) for sample in samples]
    )
    targets = pad_sequence(targets, batch_first=True).squeeze(-1)

    # transcriprions for metric calculation
    texts = [sample.text for sample in samples]

    # audio for debug or logging
    audio_paths = [sample.audio_path for sample in samples]
    audio_waves = [sample.audio_wave for sample in samples]
    wave_durations = [sample.duration for sample in samples]

    return ASRBatch(
        features=features,
        features_len=features_len,
        targets=targets,
        targets_len=target_len,
        audio_paths=audio_paths,
        audio_waves=audio_waves,
        waves_dur=wave_durations,
        texts=texts
    )
