from typing import List

import editdistance
import torch
from torchmetrics import Metric

from asr.base.base_text_encoder import BaseTextEncoder
from asr.data.batch import ASRBatch


def calc_cer(target_text: str, predicted_text: str) -> float:
    if len(target_text) == 0:
        return 1
    return (
        editdistance.distance(target_text, predicted_text),
        len(target_text)
    )


def calc_wer(target_text: str, predicted_text: str) -> float:
    splitted_target_text: str = target_text.split(' ')
    if len(splitted_target_text) == 0:
        return 1
    return (
        editdistance.distance(splitted_target_text, predicted_text.split(' ')),
        len(splitted_target_text)
    )


class WER(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "wer_num", default=torch.tensor(0), dist_reduce_fx="sum",
        )
        self.add_state(
            "wer_den", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "cer_num", default=torch.tensor(0), dist_reduce_fx="sum",
        )
        self.add_state(
            "cer_den", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, batch: ASRBatch, text_encoder) -> None:

        predictions = torch.argmax(batch.log_probs.cpu(), dim=-1).numpy()
        lengths = batch.log_probs_lengths.detach().numpy()

        for log_prob_vec, length, target_text in zip(predictions, lengths, batch.transcriptions):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(text_encoder, "ctc_decode"):
                pred_text = text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = text_encoder.decode(log_prob_vec[:length])
            
            wer_num, wer_den = calc_wer(target_text, pred_text)
            cer_num, cer_den = calc_cer(target_text, pred_text)

            self.wer_num += wer_num
            self.wer_den += wer_den
            self.cer_num += cer_num
            self.cer_den += cer_den

    def compute(self) -> float:

        return {
            'wer': (self.wer_num / self.wer_den).item() if self.wer_den else 1,
            'cer': (self.cer_num / self.cer_den).item() if self.cer_den else 1,
        }

    def reset(self) -> None:
        self.wer_num = torch.empty_like(self.wer_num).fill_(0)
        self.wer_den = torch.empty_like(self.wer_den).fill_(0)
        self.cer_num = torch.empty_like(self.cer_num).fill_(0)
        self.cer_den = torch.empty_like(self.cer_den).fill_(0)
