import torch
from torchmetrics import Metric
import editdistance


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

    def update(self, pred_text: str, target_text: str) -> None:

        wer_num, wer_den = calc_wer(target_text, pred_text)
        cer_num, cer_den = calc_cer(target_text, pred_text)

        self.wer_num += wer_num
        self.wer_den += wer_den
        self.cer_num += cer_num
        self.cer_den += cer_den

    def compute(self) -> float:

        return (
            (self.wer_num / self.wer_den).item() if self.wer_den else 1,
            self.wer_num.item(),
            self.wer_den.item(),
            (self.cer_num / self.cer_den).item() if self.cer_den else 1,
            self.cer_num.item(),
            self.cer_den.item(),
        )
