import torch
from torch.nn import CTCLoss
from asr.data.batch import ASRBatch


class CTCLossWrapper(CTCLoss):

    def forward(self, batch: ASRBatch) -> torch.tensor:

        log_probs_t = torch.transpose(batch.log_probs, 0, 1)

        return super().forward(
            log_probs=log_probs_t,
            targets=batch.transcriptions_encoded,
            input_lengths=batch.spectrograms_lengths,
            target_lengths=batch.transcriptions_encoded_lengths,
        )
