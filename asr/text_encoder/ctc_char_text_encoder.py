from typing import List, NamedTuple, Optional
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(
            self,
            alphabet: List[str] = None,
    ):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.char2ind[self.EMPTY_TOK]  # == 0
        decoded_output = []
        for ind in inds:
            if ind == last_char:
                continue
            elif ind != self.char2ind[self.EMPTY_TOK]:
                decoded_output.append(self.ind2char[ind])
            last_char = ind
        return "".join(decoded_output)

    def ctc_beam_search(
            self, probs: torch.tensor, probs_length: int, beam_size: int = 100,
    ) -> List:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        probs = probs[:probs_length, :]
        assert probs.size(0) == probs_length

        probs = probs.cpu().detach().numpy()
        paths = {("", self.EMPTY_TOK): 1.0}

        for i, next_char_probs in enumerate(probs):
            next_char_probs = next_char_probs[:probs_length]
            paths = self._extend_merge(next_char_probs, paths, self.ind2char)
            paths = dict(list(sorted(paths.items(), key=lambda x: x[1]))[-beam_size:])

        return [
            (prefix.strip(), score) for (prefix, _), score in sorted(paths.items(), key=lambda x: -x[1])
        ]

    def _extend_merge(self, next_char_probs, paths, ind2char):
        new_paths = defaultdict(float)
        for next_char_ind, next_char_prob in enumerate(next_char_probs):
            next_char = ind2char[next_char_ind]
            for (text, last_char), path_prob in paths.items():
                next_path = text if next_char == last_char else (text + next_char)
                next_path = next_path.replace(self.EMPTY_TOK, "")
                new_paths[(next_path, next_char)] += path_prob * next_char_prob

        return new_paths
