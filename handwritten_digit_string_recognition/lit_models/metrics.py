from typing import Sequence

import editdistance
import torch
from torchmetrics import Metric


class CharacterErrorRate(Metric):
    """Character error rate metric, computed using Levenshtein distance.
    
    Reference:
        https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/lit_models/metrics.py
    """

    def __init__(self, ignore_tokens: Sequence[int], *args) -> None:
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.error: torch.Tensor
        self.total: torch.Tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        N = preds.shape[0]
        for i in range(N):
            pred = [ind for ind in preds[i].tolist() if ind not in self.ignore_tokens]
            target = [
                ind for ind in targets[i].tolist() if ind not in self.ignore_tokens
            ]
            distance = editdistance.distance(pred, target)
            error = distance / max(len(pred), len(target))
            self.error = self.error + error
        self.total = self.total + N

    def compute(self) -> torch.Tensor:
        return self.error / self.total
