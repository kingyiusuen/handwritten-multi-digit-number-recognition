from typing import Sequence

from torchmetrics import Metric
import torch
import editdistance


class CharacterErrorRate(Metric):
    """Character error rate metric, computed using Levenshtein distance."""

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.error: torch.Tensor
        self.total: torch.Tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        N = preds.shape[0]
        for ind in range(N):
            pred = [_ for _ in preds[ind].tolist() if _ not in self.ignore_tokens]
            target = [_ for _ in targets[ind].tolist() if _ not in self.ignore_tokens]
            distance = editdistance.distance(pred, target)
            error = distance / max(len(pred), len(target))
            self.error = self.error + error
        self.total = self.total + N

    def compute(self) -> torch.Tensor:
        return self.error / self.total