from typing import List, Sequence

import torch
from torchmetrics import Metric


def edit_distance(pred: List[int], target: List[int]) -> int:
    dp = [[0] * (len(target) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(target) + 1):
        dp[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


class CharacterErrorRate(Metric):
    """Character error rate metric, computed using Levenshtein distance.

    target:
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
            distance = edit_distance(pred, target)
            self.error += distance / max(len(pred), len(target))
        self.total += N

    def compute(self) -> torch.Tensor:
        return self.error / self.total
