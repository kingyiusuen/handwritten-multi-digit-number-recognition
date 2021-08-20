import itertools
from typing import Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from ..models import CRNN
from .metrics import CharacterErrorRate


def first_element(
    x: torch.Tensor,
    element: Union[int, float],
    dim: int = 1,
) -> torch.Tensor:
    """Find the first occurence of element in x along a given dimension.

    Args:
        x: The input tensor to be searched.
        element: The number to look for.
        dim: The dimension to reduce.

    Returns:
        Indices of the first occurence of the element in x. If not found, return the
        length of x along dim.

    Usage:
        >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])

    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/9

        I fixed an edge case where the element we are looking for is at index 0. The
        original algorithm will return the length of x instead of 0.
    """
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    indices[(~found) & (indices == 0)] = x.shape[dim]
    return indices


class CTCLitModel(LightningModule):
    """A LightningModule for CTC loss.

    Reference:
        https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/lit_models/ctc.py
    """

    def __init__(
        self,
        padding_index: int,
        blank_index: int,
        lr: float = 0.001,
        mode: str = "min",
        monitor: str = "val/loss",
        factor: float = 0.5,
        patience: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = CRNN()
        self.loss_fn = nn.CTCLoss(blank=self.hparams.blank_index, zero_infinity=True)
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        ignore_tokens = [self.hparams.padding_index]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.mode,
            factor=self.hparams.factor,
            patience=self.hparams.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, x: torch.Tensor, max_length: int = 10):
        logits = self.model(x)
        logprobs = torch.log_softmax(logits, dim=1)
        decoded = self.greedy_decode(logprobs, max_length=max_length)
        pred_lengths = first_element(decoded, self.hparams.padding_index)
        return decoded, pred_lengths

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = first_element(y, self.hparams.padding_index).type_as(y)
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # (S, B, C)
        input_lengths = (
            torch.ones(B).type_as(logprobs_for_loss).int() * S
        )  # All are max sequence length
        target_lengths = first_element(y, self.hparams.padding_index).type_as(
            y
        )  # Length is up to first padding token
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("val/loss", loss, prog_bar=True)

        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])  # (B, max_length)
        self.val_acc(decoded, y)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(decoded, y)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logprobs = torch.log_softmax(logits, dim=1)
        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])
        self.test_acc(decoded, y)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_cer(decoded, y)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

    def greedy_decode(self, logprobs: torch.Tensor, max_length: int) -> torch.Tensor:
        B = logprobs.shape[0]
        argmax = logprobs.argmax(1)
        decoded = (
            torch.ones((B, max_length)).type_as(logprobs).int()
            * self.hparams.padding_index
        )
        for i in range(B):
            seq = [
                b
                for b, _ in itertools.groupby(argmax[i].tolist())
                if b != self.hparams.blank_index
            ][:max_length]
            for ii, char in enumerate(seq):
                decoded[i, ii] = char
        return decoded
