import itertools
from typing import List, Union

import torch
import torch.nn as nn
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from ..models import CRNN
from .metrics import CharacterErrorRate


def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    """
    Return indices of first occurence of element in x. If not found, return length of x along dim.
    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    Examples
    --------
    >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    tensor([2, 1, 3])
    """
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    return ind


class CTCLitModel(LightningModule):
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
            }
        }

    def forward(self, x: torch.Tensor, max_length: int = 10) -> List[List[int]]:
        logits = self.model(x)
        logprobs = torch.log_softmax(logits, dim=1)
        decoded = self.greedy_decode(logprobs, max_length=max_length)
        pred_lengths = first_element(decoded, self.hparams.padding_index)
        pred_nums = [num[:pred_length]for num, pred_length in zip(decoded.tolist(), pred_lengths)]
        return pred_nums

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
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S  # All are max sequence length
        target_lengths = first_element(y, self.hparams.padding_index).type_as(y)  # Length is up to first padding token
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("val/loss", loss, prog_bar=True)

        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])  # (B, max_length)
        self.val_acc(decoded, y)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(decoded, y)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
        try:
            pred_length = first_element(decoded[0], self.hparams.padding_index, dim=0).type_as(y)
            pred_num = "".join(decoded[0].tolist()[:pred_length])
            self.logger.experiment.log({"val/pred_examples": [wandb.Image(x[0], caption=pred_num)]})
        except AttributeError:
            pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logprobs = torch.log_softmax(logits, dim=1)
        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])
        self.test_acc(decoded, y)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_cer(decoded, y)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        try:
            pred_length = first_element(decoded[0], self.hparams.padding_index, dim=0).type_as(y)
            pred_num = "".join(decoded[0].tolist()[:pred_length])
            self.logger.experiment.log({"test/pred_examples": [wandb.Image(x[0], caption=pred_num)]})
        except AttributeError:
            pass

    def greedy_decode(self, logprobs: torch.Tensor, max_length: int) -> torch.Tensor:
        B = logprobs.shape[0]
        argmax = logprobs.argmax(1)
        decoded = torch.ones((B, max_length)).type_as(logprobs).int() * self.hparams.padding_index
        for i in range(B):
            seq = [b for b, _ in itertools.groupby(argmax[i].tolist()) if b != self.hparams.blank_index][:max_length]
            for ii, char in enumerate(seq):
                decoded[i, ii] = char
        return decoded
