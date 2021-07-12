from typing import Optional

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from .base_data_module import BaseDataModule
from .utils import split_dataset


TRAIN_FRACTION = 0.8


class SingleDigitMNIST(BaseDataModule):
    """Data module for the standard single digit MNIST dataset.

    Args:
        kwargs: Keyword arguments to BaseDataModule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = transforms.ToTensor()

    def prepare_data(self) -> None:
        """Download the MNIST dataset."""
        MNIST(self.data_dirname(), train=True, download=True)
        MNIST(self.data_dirname(), train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            full_dataset = MNIST(
                self.data_dirname(), train=True, transform=self.transform
            )
            self.dataset["train"], self.dataset["val"] = split_dataset(
                full_dataset, TRAIN_FRACTION, seed=42
            )

        if stage in ("test", None):
            self.dataset["test"] = MNIST(
                self.data_dirname(), train=False, transform=self.transform
            )
