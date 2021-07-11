from pathlib import Path
from typing import Dict, Union, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset


class BaseDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset: Dict[str, Union[Dataset, Subset]] = {}

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    def prepare_data(self):
        """Download and preprocess datasets."""

    def setup(self, stage: Optional[str] = None):
        """Should popularize self.dataset after being called."""

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
                
    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
