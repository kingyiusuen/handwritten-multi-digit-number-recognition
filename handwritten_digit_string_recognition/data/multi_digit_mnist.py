import logging
from typing import Optional

import h5py
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode
from pytorch_lightning.utilities.seed import seed_everything

from .base_data_module import BaseDataModule
from .single_digit_mnist import SingleDigitMNIST
from .utils import BaseDataset, DatasetGenerator


logging.basicConfig(level=logging.INFO)


class MultiDigitMNIST(BaseDataModule):
    def __init__(
        self,
        num_train: int = 1000,
        num_val: int = 200,
        num_test: int = 200,
        min_length: int = 1,
        max_length: int = 5,
        min_overlap: float = 0.0,
        max_overlap: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0 < min_length < max_length
        assert 0 <= min_overlap < max_overlap

        self.num_samples = {"train": num_train, "val": num_val, "test": num_test}
        self.min_length = min_length
        self.max_length = max_length
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap

        self.padding_index = 10
        self.blank_index = 11
        self.single_digit_mnist = SingleDigitMNIST()
        self.transform = {
            "train": transforms.Compose([
                transforms.RandomAffine(
                    degrees=(-0.05, 0.05),
                    scale=(0.7, 1.0),
                    shear=(-30, 30),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ToTensor(),
            ]),
            "val/test": transforms.ToTensor(),
        }

    @property
    def dataset_dirname(self):
        return self.data_dirname() / "multi_digit_mnist" / "processed"

    @property
    def dataset_filename(self):
        return (
            self.dataset_dirname /
            f"ml_{self.max_length}_o{self.min_overlap:.2f}_{self.max_overlap:.2f}_" \
            f"ntr{self.num_samples['train']}_nv{self.num_samples['val']}_nte{self.num_samples['test']}.h5"
        )

    def prepare_data(self):
        if self.dataset_filename.is_file():
            return
        self.single_digit_mnist.prepare_data()
        self.single_digit_mnist.setup()
        self.dataset_dirname.mkdir(parents=True, exist_ok=True)
        seed_everything(1234)
        with h5py.File(self.dataset_filename, "w") as f:
            for split in ("train", "val", "test"):
                logging.info(f"Preparing {split} dataset...")
                image_generator = DatasetGenerator(
                    self.single_digit_mnist.dataset[split],
                    min_length=self.min_length,
                    max_length=self.max_length,
                    min_overlap=self.min_overlap,
                    max_overlap=self.max_overlap,
                    padding_index=self.padding_index,
                )
                images, labels = image_generator.generate(self.num_samples[split])
                f.create_dataset(f"X_{split}", data=images, dtype="f4", compression="lzf")
                f.create_dataset(f"y_{split}", data=labels, dtype="i1", compression="lzf")
        logging.info(f"Dataset saved to {str(self.dataset_filename)}")

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            with h5py.File(self.dataset_filename, "r") as f:
                X_train = [Image.fromarray(_) for _ in f["X_train"][:]]
                y_train = torch.IntTensor(f["y_train"])
                X_val = [Image.fromarray(_) for _ in f["X_val"][:]]
                y_val = torch.IntTensor(f["y_val"])
            self.dataset["train"] = BaseDataset(X_train, y_train, transform=self.transform["train"])
            self.dataset["val"] = BaseDataset(X_val, y_val, transform=self.transform["val/test"])

        if stage in ("test", None):
            with h5py.File(self.dataset_filename, "r") as f:
                X_test = f["X_test"][:]
                y_test = torch.IntTensor(f["y_test"][:])
            self.dataset["test"] = BaseDataset(X_test, y_test, transform=self.transform["val/test"])
