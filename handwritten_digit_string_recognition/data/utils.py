import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, random_split


class BaseDataset(Dataset):
    def __init__(
        self,
        images: Sequence,
        targets: Sequence,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        self.images = images
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        image, target = self.images[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


class DatasetGenerator:
    def __init__(
        self,
        single_digit_mnist,
        min_length,
        max_length,
        min_overlap,
        max_overlap,
        padding_index,
    ):
        self.single_digit_mnist = single_digit_mnist
        self.min_length = min_length
        self.max_length = max_length
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.padding_index = padding_index

        self.samples_by_digit = self.get_samples_by_digit()

    def get_samples_by_digit(self) -> Dict[int, List]:
        """Stores a collection of images for each digit."""
        samples_by_digit = defaultdict(list)
        for image, digit in self.single_digit_mnist:
            samples_by_digit[digit].append(image.squeeze())
        return samples_by_digit

    def generate(self, num_samples):
        labels = torch.full((num_samples, self.max_length), self.padding_index)
        images = torch.zeros((num_samples, 32, 28 * self.max_length))
        for i in range(num_samples):
            rand_num = random.randint(
                int("1" + "0" * (self.min_length - 1)),
                int("1" + "0" * (self.max_length - 1))
            )
            for j, digit in enumerate(str(rand_num)):
                labels[i, j] = int(digit)
            images[i] = self.construct_image_from_number(rand_num)
        return images, labels

    def construct_image_from_number(self, number):
        """Concatenate images of single digits."""
        overlap = random.uniform(self.min_overlap, self.max_overlap)
        overlap_width = int(overlap * 28)
        width_increment = 28 - overlap_width
        x, y = 0, 2  # Left and top paddings
        multi_digit_image = torch.zeros((32, 28 * self.max_length))
        digit_samples = self.select_digit_samples()
        for digit in str(number):
            digit_image = digit_samples[int(digit)]
            digit_image[:, :overlap_width] = torch.maximum(
                multi_digit_image[y:y+28, x:x+overlap_width],
                digit_image[:, :overlap_width]
            )
            multi_digit_image[y:y+28, x:x+28] = digit_image
            x += width_increment
        return multi_digit_image

    def select_digit_samples(self):
        return {i: random.choice(self.samples_by_digit[i]) for i in range(10)}


def split_dataset(dataset, fraction, seed):
    num_samples = len(dataset)
    split_a_size = int(num_samples * fraction)
    split_b_size = num_samples - split_a_size
    return random_split(dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed))
