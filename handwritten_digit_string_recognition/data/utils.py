import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset, random_split


class BaseDataset(Dataset):
    def __init__(
        self,
        images: Union[Sequence, torch.Tensor],
        targets: Union[Sequence, torch.Tensor],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.images = images
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.targets)

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset."""
        image, target = self.images[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


class DatasetGenerator:
    def __init__(
        self,
        single_digit_mnist: Dataset,
        max_length: int,
        min_overlap: float,
        max_overlap: float,
        padding_index: int,
    ) -> None:
        self.single_digit_mnist = single_digit_mnist
        self.max_length = max_length
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.padding_index = padding_index

        self.mnist_digit_dim = 28
        self.samples_by_digit = self._get_samples_by_digit()

    def _get_samples_by_digit(self) -> Dict[int, List]:
        """Stores a collection of images for each digit."""
        samples_by_digit = defaultdict(list)
        for image, digit in self.single_digit_mnist:
            samples_by_digit[digit].append(image.squeeze())
        blank_image = torch.zeros((self.mnist_digit_dim, self.mnist_digit_dim))
        samples_by_digit[-1].append(blank_image)
        return samples_by_digit

    def generate(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main methods to generate a dataset.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Images and labels (padded).
        """
        labels = torch.full((num_samples, self.max_length), self.padding_index)
        images = torch.zeros((num_samples, 32, self.mnist_digit_dim * self.max_length))
        for i in range(num_samples):
            rand_num = self._get_random_number()
            for j, digit in enumerate(str(rand_num)):
                labels[i, j] = int(digit)
            images[i] = self._construct_image_from_number(rand_num)
        return images, labels

    def _get_random_number(self) -> int:
        """Generate a random number.

        The probabiltiy of getting a small number is artifically inflated; otherwise,
        there will be not enough numbers of short lengths and the model will not
        generalize well.
        """
        num_digits_choices = list(range(1, self.max_length + 1))
        probs = [n / sum(num_digits_choices) for n in num_digits_choices]
        num_digits = random.choices(num_digits_choices, weights=probs)[0]
        rand_num = random.randint(
            int("1" + "0" * (num_digits - 1)), int("1" + "0" * num_digits) - 1
        )
        return rand_num

    def _construct_image_from_number(self, number: int) -> torch.Tensor:
        """Concatenate images of single digits."""
        overlap = random.uniform(self.min_overlap, self.max_overlap)
        overlap_width = int(overlap * self.mnist_digit_dim)
        width_increment = self.mnist_digit_dim - overlap_width
        x, y = 0, 2  # Current pointers at x and y coordinates
        digits = self._add_left_and_right_paddings(number)
        multi_digit_image = torch.zeros((32, self.mnist_digit_dim * self.max_length))
        for digit in digits:
            digit_image = random.choice(self.samples_by_digit[digit])
            digit_image = torch.clone(
                digit_image
            )  # To avoid overwriting the original image
            digit_image[:, :overlap_width] = torch.maximum(
                multi_digit_image[y : y + self.mnist_digit_dim, x : x + overlap_width],
                digit_image[:, :overlap_width],
            )
            multi_digit_image[
                y : y + self.mnist_digit_dim, x : x + self.mnist_digit_dim
            ] = digit_image
            x += width_increment
        return multi_digit_image

    def _add_left_and_right_paddings(self, number: int) -> List[int]:
        """Add paddings to left and right of the number."""
        digits = [int(digit) for digit in list(str(number))]
        remanining_length = self.max_length - len(digits)
        left_padding = random.randint(0, remanining_length)
        right_padding = remanining_length - left_padding
        digits = [-1] * left_padding + digits + [-1] * right_padding
        return digits


def split_dataset(dataset: Dataset, fraction: float, seed: int) -> List[Subset]:
    """Split a dataset into two."""
    num_samples = len(dataset)
    split_a_size = int(num_samples * fraction)
    split_b_size = num_samples - split_a_size
    return random_split(
        dataset,
        [split_a_size, split_b_size],
        generator=torch.Generator().manual_seed(seed),
    )
