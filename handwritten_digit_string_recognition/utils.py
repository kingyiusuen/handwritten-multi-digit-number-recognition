# Refernce:
# https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/util.py

import base64
from io import BytesIO
from pathlib import Path
from typing import Union

import smart_open
from PIL import Image


def read_image_pil_file(
    image_file: Union[BytesIO, Path, str], grayscale: bool = False
) -> Image:
    """Returns a Pillow Image."""
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image


def read_image_pil(image_uri: Union[Path, str], grayscale: bool = False) -> Image:
    """Read images from URL or local filesystem."""
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_b64_image(b64_string: str, grayscale: bool = False) -> Image:
    """Load base64-encoded images."""
    try:
        _, b64_data = b64_string.split(",")
        image_file = BytesIO(base64.b64decode(b64_data))
        return read_image_pil_file(image_file, grayscale)
    except Exception as exception:
        raise ValueError(
            f"Could not load image from b64 {b64_string}: {exception}"
        ) from exception
