from pathlib import Path
from typing import Union

import torch
import torchvision.transforms as transforms
from PIL import Image

from . import utils
from .lit_models import CTCLitModel


MODEL_CKPT_FILENAME = Path(__file__).resolve().parents[1] / "artifacts" / "model.pt"


class Recognizer:
    """Model used for production."""

    def __init__(self):
        self.transform = transforms.ToTensor()
        self.model = CTCLitModel.load_from_checkpoint(MODEL_CKPT_FILENAME)
        self.model.freeze()

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict the number in the input image.

        Args:
            image: can be a path to the image file or an instance of Pillow image.

        Returns:
            The predicted number. "None" if the model cannot detect any number.
        """
        if isinstance(image, Image.Image):
            image_pil = image
        else:
            image_pil = utils.read_image_pil(image, grayscale=True)
        image_tensor = self.transform(image_pil)
        decoded, pred_lengths = self.model(image_tensor.unsqueeze(0))
        # Remove the paddings
        digit_lists = decoded[0][: pred_lengths[0]]
        if len(digit_lists) == 0:
            return "None"
        # Concatenate the digits
        pred_num = "".join(str(i) for i in digit_lists.tolist())
        return pred_num
