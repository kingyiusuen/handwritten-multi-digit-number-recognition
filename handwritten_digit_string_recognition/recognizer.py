from pathlib import Path
from typing import Union

import torch
import torchvision.transforms as transforms
from PIL import Image

from . import utils
from .lit_models import CTCLitModel


MODEL_CKPT_FILENAME = Path(__file__).resolve().parents[1] / "artifacts" / "model.pt"


class Recognizer:
    def __init__(self):
        self.transform = transforms.ToTensor()
        model = CTCLitModel.load_from_checkpoint(MODEL_CKPT_FILENAME)
        model.freeze()
        self.scripted_model = model.to_torchscript()

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        if isinstance(image, Image.Image):
            image_pil = image
        else:
            image_pil = utils.read_image_pil(image, grayscale=True)
        image_tensor = self.transform(image_pil)
        pred_num = self.scripted_model(image_tensor.unsqueeze(0))[0]
        return pred_num
