from http import HTTPStatus

from PIL import ImageStat

import handwritten_digit_string_recognition.utils as utils
from handwritten_digit_string_recognition import Recognizer


model = Recognizer()


def lambda_handler(event, context):
    image = _load_image(event)
    pred_num = model.predict(image)
    image_stat = ImageStat.Stat(image)
    print(f"METRIC image_mean_intensity {image_stat.mean[0]}")
    print(f"INFO pred {pred_num}")
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"prediction": pred_num},
    }


def _load_image(event):
    b64_string = event.get("image")
    if b64_string is None:
        return "no b64_string provided in event"
    return utils.read_b64_image(b64_string, grayscale=True)
