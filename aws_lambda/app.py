import logging

import handwritten_multi_digit_number_recognition.utils as utils
from handwritten_multi_digit_number_recognition import Recognizer


logger = logging.getLogger()
logger.setLevel(logging.INFO)


model = Recognizer()


def lambda_handler(event, context):
    b64_string = event.get("image")

    if b64_string is None:
        return {
            "status-code": 400,
            "message": "Invalid request. Missing input.",
        }

    try:
        pil_image = utils.read_b64_image(b64_string, grayscale=True)
    except Exception as e:
        return {
            "status-code": 400,
            "message": "Invalid request. Image cannot be decoded.",
        }

    try:
        pred_num = model.predict(pil_image)
        return {
            "status-code": 200,
            "message": "Success",
            "data": {"prediction": pred_num},
        }
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return {
            "status-code": 500,
            "message": "Unhandled error",
        }
