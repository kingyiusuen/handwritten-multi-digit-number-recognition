from http import HTTPStatus
from typing import Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import ImageStat

import handwritten_digit_string_recognition.utils as utils
from handwritten_digit_string_recognition import Recognizer
from schemas import Image


app = FastAPI(
    title="Handwritten Digit String Recognition",
    description="Recognize handwritten digit string",
    version="0.1",
)


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model():
    """Load model for inference."""
    global model
    model = Recognizer()
    

@app.get("/")
def index():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict")
def predict(request: Request, image: Image) -> Dict:
    """Predict a number from the input image."""
    pil_image = utils.read_b64_image(image.image, grayscale=True)
    pred_num = model.predict(pil_image)
    image_stat = ImageStat.Stat(pil_image)
    print(f"METRIC image_mean_intensity {image_stat.mean[0]}")
    print(f"METRIC image_width {pil_image.width}")
    print(f"METRIC image_height {pil_image.height}")
    print(f"INFO pred {pred_num}")
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"prediction": pred_num},
    }
    return response
