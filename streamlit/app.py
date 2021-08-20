import base64
import json
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit as st


API_ENDPOINT = (
    "https://8ql3q4h9je.execute-api.us-east-1.amazonaws.com/v1/predict-number"
)
IMAGE_SCALE_FACTOR = 4


def convert_image_to_b64(canvas_image):
    image_pil = Image.fromarray(canvas_image.astype(np.uint8))
    image_pil = image_pil.convert("L")
    width, height = image_pil.size
    image_pil = image_pil.resize(
        (width // IMAGE_SCALE_FACTOR, height // IMAGE_SCALE_FACTOR)
    )
    image_file = BytesIO()
    image_pil.save(image_file, "png")
    image_file.seek(0)
    image_bytes = image_file.getvalue()
    b64_string = base64.b64encode(image_bytes).decode("utf8")
    image_file.close()
    return b64_string


def get_prediction(b64_string):
    headers = {
        "Content-type": "application/json",
        "x-api-key": st.secrets["api_key"],
    }
    data = json.dumps({"image": f"data:image/png;base64,{b64_string}"})
    response = requests.post(API_ENDPOINT, data=data, headers=headers)
    return response


def main():
    st.set_page_config(page_title="Handwritten Multi-Digit Number Recognizer")
    st.title("Handwritten Multi-Digit Number Recognizer")

    st.write("Draw some numbers:")
    canvas_result = st_canvas(
        stroke_width=8,
        stroke_color="white",
        background_color="black",
        background_image=None,
        update_streamlit=True,
        height=128,
        width=560,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Submit"):
        if canvas_result.image_data is None:
            st.error("Please draw something before clicking the submit button.")
        else:
            with st.spinner("Wait for it..."):
                b64_string = convert_image_to_b64(canvas_result.image_data)
                response = get_prediction(b64_string).json()
                if "status-code" not in response:
                    st.error("Internal Server Error")
                elif response["status-code"] != 200:
                    st.error(response["message"])
                else:
                    prediction = response["data"]["prediction"]
                    st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
