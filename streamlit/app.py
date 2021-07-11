import base64
import json
import requests
from io import BytesIO

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image


IMAGE_SCALE_FACTOR = 4


def convert_image_to_b64(canvas_image):
    image_pil = Image.fromarray(canvas_image[:,:,0].astype(np.uint8))
    image_pil = image_pil.convert("L")
    width, height = image_pil.size
    image_pil.resize((width // IMAGE_SCALE_FACTOR, height // IMAGE_SCALE_FACTOR))
    image_file = BytesIO()
    image_pil.save(image_file, "bmp")
    image_file.seek(0)
    image_bytes = image_file.getvalue()
    b64_string = base64.b64encode(image_bytes).decode("utf8")
    image_file.close()
    return b64_string


def get_prediction(b64_string):
    url = "http://localhost:9000/2015-03-31/functions/function/invocations"
    headers = {"Content-type": "application/json"}
    data = json.dumps({"image": f"data:image/bmp;base64,{b64_string}"})
    response = requests.post(url, data=data, headers=headers)
    return response


st.set_page_config(page_title="Handwritten Digit String Recognizer")
st.title("Handwritten Digit String Recognizer")


st.write("Draw some numbers:")
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    background_image=None,
    update_streamlit=False,
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
            response = get_prediction(b64_string)
            prediction = response.json()["data"]["prediction"]
            st.header("Output")
            st.text(prediction)
