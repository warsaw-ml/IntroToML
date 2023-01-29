import os

import cv2
import mediapipe as mp
import pyrootutils
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)

from src.utils.functions import inference_picture, load_lottieurl
from src.utils.predict import Predict


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Age Prediction", page_icon=":anger:")


# website title and description
st.title("Real-time age prediction")

tab1, tab2 = st.tabs(["Live", "Photos"])

with tab1:

    st.caption("Click checkbox to start")
    # process initialization
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    SIZE = 0.1
    pred = {}
    # loading the animation
    lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        model = Predict()
        while run:
            st.empty()
            # capture image from the camera
            _, frame = camera.read()
            image_rows, image_cols, _ = frame.shape

            # convert BGR colours to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = inference_picture(image, mp_drawing, mp_face_detection, face_detection, model, SIZE=0.1)

            # show the final image
            FRAME_WINDOW.image(image)
        else:
            # if app is not working - display screensaver
            st_lottie(lottie_coding, height=400)
            st.markdown(
                "<h1 style='text-align: center; color: grey;'>App is turned off</h1>",
                unsafe_allow_html=True,
            )

with tab2:

    output_folder = "combined_images"
    os.makedirs(output_folder, exist_ok=True)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        model = Predict()
        st.empty()
        file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if file != []:
            i = 0
            for file in file:
                i += 1
                import numpy as np

                image = Image.open(file)

                image = np.array(image)

                image = inference_picture(
                    image,
                    mp_drawing,
                    mp_face_detection,
                    face_detection,
                    model,
                    SIZE=0.1,
                )

                output_path = os.path.join(output_folder, f"image_{i}.png")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_path, image)

            st.success("Done! Images are saved in the folder 'combined_images'.")
