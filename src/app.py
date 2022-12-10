import cv2
import mediapipe as mp
import requests
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Age Prediction", page_icon=":anger:")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# website title and description
st.title("Real-time age prediction")
st.caption("Click checkbox to start")

# process initialization
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# loading the animation
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    while run:
        st.empty()
        # capture image from the camera
        _, frame = camera.read()

        # convert BGR colours to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image.flags.writeable = True

        if results.detections:
            for detection in results.detections:
                # <code>
                # machine learning things - model prediction for detection and printing age on the image
                # </code>
                mp_drawing.draw_detection(image, detection)

        # show the final image
        FRAME_WINDOW.image(image)
    else:
        # if app is not working - display screensaver
        st_lottie(lottie_coding, height=400)
        st.markdown(
            "<h1 style='text-align: center; color: grey;'>App is turned off</h1>",
            unsafe_allow_html=True,
        )
