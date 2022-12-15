import cv2
import mediapipe as mp
import requests
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

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
        image_rows, image_cols, _ = frame.shape

        # convert BGR colours to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image.flags.writeable = True

        if results.detections:
            for detection in results.detections:
                relative_bounding_box = detection.location_data.relative_bounding_box
                print(relative_bounding_box)
                rect_start_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin,
                    relative_bounding_box.ymin,
                    image_cols,
                    image_rows
                )
                rect_end_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height,
                    image_cols,
                    image_rows
                )
                cropped_image = image[rect_start_point[1]:rect_end_point[1], rect_start_point[0]:rect_end_point[0]]
                cv2.imwrite("image.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                print(rect_start_point)
                print(rect_end_point)
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
