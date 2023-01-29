import pyrootutils
import cv2
import mediapipe as mp
import requests
import streamlit as st
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from PIL import Image
from streamlit_lottie import st_lottie

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.predict import Predict


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
SIZE = 0.1
pred = {}

# loading the animation
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    model = Predict()

    # run the prediction loop
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
            # print(results.detections)
            for detection in results.detections:

                relative_bounding_box = detection.location_data.relative_bounding_box
                rect_start_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols, image_rows
                )
                rect_end_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height,
                    image_cols,
                    image_rows,
                )

                if rect_start_point is not None and rect_end_point is not None:
                    width = rect_end_point[0] - rect_start_point[0]
                    height = rect_end_point[1] - rect_start_point[1]
                    resized_width_params = (
                        int(rect_start_point[0] - width * SIZE),
                        int(rect_end_point[0] + width * SIZE),
                    )

                    resized_height_params = (
                        int(rect_start_point[1] - height * SIZE),
                        int(rect_end_point[1] + height * SIZE),
                    )

                    image_height, image_width, _ = image.shape
                    resized_width_params = (
                        max(0, resized_width_params[0]),
                        min(image_width, resized_width_params[1]),
                    )
                    resized_height_params = (
                        max(0, resized_height_params[0]),
                        min(image_height, resized_height_params[1]),
                    )

                    cropped_image = image[
                        resized_height_params[0] : resized_height_params[1],
                        resized_width_params[0] : resized_width_params[1],
                    ]

                    cv2.imwrite("image.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                    face = Image.fromarray(cropped_image)
                    face = face.convert("RGB")

                    prediction = model.predict(face)
                    print(prediction)
                    image = cv2.putText(
                        image,
                        f"Age: {int(prediction)}",
                        (rect_start_point[0], rect_start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

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
