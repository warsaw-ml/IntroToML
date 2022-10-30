import cv2
import requests
import streamlit as st

from streamlit_lottie import st_lottie
from PIL import Image

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
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

while run:
    st.empty()
    # capture image from the camera
    _, frame = camera.read()

    # convert BGR colours to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # <code>
    # machine learning things
    # </code>

    # show the final image
    FRAME_WINDOW.image(frame)
else:
    st_lottie(lottie_coding, height=400)
    st.title('App stopped - click "Run" to start')