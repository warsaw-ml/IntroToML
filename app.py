import cv2
import streamlit as st

# website title and description 
st.title("Real-time age prediction")
st.caption("Click checkbox to start")

# process initialization
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)


while run:
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
    st.write('Stopped')