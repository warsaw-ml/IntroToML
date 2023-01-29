import cv2
import requests
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from PIL import Image


def load_lottieurl(url):
    """
    Retrieve the JSON data from a given Lottie animation URL.

    Args:
        url (str): The URL of the Lottie animation.

    Returns:
        dict: The JSON data of the Lottie animation.
    """
    # Make a GET request to the specified URL
    r = requests.get(url)

    # Check the status code of the response
    if r.status_code != 200:
        return None

    # Return the JSON data
    return r.json()


def inference_picture(image, mp_drawing, mp_face_detection, face_detection, model, SIZE=0.1):
    """
    Infers the age of a person in a given image using a model and a face detection library.

    Args:
        image (ndarray): The image to be processed.
        mp_drawing (obj): The object of multiperson drawing.
        mp_face_detection (obj): The object of multiperson face detection.
        face_detection (obj): The object of face detection.
        model (obj): The age inference model.
        SIZE (float): The size of the bounding box.

    Returns:
        ndarray: The image with age inferences added.
    """

    # Get the dimensions of the image
    image_rows, image_cols, _ = image.shape

    # Run face detection on the image
    results = face_detection.process(image)

    # Make the image writable
    image.flags.writeable = True

    # Check if any faces were detected
    if results.detections:
        # print(results.detections)
        for detection in results.detections:
            # Get the bounding box of the face
            relative_bounding_box = detection.location_data.relative_bounding_box
            rect_start_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin,
                relative_bounding_box.ymin,
                image_cols,
                image_rows,
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

                # Crop the image to the bounding box
                cropped_image = image[
                    resized_height_params[0] : resized_height_params[1],
                    resized_width_params[0] : resized_width_params[1],
                ]

                # saving analysed image - used for debugging
                # cv2.imwrite("image.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                # Convert the cropped image to a PIL Image
                face = Image.fromarray(cropped_image)
                face = face.convert("RGB")

                # Run the age inference model on the face
                prediction = model.predict(face)

                # Add the age inference to the image
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

            # Draw the detection on the image
            mp_drawing.draw_detection(image, detection)
    return image
