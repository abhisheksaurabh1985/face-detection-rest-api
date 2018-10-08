# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
FACE_DETECTION_REST_API_URL = "http://0.0.0.0:8000/predict"
IMAGE_PATH = "../iron_chic.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(FACE_DETECTION_REST_API_URL,
                  files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    print("Face bounding box coordinates and the corresponding confidence scores: \n", r["predictions"])
# otherwise, the request failed
else:
    print("Request failed")
