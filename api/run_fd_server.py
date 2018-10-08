import imutils
from imutils import face_utils
import cv2
import dlib
import numpy as np
import flask
from PIL import Image
import io
import json
import settings


# Initialize the flask application and the face detection model
app = flask.Flask(__name__)
model = None
print("Model before loading: ", type(model))
model = cv2.dnn.readNetFromCaffe("./model/deploy.prototxt.txt",
                                 "./model/res10_300x300_ssd_iter_140000.caffemodel")
print("Model after loading: ", type(model))

def get_image_in_opencv_format(image):
	"""
	Converts image from PIL to OpenCV format.
	"""
	open_cv_image = np.array(image)
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	return open_cv_image


def detect_face(model, image, min_confidence=0.5, img_width=300, img_height=300):
	"""
	Returns the image with a bounding box around each face and a confidence score
	for the face detected. Works with multiple faces.

	: param net: Model which is to be used for face detection.
	: param image: Image in which face is to be detected.
	: min_confidence: Minimum confidence for the model to identify a region as a
	face. Default is 0.5.

	: return confidence_score: List with the confidence score for each face detected.
	E.g. confidence:  [0.9997495, 0.9973845, 0.8715456]
	: return bbox_coords: List with the coordinates of top left and bottom right corner of
	the bounding box for all the faces. Coordinates of each face is in a tuple.
	E.g. box:  [[262, 58, 434, 273], [136, 126, 261, 323], [7, 149, 141, 350]]
	"""
	confidence_score = []
	bbox_coords = []

	(h, w) = image.shape[:2]
	print("h: ", h)
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (img_width, img_height)), 1.0,
		(img_width, img_height), (104.0, 177.0, 123.0))
	print("blob: ", blob)
	print("type(blob): ", type(blob))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	print("type(model): ", type(model))
	model.setInput(blob)
	detections = model.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > min_confidence:
			confidence_score.append(round(float(confidence), 3))
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			[startX, startY, endX, endY] = box.astype("int")
			bbox_coords.append([int(startX), int(startY), int(endX), int(endY)])

			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	return image, confidence_score, bbox_coords


@app.route("/")
def homepage():
	return "Welcome to the Face Detection REST API!"


@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	print("flask.request.method: ", flask.request.method)
	# Ensure that the image was properly uploaded to the end point
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# Read the image in PIL format. This will be changed in the OpenCV format later.
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = get_image_in_opencv_format(image)
			print("Image converted into OpenCV format")
			face_detected_image, confidence_score, bbox_coords = detect_face(model,
			                                                                 image,
			                                                                 min_confidence=settings.MIN_CONFIDENCE,
			                                                                 img_width=settings.IMAGE_WIDTH,
			                                                                 img_height=settings.IMAGE_HEIGHT)
			data["predictions"] = {"confidence_score": confidence_score, "bbox_coords": bbox_coords}
			data["success"] = True
			print("***********  data ***********: \n", data)
	# Return the data dictionary as JSON object
	return flask.jsonify(data)


if __name__ == "__main__":
	print("[INFO] Loading CNN Face Detection Model and starting the Flask server. \n")
	print("Please wait until the server has fully started.")
	load_model()
	# app.debug = True
	app.run(host='0.0.0.0')
