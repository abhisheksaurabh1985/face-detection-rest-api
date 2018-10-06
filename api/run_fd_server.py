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

def load_model():
    global model
    print("[INFO] loading model...")
    model = cv2.dnn.readNetFromCaffe(settings.PATH_CAFFE_PROTOTXT_FILE,
                                     settings.PATH_CAFFE_PRETRAINED_MODEL)


def get_image_in_opencv_format(image):
	open_cv_image = np.array(image)
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	return open_cv_image


def detect_face(model, image, min_confidence=0.5, img_width=300, img_height=300):
	confidence_score = []
	bbox_coords = []

	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (img_width, img_height)), 1.0,
		(img_width, img_height), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
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


@app.route("/predict", methods=["POST"])
def predict():

	data = {"success": False}
	# Ensure that the image was properly uploaded to the end point
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# Read the image in PIL format. This will be changed in the OpenCV format later.
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			image = get_image_in_opencv_format(image)

			face_detected_image, confidence_score, bbox_coords = detect_face(model,
			                                                                 image,
			                                                                 min_confidence=settings.MIN_CONFIDENCE,
			                                                                 img_width=settings.IMAGE_WIDTH,
			                                                                 img_height=settings.IMAGE_HEIGHT)
			data["predictions"] = {"confidence_score": confidence_score, "bbox_coords": bbox_coords}
			data["success"] = True

	# Return the data dictionary as JSON object
	return flask.jsonify(data)


if __name__ == "__main__":
	print("[INFO] Loading CNN Face Detection Model and starting the Flask server. \n")
	print("Please wait until the server has fully started.")
	load_model()
	app.run(host='0.0.0.0')
