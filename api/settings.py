import cv2

# Model related settings
PATH_CAFFE_PROTOTXT_FILE = "/home/project/face_detection_REST_API/model/deploy.prototxt.txt"
PATH_CAFFE_PRETRAINED_MODEL = "./model/res10_300x300_ssd_iter_140000.caffemodel"
MIN_CONFIDENCE = 0.5
# model = cv2.dnn.readNetFromCaffe(PATH_CAFFE_PROTOTXT_FILE,
#                                  PATH_CAFFE_PRETRAINED_MODEL)

# Image related settings
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
