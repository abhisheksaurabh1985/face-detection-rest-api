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

# Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25
IMAGE_DTYPE = "float32"
