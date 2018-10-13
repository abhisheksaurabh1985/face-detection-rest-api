### Flask Rest API for Face Detection in Production Using Docker, NGINX and Gunicorn

See the accompanying blog post for full details:
[Face Detection Rest API in Production using Docker, NGINX and Flask](https://medium.com/p/1f205633d2d6/edit)

The goal of this project is to explain how can we deploy a scalable deep learning model in production using Docker and NGINX. Focus is _NOT_ on the model itself.

That said, the face detection model from `OpenCV` used here performs reasonably well. It performs well even when the face is occluded but usually fails when the face is not well illuminated. More specifically, I've used the face detection model in the `dnn` module of `OpenCV`. A bit of detail on the model can be found on [this](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) PyImageSearch blog. The model itself consists of two files which are as follows:
1. `deploy.prototxt.txt` which defines the model architecture and,
2. `res10_300x300_ssd_iter_140000.caffemodel` which contains the weights of the actual layers.


#### Steps to run
1. Open a terminal and run `docker-compose up --build`. Wait until the following output shows up on the terminal.
```
Attaching to api, nginx
api      | [2018-10-08 17:31:36 +0000] [1] [INFO] Starting gunicorn 19.9.0
api      | [2018-10-08 17:31:36 +0000] [1] [INFO] Listening at: http://0.0.0.0:8000 (1)
api      | [2018-10-08 17:31:36 +0000] [1] [INFO] Using worker: sync
api      | [2018-10-08 17:31:36 +0000] [9] [INFO] Booting worker with pid: 9
```
2. Using __curl__: In another terminal, run the following command:
```
curl -X POST -i http://0.0.0.0:8000/predict -F 'image=@/home/abhishek/Desktop/Workspace/practice/face_detection/iron_chic.jpg'
```

Following output will show up:

```
HTTP/1.1 100 Continue

HTTP/1.1 200 OK
Server: gunicorn/19.9.0
Date: Mon, 08 Oct 2018 17:34:18 GMT
Connection: close
Content-Type: application/json
Content-Length: 138

{"predictions":{"bbox_coords":[[256,51,431,278],[134,133,266,327],[11,146,145,347]],"confidence_score":[1.0,0.998,0.607]},"success":true}
```
3.Using python `request` module: `cd` into `api` and run `python demo_request.py`. Following shall be the output:
```
('Face bounding box coordinates and the corresponding confidence scores: \n', {u'bbox_coords': [[256, 51, 431, 278], [134, 133, 266, 327], [11, 146, 145, 347]], u'confidence_score': [1.0, 0.998, 0.607]})
```

#### Output description
```
{"predictions":{"bbox_coords":[[256,51,431,278],[134,133,266,327],[11,146,145,347]],"confidence_score":[1.0,0.998,0.607]},"success":true}
```
- Output contains the confidence score for each face detection.
- The bounding box coordinates are the coordinates of the corresponding faces. The coordinates are in the following order: `(top_left_x_coordinate, top_left_y_coordinate, bottom_right_x_coordinate, bottom_right_y_coordinate).`
- Length of any of these lists equals the number of face detected.


#### TODO
1. Handling of incoming requests in a batch using `Redis`
2. Stress test the API
