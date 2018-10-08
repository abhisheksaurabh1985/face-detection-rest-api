## Flask Rest API for Face Detection in Production Using Docker, NGINX and Gunicorn

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
Output contains the confidence score for each face detection. The bounding box coordinates are the coordinates of the corresponding faces. Length of any of these lists equals the number of face detected.
