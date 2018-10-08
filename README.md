##### Consuming the Face Detection API
  1. Through `curl`
```python
curl -X POST -F image=@../rooster.jpg 'http://localhost:5000/predict'
curl -X POST -F image=@../iron_chic.jpg 'http://localhost:5000/predict'
```
  2. The script `demo_request.py` handles the request and response to and from the API. It makes use of the `request` package in Python. To run this script, first ensure that the Flask web server is running (i.e. run `run_fd_server.py`). Then in a separate shell run `demo_request.py`.


##### Current status
Suitable for single threaded use with no concurrent requests.

##### TODO
1. Use `Redis` to handle multiple requests


##### Errors and solutions
1. __docker-compose up doesn't rebuild image although Dockerfile has changed__
`docker-compose up --build`. `docker-compose up --force-recreate` didn't work.

2. GET request works but the POST throws an error. Model is not loading.  
```bash
curl -X GET -i 'http://0.0.0.0:8000/'

curl -X POST -i 'http://0.0.0.0:8000/predict' --data 'image=@/home/abhishek/Desktop/Workspace/practice/face_detection/iron_chic.jpg'

curl -X POST -i 'http://0.0.0.0:8000/predict' -F 'image=@/home/abhishek/Desktop/Workspace/practice/face_detection/iron_chic.jpg'

curl -X POST -i http://0.0.0.0:8000/predict -F 'image=@/home/abhishek/Desktop/Workspace/practice/face_detection/iron_chic.jpg'
```

```bash
abhishek@abhishek-HP-EliteBook-840-G3:~/Desktop/Workspace/practice/face_detection/face_detection_rest_API$ docker-compose ps
Name               Command               State                Ports               
---------------------------------------------------------------------------------
api     gunicorn -w 1 -b :8000 wsg ...   Up      6006/tcp, 0.0.0.0:8000->8000/tcp
nginx   nginx -g daemon off;             Up      80/tcp, 0.0.0.0:8001->8001/tcp
```
