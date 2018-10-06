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
