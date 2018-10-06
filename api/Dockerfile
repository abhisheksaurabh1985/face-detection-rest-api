FROM ubuntu:16.04
# Built on https://github.com/waleedka/modern-deep-learning-docker/blob/master/Dockerfile
MAINTAINER Abhishek Saurabh <abhishek_saurabh001@yahoo.co.in>


# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev

#
# Python 3.5
#
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow
# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image pandas matplotlib Cython requests flask

#
# Tensorflow 1.6.0 - CPU
#
RUN pip3 install --no-cache-dir --upgrade tensorflow

# Expose port for TensorBoard
EXPOSE 6006

#
# OpenCV 3.4.1
#
# Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall

# RUN apt-get install -y --no-install-recommends \
#    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
#    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
#    liblapacke-dev checkinstall

# Get source from github
RUN git clone -b 3.4.1 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          .. && \
    make -j"$(nproc)" && \
    make install

#
# Caffe
#
# Dependencies
RUN apt-get install -y --no-install-recommends \
    cmake libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev \
    libhdf5-serial-dev protobuf-compiler liblmdb-dev libgoogle-glog-dev \
    libboost-all-dev && \
    pip3 install lmdb
# Get source. Use master branch because the latest stable release (rc3) misses critical fixes.
RUN git clone -b master --depth 1 https://github.com/BVLC/caffe.git /usr/local/src/caffe
# Python dependencies
RUN pip3 --no-cache-dir install -r /usr/local/src/caffe/python/requirements.txt
# Compile
RUN cd /usr/local/src/caffe && mkdir build && cd build && \
    cmake -D CPU_ONLY=ON -D python_version=3 -D BLAS=open -D USE_OPENCV=ON .. && \
    make -j"$(nproc)" all && \
    make install
# Enivronment variables
ENV PYTHONPATH=/usr/local/src/caffe/python:$PYTHONPATH \
	PATH=/usr/local/src/caffe/build/tools:$PATH
# Fix: old version of python-dateutil breaks caffe. Update it.
RUN pip3 install --no-cache-dir python-dateutil --upgrade

#
# Java
#
# Install JDK (Java Development Kit), which includes JRE (Java Runtime
# Environment). Or, if you just want to run Java apps, you can install
# JRE only using: apt install default-jre
RUN apt-get install -y --no-install-recommends default-jdk

#
# Keras 2.1.5
#
RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras

#
# PyTorch 0.3.1
#
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl && \
    pip3 install torchvision

#
# PyCocoTools
#
# Using a fork of the original that has a fix for Python 3.
# I submitted a PR to the original repo (https://github.com/cocodataset/cocoapi/pull/50)
# but it doesn't seem to be active anymore.
RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

# imutils and dlib
# Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems.
RUN pip3 install imutils
RUN pip3 install dlib

# flask, redis, gunicorn
RUN pip3 install redis
RUN pip3 install gunicorn

# make directories suited to your application
RUN mkdir -p /home/project/face_detection_REST_API
# RUN mkdir -p /home/project/face_detection_REST_API/model
WORKDIR /home/project/face_detection_REST_API

# copy and install packages for flask
# ADD requirements.txt /home/project/face_detection_REST_API
# RUN pip3 install --no-cache-dir -r requirements.txt

# copy contents from your local to your docker container
# COPY . /home/project/face_detection_REST_API
# ADD ./model /home/project/face_detection_REST_API/
COPY . .
# RUN pip3 install --no-cache-dir -r requirements.txt
