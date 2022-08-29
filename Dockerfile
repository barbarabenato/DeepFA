FROM ubuntu:18.04


WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y \
    apt-utils build-essential git vim swig libatlas-base-dev libopenblas-dev

## install python 3.7
RUN apt-get install --no-install-recommends -y \
    python3.8 python3-pip python3.8-dev python3-setuptools

RUN git clone https://github.com/barbarabenato/DeepFA.git

RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --upgrade pip

RUN python -m pip install cython numpy

RUN cd /app/DeepFA/ && python -m pip install dist/pyift-0.1-cp38-cp38-linux_x86_64.whl

#RUN python -m pip install .

#SHELL ["/bin/bash", "-c"]

