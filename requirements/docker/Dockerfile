FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

RUN apt update
RUN apt install -y python3.11 python3-pip
RUN echo "alias python=python3" >> ~/.bashrc

RUN mkdir /home/tmp
COPY ./requirements/requirements.txt /home/tmp/
WORKDIR /home/tmp
RUN pip3 install -r requirements.txt
WORKDIR /
RUN rm -rf /home/tmp

