# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update

# set it to run as non-interactive
ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# update/upgrade apt
ENV TZ=Europe/Paris
RUN apt upgrade -y

#install basics
RUN apt-get install git -yq
RUN apt-get install curl -yq

#install miniconda
ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh Miniconda3.sh
RUN bash Miniconda3.sh -b -p /miniconda
ENV PATH="/miniconda/bin:${PATH}"
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN rm Miniconda3.sh
RUN conda install wget -y

# create and activate conda environment
RUN conda create -q -n run-environement python="3.11.0" numpy scipy matplotlib
RUN /bin/bash -c "source activate run-environement"
RUN conda install python="3.11.0" pip

# install basics
RUN apt-get update
RUN apt-get install less nano -yq
RUN apt-get -yq install build-essential
RUN apt-get -yq install libsuitesparse-dev
RUN apt-get -yq install wget
RUN apt-get -yq install unzip
RUN apt-get -yq install lsof
RUN apt-get update
RUN apt-get -yq install libsm6 libxrender1 libfontconfig1 libglib2.0-0

# install vLLM essentials
ARG VLLM_VERSION=0.5.2
ARG PYTHON_VERSION=311
RUN pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install vllm-flash-attn
RUN pip install prometheus-eval 

# required packages
# TODO: transform this into requirements.txt
#RUN pip install datasets
#RUN pip install bert-score
#RUN pip install nltk
#RUN pip install -U scikit-learn
#RUN pip install hydra-core --upgrade
#RUN pip install jieba

# install from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt