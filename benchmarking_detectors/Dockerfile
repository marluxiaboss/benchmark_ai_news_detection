# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04 as base

# Set it to run as non-interactive
ARG DEBIAN_FRONTEND=noninteractive

# Combine update, upgrade, and installations to reduce layers and ensure cleanup
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    git \
    curl \
    less \
    nano \
    build-essential \
    libsuitesparse-dev \
    wget \
    unzip \
    lsof \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and cleanup in one layer
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /miniconda/bin/conda clean -tipy

ENV PATH="/miniconda/bin:${PATH}"

# Combine conda installations
RUN conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda && \
    conda create -q -n run-environment python=3.11.0 numpy scipy matplotlib && \
    conda clean -afy

# Activate conda environment is not effective in Dockerfile, use ENV to specify the path
ENV PATH /miniconda/envs/run-environment/bin:$PATH

# Install vLLM essentials and other requirements from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir https://github.com/vllm-project/vllm/releases/download/v0.5.2/vllm-0.5.2-cp311-cp311-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir vllm-flash-attn prometheus-eval && \
    pip install --no-cache-dir -r requirements.txt

# Clean up pip cache
RUN rm -rf /root/.cache/pip/*

# Set the entrypoint to bash for convenience
ENTRYPOINT ["/bin/bash"]