FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# PyTorch CUDA 12.8
# ------------------------------------------------------------
RUN python3 -m pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------
# Qwen Image 2.1 Turbo dependencies
#
# Official Turbo requires current Diffusers support for the
# checkpoint-provided sigma schedule.
# ------------------------------------------------------------
RUN python3 -m pip install \
    "git+https://github.com/huggingface/diffusers.git" \
    "transformers>=5.17.0" \
    accelerate \
    safetensors \
    pillow \
    huggingface_hub \
    runpod

# ------------------------------------------------------------
# Copy RunPod handler
# ------------------------------------------------------------
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
