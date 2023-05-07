FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

RUN apt update && \
    apt install -y git git-lfs python3.10 python3.10-venv python3.10-dev && \
    python3.10 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch==1.13.1 transformers

ENV PATH=/workspace/venv/bin:$PATH

WORKDIR /workspace

CMD ["/bin/bash"]

# nvidia-docker build -f Dockerfile . -t oasst-openai-evals
# nvidia-docker run -v `pwd`/main.py:/workspace/main.py -v `pwd`/openai-evals:/workspace/openai-evals -v `pwd`/runs:/workspace/runs -it --rm --env CUDA_VISIBLE_DEVICES=0 oasst-openai-evals
