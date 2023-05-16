FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

RUN apt update && \
    apt install -y git git-lfs wget vim python3.10 python3.10-venv python3.10-dev

RUN python3.10 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch==1.13.1 tokenizers transformers accelerate protobuf

RUN source venv/bin/activate && \
    pip install git+https://github.com/tju01/evals.git#egg=evals && \
    pip install --upgrade sacrebleu

ENV PATH=/workspace/venv/bin:$PATH

WORKDIR /workspace

CMD ["/bin/bash"]

# nvidia-docker build -f Dockerfile . -t oasst-openai-evals
# nvidia-docker run -v `pwd`/main.py:/workspace/main.py -v `pwd`/reports:/workspace/reports -it --rm --env CUDA_VISIBLE_DEVICES=0 oasst-openai-evals
