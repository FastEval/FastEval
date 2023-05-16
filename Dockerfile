FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

RUN apt update && \
    apt install -y git git-lfs wget vim python3.10 python3.10-venv python3.10-dev

ENV TERM=xterm-256color
RUN git clone https://github.com/juncongmoo/pyllama && \
    pyllama/llama/download_community.sh 30B llama

RUN python3.10 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch==1.13.1 tokenizers==0.13.3 git+https://github.com/huggingface/transformers.git@28f26c107b4a1c5c7e32ed4d9575622da0627a40#egg=transformers[dev-torch] accelerate protobuf

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
RUN source venv/bin/activate && \
    python venv/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir llama --output_dir llama-30b-hf --model_size 30B

RUN git clone --depth 1 https://huggingface.co/OpenAssistant/oasst-sft-7-llama-30b-xor

RUN source venv/bin/activate && \
    python oasst-sft-7-llama-30b-xor/xor_codec.py oasst-sft-7-llama-30b oasst-sft-7-llama-30b-xor/oasst-sft-7-llama-30b-xor llama-30b-hf

RUN source venv/bin/activate && \
    pip install git+https://github.com/tju01/evals.git#egg=evals && \
    pip install --upgrade sacrebleu

ENV PATH=/workspace/venv/bin:$PATH

WORKDIR /workspace

CMD ["/bin/bash"]

# nvidia-docker build -f Dockerfile . -t oasst-openai-evals
# nvidia-docker run -v `pwd`/main.py:/workspace/main.py -v `pwd`/reports:/workspace/reports -it --rm --env CUDA_VISIBLE_DEVICES=0 oasst-openai-evals
