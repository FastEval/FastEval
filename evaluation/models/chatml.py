import torch

from .huggingface import Huggingface

class ChatML(Huggingface):
    def __init__(self, model_path, **kwargs):
        # https://github.com/openai/openai-python/blob/main/chatml.md
        super().__init__(
            model_path,
            user='<|im_start|>user\n',
            assistant='<|im_start|>assistant\n',
            system='<|im_start|>system\n',
            # https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py
            default_system='A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.',
            end='<|im_end|>\n',
            **kwargs,
        )
