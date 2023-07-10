import torch

from .huggingface import Huggingface

class Starchat(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,
            user='<|user|>\n',
            assistant='<|assistant|>',
            system='<|system|>\n',
            default_system='',
            end='<|end|>\n',
            **kwargs,
        )
