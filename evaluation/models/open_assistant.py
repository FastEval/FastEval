import torch

from .huggingface import Huggingface

class OpenAssistant(Huggingface):
    def __init__(self, model_path):
        super().__init__(
            model_path,
            user='<|prompter|>',
            assistant='<|assistant|>',
            end='tokenizer-eos-token',
        )

    @staticmethod
    def get_dtype(model_path: str):
        if 'pythia' in model_path or 'llama' in model_path:
            return torch.float16
        elif 'falcon' in model_path:
            return torch.bfloat16
        raise
