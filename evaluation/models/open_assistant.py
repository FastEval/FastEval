import torch

from .huggingface import Huggingface

class OpenAssistant(Huggingface):
    def __init__(self, model_path):
        if 'pythia' in model_path:
            dtype = torch.float16
        elif 'llama' in model_path:
            dtype = torch.float16
        elif 'falcon' in model_path:
            dtype = torch.bfloat16
        else:
            raise

        super().__init__(
            model_path,

            dtype=dtype,

            user='<|prompter|>',
            assistant='<|assistant|>',
            end='tokenizer-eos-token',
        )
