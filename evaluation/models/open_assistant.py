import torch

from .huggingface import Huggingface

class OpenAssistant(Huggingface):
    def __init__(self, model_path, **kwargs):
        if 'llama' in model_path:
            eos_token = '</s>'
        elif 'pythia' in model_path or 'falcon' in model_path:
            eos_token = '<|endoftext|>'
        else:
            raise

        super().__init__(
            model_path,
            user='<|prompter|>',
            assistant='<|assistant|>',
            end=eos_token,
            **kwargs,
        )
