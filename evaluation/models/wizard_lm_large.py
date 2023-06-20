import torch

from .huggingface import Huggingface

class WizardLMLarge(Huggingface):
    def __init__(self, model_path):
        super().__init__(
            model_path,

            # https://github.com/nlpxucan/WizardLM
            prefix=('A chat between a curious user and an artificial intelligence assistant. '
                + "The assistant gives helpful, detailed, and polite answers to the user's questions. "),
            user='USER: ',
            assistant='ASSISTANT: ',
            end=' ',
        )

    @staticmethod
    def get_dtype(model_path: str):
        if '30B' in model_path:
            return torch.float32
