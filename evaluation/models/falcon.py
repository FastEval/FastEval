import torch

from .huggingface import Huggingface

class FalconInstruct(Huggingface):
    def __init__(self, model_path):
        super().__init__(
            model_path,

            dtype=torch.bfloat16,

            # https://huggingface.co/tiiuae/falcon-7b-instruct/discussions/1#64708b0a3df93fddece002a4
            user='User: ',
            assistant='Assistant: ',
            end='\n',
        )
