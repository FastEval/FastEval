from .huggingface import Huggingface

class OpenchatLlama2V1(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,
            user='User: ',
            assistant='Assistant: ',
            end='<|end_of_turn|>',
            **kwargs,
        )
