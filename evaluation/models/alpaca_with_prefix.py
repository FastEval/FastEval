from .huggingface import Huggingface

class AlpacaWithPrefix(Huggingface):
    def __init__(self, model_path):
        super().__init__(
            model_path,
            prefix='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
            user='### Instruction:\n',
            assistant='### Response:\n',
            end='\n\n',
        )
