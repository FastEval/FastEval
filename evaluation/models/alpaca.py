from .huggingface import Huggingface

class Alpaca(Huggingface):
    def __init__(self, model_path):
        super().__init__(
            model_path,
            user='### Instruction:\n',
            assistant='### Response:\n',
            end='\n\n',
        )
