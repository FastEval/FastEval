from .huggingface import Huggingface

class Alpaca(Huggingface):
    def __init__(self, model_path):
        super().__init__(
            model_path,

            # https://huggingface.co/tiiuae/falcon-7b-instruct/discussions/1#64708b0a3df93fddece002a4
            user='### Instruction:\n',
            assistant='### Response:\n',
            end='\n\n',
        )
