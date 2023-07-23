from .huggingface import Huggingface

class Dolphin(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,
            user='USER: ',
            assistant='ASSISTANT: ',
            system='SYSTEM: ',
            default_system='You are a helpful assistant chatbot that answers the users questions.',
            end='\n',
            **kwargs,
        )
