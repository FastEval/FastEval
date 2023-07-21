from .huggingface import Huggingface

class FreeWilly2(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,
            user='### User:\n',
            assistant='### Assistant:\n',
            system='### System:\n',
            default_system="You are Free Willy, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.",
            end='\n\n',
            **kwargs,
        )
