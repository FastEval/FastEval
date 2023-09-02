from .huggingface import Huggingface


class WizardLM(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,
            prefix="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
            user="USER: ",
            assistant="ASSISTANT: ",
            end=" ",
            **kwargs,
        )
