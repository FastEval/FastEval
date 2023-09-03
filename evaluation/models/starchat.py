from .huggingface import Huggingface


class Starchat(Huggingface):
    def __init__(self, model_path, *, default_system_message="", **kwargs):
        super().__init__(
            model_path,
            user="<|user|>\n",
            assistant="<|assistant|>",
            system="<|system|>\n",
            default_system=default_system_message,
            end="<|end|>\n",
            **kwargs,
        )
