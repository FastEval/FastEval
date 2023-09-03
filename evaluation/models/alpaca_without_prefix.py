from .huggingface import Huggingface


class AlpacaWithoutPrefix(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,
            user="### Instruction:\n",
            assistant="### Response:\n",
            end="\n\n",
            **kwargs,
        )
