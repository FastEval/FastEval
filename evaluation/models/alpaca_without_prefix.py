from .huggingface import Huggingface


class AlpacaWithoutPrefix(Huggingface):
    async def init(self, model_path, **kwargs):
        await super().init(
            model_path,
            user="### Instruction:\n",
            assistant="### Response:\n",
            end="\n\n",
            **kwargs,
        )
