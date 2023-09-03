from .huggingface import Huggingface


class AlpacaWithPrefix(Huggingface):
    async def init(self, model_path, **kwargs):
        await super().init(
            model_path,
            prefix="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
            user="### Instruction:\n",
            assistant="### Response:\n",
            end="\n\n",
            **kwargs,
        )
