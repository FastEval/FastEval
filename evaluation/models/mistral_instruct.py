from .huggingface import Huggingface


class MistralInstruct(Huggingface):
    async def init(self, model_path, **kwargs):
        await super().init(
            model_path,
            user="[INST] ",
            assistant="",
            end=" [/INST]",
            end2="</s> ",
            **kwargs,
        )
