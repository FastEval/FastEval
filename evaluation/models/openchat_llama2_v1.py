from .huggingface import Huggingface

class OpenchatLlama2V1(Huggingface):
    async def init(self, model_path, **kwargs):
        await super().init(
            model_path,
            user='User: ',
            assistant='Assistant: ',
            end='<|end_of_turn|>',
            **kwargs,
        )
