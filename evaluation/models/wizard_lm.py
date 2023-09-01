from .huggingface import Huggingface

class WizardLM(Huggingface):
    async def init(self, model_path, **kwargs):
        await super().init(
            model_path,
            prefix="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
            user='USER: ',
            assistant='ASSISTANT: ',
            end=' ',
            **kwargs,
        )
