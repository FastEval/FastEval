from .huggingface import Huggingface

class Guanaco(Huggingface):
    async def init(self, model_path, **kwargs):
        await super().init(
            model_path,

            # https://huggingface.co/timdettmers/guanaco-33b-merged/discussions/4
            prefix=('A chat between a curious human and an artificial intelligence assistant. '
                + "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"),
            user='### Human: ',
            assistant='### Assistant: ',
            end='\n',

            **kwargs,
        )
