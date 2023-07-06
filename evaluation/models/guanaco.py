from .huggingface import Huggingface

class Guanaco(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,

            # https://huggingface.co/timdettmers/guanaco-65b-merged/discussions/1
            # I don't get that error but an error about maximum recursion depth exceeded though.
            tokenizer_path='TheBloke/guanaco-65B-HF',

            # https://huggingface.co/timdettmers/guanaco-33b-merged/discussions/4
            prefix=('A chat between a curious human and an artificial intelligence assistant. '
                + "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"),
            user='### Human: ',
            assistant='### Assistant: ',
            end='\n',

            **kwargs,
        )
