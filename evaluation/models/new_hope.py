from .alpaca_without_prefix import AlpacaWithoutPrefix

class NewHope(AlpacaWithoutPrefix):
    def __init__(self, model_path, **kwargs):
        super().__init__(
            model_path,
            end2='</s><s> ',
            **kwargs,
        )
