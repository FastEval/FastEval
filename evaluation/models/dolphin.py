from .huggingface import Huggingface

class Dolphin(Huggingface):
    async def init(self, model_path, *, default_system_message=None, **kwargs):
        if default_system_message is None:
            default_system_message = 'You are a helpful assistant chatbot that answers the users questions.'

        await super().init(
            model_path,
            user='USER: ',
            assistant='ASSISTANT: ',
            system='SYSTEM: ',
            default_system=default_system_message,
            end='\n',
            **kwargs,
        )
