from .huggingface import Huggingface


class OpenAssistant(Huggingface):
    async def init(self, model_path, *, default_system_message=None, **kwargs):
        if "llama" in model_path.lower():
            eos_token = "</s>"
        elif (
            "pythia" in model_path
            or "falcon" in model_path
            or "starcoder" in model_path
        ):
            eos_token = "<|endoftext|>"
        else:
            raise Exception("This type of OpenAssistant model is not supported yet.")

        if default_system_message is None:
            additional_args = {}
        else:
            additional_args = {
                "system": "<|system|>",
                "default_system": default_system_message,
            }

        await super().init(
            model_path,
            user="<|prompter|>",
            assistant="<|assistant|>",
            end=eos_token,
            **additional_args,
            **kwargs,
        )
