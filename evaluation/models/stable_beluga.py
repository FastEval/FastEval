from .huggingface import Huggingface


class StableBeluga(Huggingface):
    async def init(self, model_path, *, default_system_message=None, **kwargs):
        if default_system_message is None:
            default_system_message = "You are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal."

        await super().init(
            model_path,
            user="### User:\n",
            assistant="### Assistant:\n",
            system="### System:\n",
            default_system=default_system_message,
            end="\n\n",
            **kwargs,
        )
