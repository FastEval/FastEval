from .huggingface import Huggingface

class ChatML(Huggingface):
    def __init__(self, model_path, **kwargs):
        if model_path == 'mosaicml/mpt-7b-chat':
            # https://github.com/mosaicml/llm-foundry/blob/a936df02bb65cdac2279c84bc17465cc6cbd196a/scripts/inference/hf_chat.py#L52
            default_system = ('    - You are a helpful assistant chatbot trained by MosaicML.\n'
                + '    - You answer questions.\n'
                + '    - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n'
                + '    - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.')
        elif model_path == 'mosaicml/mpt-30b-chat':
            # https://github.com/mosaicml/llm-foundry/blob/1f54d26b75b60a8f84a7dc087e130833bf5fc42d/scripts/inference/hf_chat.py#L34
            default_system = 'A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.'
        else:
            raise

        # https://github.com/openai/openai-python/blob/main/chatml.md
        super().__init__(
            model_path,
            user='<|im_start|>user\n',
            assistant='<|im_start|>assistant\n',
            system='<|im_start|>system\n',
            default_system=default_system,
            end='<|im_end|>\n',
            **kwargs,
        )
