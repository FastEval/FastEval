import transformers

from .huggingface import Huggingface

class Llama2Chat(Huggingface):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, user=None, assistant=None, end=None)
        self.tokenizer_with_eos = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path, add_bos_token=True, add_eos_token=True)
        self.tokenizer_without_eos = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path, add_bos_token=True, add_eos_token=True)

    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py
    def _conversation_to_prompt(self, conversation):
        default_system = ('You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n'
            + "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.")

        instruction_start = '[INST]'
        instruction_end = '[/INST]'

        system_start = '<<SYS>>\n'
        system_end = '\n<</SYS>>\n\n'

        if conversation[0][0] != 'system':
            conversation.insert(0, ('system', default_system))

        conversation = [
            (conversation[1][0], system_start + conversation[0][1] + system_end + conversation[1][1]),
        ] + conversation[2:]

        prompt_items = []
        for item_type, item in conversation:
            if item_type == 'user' or item_type == 'system':
                prompt_items.append(instruction_start + ' ' + item.strip() + ' ' + instruction_end)
            elif item_type == 'assistant':
                prompt_items[-1] += ' ' + item.strip() + ' '

        prompt_tokens = []
        for prompt_item in prompt_items[:-1]:
            prompt_tokens += self.tokenizer_with_eos.encode(prompt_item)
        prompt_tokens += self.tokenizer_without_eos.encode(prompt_items[-1])

        return ('tokens', prompt_tokens)
