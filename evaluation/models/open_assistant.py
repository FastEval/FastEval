import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import put_system_message_in_prompter_message

class OpenAssistant:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto').eval()

    def _conversation_item_to_prompt(self, item_type, item):
        if item_type == 'assistant':
            return '<|assistant|>' + item + self.tokenizer.eos_token
        elif item_type == 'user':
            return '<|prompter|>' + item + self.tokenizer.eos_token
        else:
            raise

    def _conversation_to_prompt(self, conversation):
        conversation = put_system_message_in_prompter_message(conversation)
        return ''.join(self._conversation_item_to_prompt(item_type, item) for item_type, item in conversation) + '<|assistant|>'

    def reply(self, conversation):
        prompt = self._conversation_to_prompt(conversation)

        # TODO: max_length should be taken from the model and not hardcoded.
        model_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2047 - 400).to(0)

        if 'token_type_ids' in model_input:
            del model_input['token_type_ids']

        model_output = self.model.generate(
            **model_input,
            min_new_tokens=1,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.2,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]

        return self.tokenizer.decode(model_output).split('<|assistant|>')[-1].replace(self.tokenizer.eos_token, '').strip()
