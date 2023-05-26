import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import put_system_message_in_prompter_message

# See https://huggingface.co/timdettmers/guanaco-33b-merged/discussions/4
# for a discussion of the prompt format

class Guanaco:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto').eval()

    def _conversation_item_to_prompt(self, item_type, item):
        if item_type == 'assistant':
            return '### Assistant: ' + item + '\n'
        elif item_type == 'user':
            return '### Human: ' + item + '\n'
        else:
            raise

    def _conversation_to_prompt(self, conversation):
        conversation = put_system_message_in_prompter_message(conversation)
        prompt = ('A chat between a curious human and an artificial intelligence assistant. '
            + "The assistant gives helpful, detailed, and polite answers to the user's questions.\n")
        return prompt + ''.join(self._conversation_item_to_prompt(item_type, item) for item_type, item in conversation) + '### Assistant: '

    def reply(self, conversation):
        prompt = self._conversation_to_prompt(conversation)

        # TODO: max_length should be taken from the model and not hardcoded.
        model_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2047 - 400).to(0)

        # TODO: These parameters are copied from OpenAssistant. Maybe change them for guanaco.
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

        return self.tokenizer.decode(model_output).split('### Assistant: ')[-1].replace(self.tokenizer.eos_token, '').strip()
