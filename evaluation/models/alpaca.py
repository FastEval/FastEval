import torch
import transformers

from .utils import put_system_message_in_prompter_message

class Alpaca:
    def __init__(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline('text-generation', model=model_path, tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')

    def _conversation_item_to_prompt(self, item_type, item):
        if item_type == 'assistant':
            return '### Response:\n' + item + '\n\n'
        elif item_type == 'user':
            return '### Instruction:\n' + item + '\n\n'
        else:
            raise

    def _conversation_to_prompt(self, conversation):
        conversation = put_system_message_in_prompter_message(conversation)
        return ''.join(self._conversation_item_to_prompt(item_type, item) for item_type, item in conversation) + '### Response:\n'

    def reply(self, conversation):
        prompt = self._conversation_to_prompt(conversation)
        model_output = self.pipeline(prompt, max_new_tokens=400, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)[0]
        return model_output['generated_text'].split('### Response:\n')[-1].replace(self.tokenizer.eos_token, '').strip()
