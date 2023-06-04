import torch
import transformers

from .utils import put_system_message_in_prompter_message

class OpenAssistant:
    def __init__(self, model_path):
        if 'falcon' in model_path:
            self.base_model_type = 'falcon'
            dtype = torch.bfloat16
        elif 'llama' in model_path:
            self.base_model_type = 'llama'
            dtype = torch.float16
        else:
            raise

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline('text-generation', model=model_path, tokenizer=self.tokenizer,
            torch_dtype=dtype, trust_remote_code=True, device_map='auto')

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

        if self.base_model_type == 'llama':
            kwargs = { 'temperature': 0.8, 'repetition_penalty': 1.2, 'top_p': 0.9 }
        elif self.base_model_type == 'falcon':
            kwargs = { 'top_k': 10 }

        model_output = self.pipeline(
            prompt,
            min_new_tokens=1,
            max_new_tokens=400,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )[0]

        return model_output['generated_text'].split('<|assistant|>')[-1].replace(self.tokenizer.eos_token, '').strip()
