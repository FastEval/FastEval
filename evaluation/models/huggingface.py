import torch
import transformers

from .utils import put_system_message_in_prompter_message

class Huggingface:
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path=None,
        dtype=torch.float16,
        prefix: str,
        user: str,
        assistant: str,
        end: str,
    ):
        if tokenizer_path is None:
            tokenizer_path = model_path

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        self.pipeline = transformers.pipeline('text-generation', model=model_path, tokenizer=self.tokenizer,
            torch_dtype=dtype, trust_remote_code=True, device_map='auto')

        self.prefix = prefix
        self.user = user
        self.assistant = assistant
        self.end = end

    def _conversation_item_to_prompt(self, item_type, item):
        if item_type == 'assistant':
            return self.assistant + item + self.end
        elif item_type == 'user':
            return self.user + item + self.end
        else:
            raise

    def _conversation_to_prompt(self, conversation):
        conversation = put_system_message_in_prompter_message(conversation)
        return self.prefix + ''.join(self._conversation_item_to_prompt(item_type, item) for item_type, item in conversation) + self.assistant

    def reply(self, conversation):
        prompt = self._conversation_to_prompt(conversation)
        response = self.pipeline(prompt, max_new_tokens=400, do_sample=True, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        response = response[0]['generated_text'][len(prompt):]
        response = response.split(self.user)[0] # some models continue to simulate the user and further assistant conversation
        response = response.strip()
        return response
