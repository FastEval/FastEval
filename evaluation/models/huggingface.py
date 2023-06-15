import threading

import torch
import transformers

from .utils import put_system_message_in_prompter_message

pipeline = None
pipeline_lock = threading.Lock()

def conversation_item_to_prompt(*, user, assistant, end, item_type, item):
    if item_type == 'assistant':
        return assistant + item + end
    elif item_type == 'user':
        return user + item + end
    else:
        raise

def conversation_to_prompt(*, conversation, prefix, user, assistant, end):
    conversation = put_system_message_in_prompter_message(conversation)
    return prefix + ''.join(conversation_item_to_prompt(item_type=item_type, item=item, user=user, assistant=assistant, end=end)
        for item_type, item in conversation) + assistant

def run_pipeline(*, tokenizer_path, model_path, dtype, conversation, user, assistant, end, prefix):
    global pipeline

    pipeline_lock.acquire()

    if (pipeline is None
            or pipeline['tokenizer_path'] != tokenizer_path
            or pipeline['model_path'] != model_path
            or pipeline['dtype'] != dtype):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        pipeline = {
            'tokenizer_path': tokenizer_path,
            'model_path': model_path,
            'dtype': dtype,
            'tokenizer': tokenizer,
            'pipeline': transformers.pipeline(
                'text-generation',
                model=model_path,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map='auto'
            ),
        }

    if end == 'tokenizer-eos-token':
        end = pipeline['tokenizer'].eos_token

    prompt = conversation_to_prompt(conversation=conversation, prefix=prefix, user=user, assistant=assistant, end=end)

    response = pipeline['pipeline'](prompt, max_new_tokens=400, do_sample=True, num_return_sequences=1, eos_token_id=pipeline['tokenizer'].eos_token_id)
    response = response[0]['generated_text'][len(prompt):]
    response = response.split(user)[0] # some models continue to simulate the user and further assistant conversation
    response = response.strip()

    pipeline_lock.release()
    return response

class Huggingface:
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path=None,
        prefix='',
        user: str,
        assistant: str,
        end: str,
    ):
        if tokenizer_path is None:
            tokenizer_path = model_path

        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.dtype = self.__class__.get_dtype(model_path)

        self.prefix = prefix
        self.user = user
        self.assistant = assistant
        self.end = end

    @staticmethod
    def get_dtype(model_path: str):
        return torch.float16

    def reply(self, conversation):
        return run_pipeline(tokenizer_path=self.tokenizer_path, model_path=self.model_path, dtype=self.dtype,
            conversation=conversation, user=self.user, assistant=self.assistant, prefix=self.prefix, end=self.end)
