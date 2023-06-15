import threading
import uuid
import time

import torch
import transformers

from .utils import put_system_message_in_prompter_message

lock = threading.Lock()
pipeline = None
current_batch = []

def conversation_to_prompt(*, conversation, prefix, user, assistant, end):
    conversation = put_system_message_in_prompter_message(conversation)
    prompt = prefix
    for item_type, item in conversation:
        if item_type == 'assistant':
            prompt += assistant + item + end
        elif item_type == 'user':
            prompt += user + item + end
        else:
            raise
    prompt += assistant
    return prompt

def process_current_batch():
    global current_batch

    time.sleep(0.05)

    lock.acquire()

    if len(current_batch) == 0:
        lock.release()
        return

    # We just store a reference to the pipeline here to make sure that the prompt is evaluated with the correct model
    # It could theoretically happen (in the rest of the code), that while we are waiting for some responses to be computed,
    # the underlying model should be changed. This is not something that should currently happen since we evaluate one
    # model at a time and wait for all the responses before switching the model.
    # But just to prevent future possible bugs, we make really sure that we have the correct model here.
    for batch_item in current_batch:
        assert batch_item['pipeline'] is pipeline

    responses = pipeline['pipeline']([batch_item['prompt'] for batch_item in current_batch],
        max_new_tokens=400, do_sample=True, num_return_sequences=1, eos_token_id=pipeline['tokenizer'].eos_token_id)
    for i in range(len(current_batch)):
        response = responses[i][0]['generated_text'][len(current_batch[i]['prompt']):]
        response = response.split(pipeline['user'])[0] # some models continue to simulate the user and further assistant conversation
        response = response.strip()
        current_batch[i]['response'] = response

    for item in current_batch:
        item['obtained_response'] = False
        with item['condition']:
            item['condition'].notify_all()

    while not all(item['obtained_response'] for item in current_batch):
        time.sleep(0.01)

    current_batch = []

    lock.release()

def wait_for_response(condition):
    thread = threading.Thread(target=process_current_batch)
    thread.start()

    with condition:
        condition.wait()
    for item in current_batch:
        if item['condition'] == condition:
            response = item['response']
            item['obtained_response'] = True
            return response

def run_pipeline(*, tokenizer_path, model_path, dtype, conversation, user, assistant, end, prefix):
    global pipeline

    lock.acquire()

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
            'user': user,
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

    condition = threading.Condition()
    current_batch.append({ 'prompt': prompt, 'condition': condition, 'pipeline': pipeline })

    lock.release()
    return wait_for_response(condition)

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
