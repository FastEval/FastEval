import os

import openai
import tenacity

from evaluation.constants import NUM_THREADS_OPENAI, DEFAULT_MAX_NEW_TOKENS

def print_retry(error):
    print('Got error from OpenAI API. Retrying.', error)

class OpenAI:
    num_threads = NUM_THREADS_OPENAI

    def __init__(self, model_name, *, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def _conversation_item_to_openai_format(self, item_type, item):
        if item_type == 'system':
            return { 'role': 'system', 'content': item }
        if item_type == 'user':
            return { 'role': 'user', 'content': item }
        if item_type == 'assistant':
            return { 'role': 'assistant', 'content': item }
        raise

    def _reply(self, *, conversation, model_name, api_base, api_key, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        return openai.ChatCompletion.create(
            api_base=api_base,
            api_key=api_key,

            model=model_name,
            messages=[self._conversation_item_to_openai_format(item_type, item) for item_type, item in conversation],
            max_tokens=max_new_tokens,

            # Hardcode default parameters from https://platform.openai.com/docs/api-reference/chat/create
            temperature=1.0,
            top_p=1.0,
            presence_penalty=0,
            frequency_penalty=0,
        )['choices'][0]['message']['content']

    @tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=180), stop=tenacity.stop_after_attempt(30), after=print_retry)
    def reply(self, conversation):
        return self._reply(
            conversation=conversation,
            model_name=self.model_name,
            api_base='https://api.openai.com/v1',
            api_key=os.environ['OPENAI_API_KEY'],
        )
