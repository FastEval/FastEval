import os
import time
import threading

import openai
import tenacity

from evaluation.constants import NUM_THREADS_OPENAI_GPT3_5, NUM_THREADS_OPENAI_GPT4, DEFAULT_MAX_NEW_TOKENS
from .open_ai_base import OpenAIBase

def print_retry(error):
    print('Got error from OpenAI API. Retrying.', error)

last_rate_limit_errors = {}
check_last_rate_limit_lock = threading.Lock()

class OpenAI(OpenAIBase):
    def __init__(self, model_name, *, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        super().__init__(model_name, max_new_tokens=max_new_tokens)

        if self.model_name.startswith('gpt-3.5-turbo'):
            self.num_threads = NUM_THREADS_OPENAI_GPT3_5
        elif self.model_name.startswith('gpt-4'):
            self.num_threads = NUM_THREADS_OPENAI_GPT4
        else:
            raise Exception('Unknown OpenAI model.')

    @tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=180), stop=tenacity.stop_after_attempt(30), after=print_retry)
    def reply(self, conversation, temperature=None):
        while True:
            while True:
                last_rate_limit_error = last_rate_limit_errors.get(self.model_name, 0)
                now = time.time()

                if now - last_rate_limit_error < 10:
                    time.sleep(10 - (now - last_rate_limit_error))
                else:
                    break

            try:
                return self._reply(
                    conversation=conversation,
                    api_base='https://api.openai.com/v1',
                    api_key=os.environ['OPENAI_API_KEY'],
                    temperature=temperature,
                )
            except openai.error.RateLimitError:
                check_last_rate_limit_lock.acquire()

                last_rate_limit_error = last_rate_limit_errors.get(self.model_name, 0)
                now = time.time()

                last_rate_limit_errors[self.model_name] = now

                check_last_rate_limit_lock.release()

                if now - last_rate_limit_error > 10:
                    print('Encountered OpenAI rate limit for ' + self.model_name + '. Trying again in a few seconds...')
