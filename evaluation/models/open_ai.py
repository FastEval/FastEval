import os
import threading
import time

from evaluation.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    NUM_THREADS_OPENAI_GPT3_5,
    NUM_THREADS_OPENAI_GPT4,
)
from evaluation.models.open_ai_base import OpenAIBase

last_rate_limit_errors = {}
last_rate_limit_errors_lock = threading.Lock()


class OpenAI(OpenAIBase):
    def __init__(
        self,
        model_name,
        *,
        default_system_message=None,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS
    ):
        super().__init__(model_name, max_new_tokens=max_new_tokens)

        self.default_system_message = default_system_message

        if self.model_name.startswith("gpt-3.5-turbo"):
            self.num_threads = NUM_THREADS_OPENAI_GPT3_5
        elif self.model_name.startswith("gpt-4"):
            self.num_threads = NUM_THREADS_OPENAI_GPT4
        else:
            raise Exception("Unknown OpenAI model.")

    def reply(self, conversation, *, temperature=None, max_new_tokens=None, stop_event):
        import openai

        if self.default_system_message is not None and conversation[0][0] != "system":
            conversation.insert(0, ("system", self.default_system_message))

        while True:
            while True:
                last_rate_limit_error = last_rate_limit_errors.get(self.model_name, 0)
                now = time.time()
                if now - last_rate_limit_error < 10:
                    time.sleep(10 - (now - last_rate_limit_error))
                else:
                    break
            try:
                return self.reply_single_try(
                    conversation=conversation,
                    api_base="https://api.openai.com/v1",
                    api_key=os.environ["OPENAI_API_KEY"],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    stop_event=stop_event,
                )
            except openai.error.RateLimitError:
                last_rate_limit_errors_lock.acquire()

                last_rate_limit_error = last_rate_limit_errors.get(self.model_name, 0)
                now = time.time()

                last_rate_limit_errors[self.model_name] = now

                last_rate_limit_errors_lock.release()

                if now - last_rate_limit_error > 10:
                    print(
                        "Encountered OpenAI rate limit for "
                        + self.model_name
                        + ". Trying again in a few seconds..."
                    )
            except openai.error.ServiceUnavailableError:
                print("OpenAI server is overloaded or not ready yet. Trying again...")
                time.sleep(1)
            except openai.error.APIError:
                print("Encountered OpenAI APIError. Trying again...")
            except openai.error.Timeout:
                print("OpenAI request timeout. Trying again...")
