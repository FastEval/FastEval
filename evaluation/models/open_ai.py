import asyncio
import os
import time

from evaluation.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    NUM_THREADS_OPENAI_GPT3_5,
    NUM_THREADS_OPENAI_GPT4,
)
from evaluation.models.open_ai_base import OpenAIBase

last_rate_limit_errors = {}
last_rate_limit_errors_lock = asyncio.Lock()


class OpenAI(OpenAIBase):
    async def init(
        self,
        model_name,
        *,
        default_system_message=None,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS
    ):
        await super().init(model_name, max_new_tokens=max_new_tokens)

        self.default_system_message = default_system_message

        if self.model_name.startswith("gpt-3.5-turbo"):
            self.semaphore = asyncio.Semaphore(NUM_THREADS_OPENAI_GPT3_5)
        elif self.model_name.startswith("gpt-4"):
            self.semaphore = asyncio.Semaphore(NUM_THREADS_OPENAI_GPT4)
        else:
            raise Exception("Unknown OpenAI model.")

    async def reply(self, conversation, *, temperature=None, max_new_tokens=None):
        import openai

        if self.default_system_message is not None and conversation[0][0] != "system":
            conversation.insert(0, ("system", self.default_system_message))

        while True:
            await self.semaphore.acquire()

            while True:
                last_rate_limit_error = last_rate_limit_errors.get(self.model_name, 0)
                now = time.time()
                if now - last_rate_limit_error < 10:
                    await asyncio.sleep(10 - (now - last_rate_limit_error))
                else:
                    break
            try:
                response = await self.reply_two_attempts_with_different_max_new_tokens(
                    conversation=conversation,
                    api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                    api_key=os.environ["OPENAI_API_KEY"],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    too_many_tokens_error=openai.error.InvalidRequestError,
                    get_error_message=lambda error: str(error),
                )

                self.semaphore.release()

                return response
            except openai.error.RateLimitError:
                self.semaphore.release()
                await last_rate_limit_errors_lock.acquire()

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
                self.semaphore.release()
                print("OpenAI server is overloaded or not ready yet. Trying again...")
                await asyncio.sleep(1)
            except openai.error.APIError:
                self.semaphore.release()
                print("Encountered OpenAI APIError. Trying again...")
            except openai.error.Timeout:
                self.semaphore.release()
                print("OpenAI request timeout. Trying again...")
