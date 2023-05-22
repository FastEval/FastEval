import openai
import tenacity

class OpenAI:
    def __init__(self, model_name):
        self.model_name = model_name

    def _conversation_item_to_openai_format(self, item_type, item):
        if item_type == 'system':
            return { 'role': 'system', 'content': item }
        if item_type == 'user':
            return { 'role': 'user', 'content': item }
        if item_type == 'assistant':
            return { 'role': 'assistant', 'content': item }
        raise

    @tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(6))
    def reply(self, conversation):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=[self._conversation_item_to_openai_format(item_type, item) for item_type, item in conversation],
            max_tokens=1024,
        )['choices'][0]['message']['content']
