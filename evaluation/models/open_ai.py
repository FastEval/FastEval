import openai

class OpenAI:
    def __init__(self, model_name):
        self.model_name = model_name

    def reply(self, question, system_message='You are a helpful assistant.'):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                { 'role': 'system', 'content': system_message },
                { 'role': 'user', 'content': question },
            ],
            max_tokens=1024,
        )['choices'][0]['message']['content']
