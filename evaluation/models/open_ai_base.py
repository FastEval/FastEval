def conversation_item_to_openai_format(item_type, item):
    if item_type == 'system':
        return { 'role': 'system', 'content': item }
    if item_type == 'user':
        return { 'role': 'user', 'content': item }
    if item_type == 'assistant':
        return { 'role': 'assistant', 'content': item }
    raise

class OpenAIBase:
    def __init__(self, model_name, *, max_new_tokens):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    async def reply_single_try(self, *, conversation, api_base, api_key, max_new_tokens=None, temperature=None, model_name=None, stop_event):
        import openai

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if temperature is None:
            temperature = 1.0

        if model_name is None:
            model_name = self.model_name

        if stop_event.is_set():
            raise Exception('Stop event is set')

        return await openai.ChatCompletion.acreate(
            api_base=api_base,
            api_key=api_key,

            model=model_name,
            messages=[conversation_item_to_openai_format(item_type, item) for item_type, item in conversation],
            max_tokens=max_new_tokens,

            temperature=temperature,

            # Hardcode default parameters from https://platform.openai.com/docs/api-reference/chat/create
            top_p=1.0,
            presence_penalty=0,
            frequency_penalty=0,
        )['choices'][0]['message']['content']
