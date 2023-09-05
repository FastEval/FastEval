import re

def conversation_item_to_openai_format(item_type, item):
    if item_type == "system":
        return {"role": "system", "content": item}
    if item_type == "user":
        return {"role": "user", "content": item}
    if item_type == "assistant":
        return {"role": "assistant", "content": item}
    raise


class OpenAIBase:
    async def init(self, model_name, *, max_new_tokens):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    async def reply_single_try(
        self,
        *,
        conversation,
        api_base,
        api_key,
        max_new_tokens=None,
        temperature=None,
        model_name=None
    ):
        import openai

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if temperature is None:
            temperature = 1.0

        if model_name is None:
            model_name = self.model_name

        return (
            await openai.ChatCompletion.acreate(
                api_base=api_base,
                api_key=api_key,
                model=model_name,
                messages=[
                    conversation_item_to_openai_format(item_type, item)
                    for item_type, item in conversation
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                # Hardcode default parameters from https://platform.openai.com/docs/api-reference/chat/create
                top_p=1.0,
                presence_penalty=0,
                frequency_penalty=0,
            )
        )["choices"][0]["message"]["content"]

    async def reply_two_attempts_with_different_max_new_tokens(self, *, too_many_tokens_error, get_error_message, max_new_tokens, **kwargs):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        try:
            return await self.reply_single_try(**kwargs, max_new_tokens=max_new_tokens)
        except too_many_tokens_error as error:
            error_message = get_error_message(error)
            error_information = re.search(
                "This model's maximum context length is ([0-9]+) tokens\. "
                + "However, you requested ([0-9]+) tokens \([0-9]+ in the messages, [0-9]+ in the completion\)\. "
                + "Please reduce the length of the messages or completion\.",
                error_message,
            )
            if error_information is None:
                raise Exception("OpenAI API Error: " + error_message)
            maximum_context_length = int(error_information.group(1))
            request_total_length = int(error_information.group(2))
            num_tokens_too_much = request_total_length - maximum_context_length
            reduced_max_new_tokens = max_new_tokens - num_tokens_too_much
            if reduced_max_new_tokens <= 0:
                return ""
            return await self.reply_single_try(**kwargs, max_new_tokens=reduced_max_new_tokens)
