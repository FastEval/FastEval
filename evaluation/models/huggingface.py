import transformers

import evaluation.models.models
import evaluation.models.huggingface_backends.hf_transformers
import evaluation.models.huggingface_backends.vllm
import evaluation.models.huggingface_backends.tgi
from evaluation.models.utils import put_system_message_in_prompter_message
from evaluation.constants import NUM_THREADS_LOCAL_MODEL, DEFAULT_MAX_NEW_TOKENS

def get_max_batch_size(model_path, max_new_tokens):
    # TODO: Check amount of GPU ram, check model size, dtype & estimate how much RAM model takes.
    # Then estimate how much ram is needed for the tokens and estimate the batch size we can fit.

    # Currently disabled. Performance benefit is questionable...
    return 1

class Huggingface:
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path=None,
        prefix='',
        user: str,
        assistant: str,
        system=None,
        default_system='',
        end: str,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    ):
        self.model_path = model_path

        self.tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

        self.eos_token = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path).eos_token

        self.prefix = prefix
        self.user = user
        self.assistant = assistant
        self.system = system
        self.default_system = default_system
        self.end = end

        self.max_new_tokens = max_new_tokens

        self.dtype = evaluation.models.models.get_dtype(model_path)
        self.backend = evaluation.models.models.get_huggingface_backend(model_path)

        if self.backend == 'vllm' or self.backend == 'tgi':
            self.num_threads = NUM_THREADS_LOCAL_MODEL
        elif self.backend == 'hf_transformers':
            self.num_threads = get_max_batch_size(model_path, max_new_tokens)
        else:
            raise

    def _conversation_to_prompt(self, conversation):
        if self.system is None:
            conversation = put_system_message_in_prompter_message(conversation)
        prompt = self.prefix
        if self.system is not None and conversation[0][0] != 'system':
            conversation.insert(0, ('system', self.default_system))
        for item_type, item in conversation:
            if item_type == 'assistant':
                prompt += self.assistant + item + self.end
            elif item_type == 'user':
                prompt += self.user + item + self.end
            elif item_type == 'system':
                prompt += self.system + item + self.end
            else:
                raise
        prompt += self.assistant
        return prompt.strip()

    def reply(self, conversation, temperature=None):
        common_kwargs = {
            'prompt': self._conversation_to_prompt(conversation),
            'tokenizer_path': self.tokenizer_path,
            'model_path': self.model_path,
            'dtype': self.dtype,
            'max_new_tokens': self.max_new_tokens,
            'temperature': temperature,
        }

        if self.backend == 'vllm':
            response = evaluation.models.huggingface_backends.vllm.run_inference(**common_kwargs)
        elif self.backend == 'tgi':
            response = evaluation.models.huggingface_backends.tgi.run_inference(**common_kwargs)
        elif self.backend == 'hf_transformers':
            response = evaluation.models.huggingface_backends.hf_transformers.run_inference(**common_kwargs, max_batch_size=self.num_threads)
        else:
            raise

        # Some models continue to simulate the user and further assistant conversation
        response = response.split(self.user)[0]

        final_substrings_to_remove = []
        for special_token in [self.end, self.eos_token]:
            final_substrings_to_remove += [special_token, special_token.replace('\n', ''),
                special_token.replace('\n', '').strip(), special_token.strip()]
        final_substrings_to_remove.append('\n')
        final_substrings_to_remove.append(' ')

        final_substrings_to_remove = [substring for substring in final_substrings_to_remove if substring != '']

        final_substrings_to_remove_unique = []
        for substring in final_substrings_to_remove:
            if substring not in final_substrings_to_remove_unique:
                final_substrings_to_remove_unique.append(substring)

        while True:
            for substring in final_substrings_to_remove_unique:
                if response.endswith(substring):
                    response = response[:-len(substring)]
                    break
            else:
                break

        response = response.lstrip('\n')
        response = response.lstrip()

        return response
