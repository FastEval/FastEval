import os
import threading

import evaluation.args
import evaluation.models.models
import evaluation.models.huggingface_backends.hf_transformers
import evaluation.models.huggingface_backends.vllm_backend
import evaluation.models.huggingface_backends.tgi
from evaluation.models.utils import put_system_message_in_user_message
from evaluation.constants import NUM_THREADS_LOCAL_MODEL, DEFAULT_MAX_NEW_TOKENS

eos_tokens = {}
eos_tokens_lock = threading.Lock()

class Huggingface:
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer=None,
        prefix='',
        user: str,
        assistant: str,
        system=None,
        default_system='',
        end: str,
        end2=None,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        dtype=None,
        inference_backend: str,
    ):
        import torch

        self.model_path = model_path

        self.tokenizer_path = model_path if tokenizer is None else tokenizer

        self.prefix = prefix
        self.user = user
        self.assistant = assistant
        self.system = system
        self.default_system = default_system
        self.end = end

        if end2 is None:
            end2 = end
        self.end2 = end2

        self.max_new_tokens = max_new_tokens

        if dtype is None:
            self.dtype = evaluation.models.models.get_dtype(model_path)
        else:
            dtypes = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32,
            }

            self.dtype = dtypes[dtype]

        self.backend = inference_backend

        self.num_threads = NUM_THREADS_LOCAL_MODEL

    def _get_eos_token(self):
        if hasattr(self, 'eos_token'):
            return self.eos_token

        import transformers

        eos_tokens_lock.acquire()
        eos_tokens[self.tokenizer_path] = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path).eos_token
        self.eos_token = eos_tokens[self.tokenizer_path]
        eos_tokens_lock.release()

        return self.eos_token

    def conversation_to_prompt(self, conversation):
        if self.system is None:
            conversation = put_system_message_in_user_message(conversation)
        prompt = self.prefix
        if self.system is not None and conversation[0][0] != 'system' and self.default_system is not None:
            conversation.insert(0, ('system', self.default_system))
        for item_type, item in conversation:
            if item_type == 'assistant':
                prompt += self.assistant + item + self.end2
            elif item_type == 'user':
                prompt += self.user + item + self.end
            elif item_type == 'system':
                prompt += self.system + item + self.end
            else:
                raise
        prompt += self.assistant
        return prompt.strip()

    def reply(self, conversation, *, temperature=None, max_new_tokens=None, stop_event):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        common_kwargs = {
            'prompt': self.conversation_to_prompt(conversation),
            'tokenizer_path': self.tokenizer_path,
            'model_path': self.model_path,
            'dtype': self.dtype,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'stop_event': stop_event,
        }

        if isinstance(common_kwargs['prompt'], tuple) and self.backend not in ['hf_transformers', 'vllm']:
            raise Exception('Only the HF transformers & vLLM backends currently support using tokens instead of text.')

        if self.backend == 'vllm':
            response = evaluation.models.huggingface_backends.vllm_backend.run_inference(**common_kwargs)
        elif self.backend == 'tgi':
            response = evaluation.models.huggingface_backends.tgi.run_inference(**common_kwargs)
        elif self.backend == 'hf_transformers':
            # The batch size can be increased (should work), but batching doesn't seem to increase performance
            response = evaluation.models.huggingface_backends.hf_transformers.run_inference(**common_kwargs, max_batch_size=1)
        else:
            raise

        # Some models continue to simulate the user and further assistant conversation
        if self.user is not None:
            response = response.split(self.user)[0]

        special_tokens = []
        if self.end2 is not None:
            special_tokens.append(self.end2)
        if self._get_eos_token() is not None:
            special_tokens.append(self._get_eos_token())

        final_substrings_to_remove = []
        for special_token in special_tokens:
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
