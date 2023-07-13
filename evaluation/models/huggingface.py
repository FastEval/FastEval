from .utils import put_system_message_in_prompter_message
import evaluation.utils
import evaluation.models.huggingface_backends.hf_transformers
import evaluation.models.huggingface_backends.vllm
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

        self.prefix = prefix
        self.user = user
        self.assistant = assistant
        self.system = system
        self.default_system = default_system
        self.end = end

        self.max_new_tokens = max_new_tokens

        self.dtype = evaluation.utils.get_dtype(model_path)
        self.use_vllm = evaluation.utils.is_vllm_supported(model_path)

        if self.use_vllm:
            self.num_threads = NUM_THREADS_LOCAL_MODEL
        else:
            self.num_threads = get_max_batch_size(model_path, max_new_tokens)

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

        if self.use_vllm:
            response = evaluation.models.huggingface_backends.vllm.run_inference(**common_kwargs)
        else:
            response = evaluation.models.huggingface_backends.hf_transformers.run_inference(**common_kwargs, max_batch_size=self.num_threads)

        response = response.split(self.user)[0] # some models continue to simulate the user and further assistant conversation
        response = response.strip()
        return response
