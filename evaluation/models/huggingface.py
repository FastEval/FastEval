import threading
import uuid
import time
import gc

import torch
import transformers
import vllm

from .utils import put_system_message_in_prompter_message
import evaluation.utils

lock = threading.Lock()
model = None
current_batch = []

def unload_model(use_lock=True):
    global model

    if use_lock:
        lock.acquire()

    if model is not None:
        model = None
        gc.collect()

    if use_lock:
        lock.release()

def process_current_batch():
    global current_batch

    time.sleep(0.05)

    lock.acquire()

    if len(current_batch) == 0:
        lock.release()
        return

    # We just store a reference to the model here to make sure that the prompt is evaluated with the correct model
    # It could theoretically happen (in the rest of the code), that while we are waiting for some responses to be computed,
    # the underlying model should be changed. This is not something that should currently happen since we evaluate one
    # model at a time and wait for all the responses before switching the model.
    # But just to prevent future possible bugs, we make really sure that we have the correct model here.
    for batch_item in current_batch:
        assert batch_item['model'] is model

    prompts = [batch_item['prompt'] for batch_item in current_batch]

    if model['use_vllm']:
        responses = model['model']['model'].generate(prompts, vllm.SamplingParams(
            # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

            best_of=None,
            presence_penalty = 0.0,
            frequency_penalty=0.0,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            use_beam_search=False,
            max_tokens=model['max_new_tokens'],
        ), use_tqdm=False)

        responses = [response.outputs[0].text.replace(model['model']['eos_token'], '') for response in responses]
    else:
        responses = model['model'](
            prompts,

            # See the following link for more details & a list of the parameters
            # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig

            # Parameters that control the length of the output
            max_new_tokens=model['max_new_tokens'],
            min_new_tokens=1,

            # Parameters that control the generation strategy used
            do_sample=True,
            num_beams=1,

            # Parameters for manipulation of the model output logits
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            typical_p=1.0,
            epsilon_cutoff=0.0,
            eta_cutoff=0.0,
            diversity_penalty=0.0,
            repetition_penalty=1.0,
            encoder_repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
            renormalize_logits=False,

            # Special tokens that can be used at generation time
            eos_token_id=model['tokenizer'].eos_token_id,
        )

        responses = [responses[i][0]['generated_text'][len(current_batch[i]['prompt']):] for response in responses]

    for i in range(len(current_batch)):
        current_batch[i]['response'] = responses[i]

    for item in current_batch:
        item['obtained_response'] = False
        with item['condition']:
            item['condition'].notify_all()

    while not all(item['obtained_response'] for item in current_batch):
        time.sleep(0.01)

    current_batch = []

    lock.release()

def wait_for_response(condition):
    thread = threading.Thread(target=process_current_batch)
    thread.start()

    with condition:
        condition.wait()
    for item in current_batch:
        if item['condition'] == condition:
            response = item['response']
            item['obtained_response'] = True
            return response

def create_model(*, model_path, tokenizer_path, dtype, use_vllm):
    if use_vllm:
        return {
            'eos_token': transformers.AutoTokenizer.from_pretrained(tokenizer_path).eos_token,
            'model': vllm.LLM(
                model=model_path,
                tokenizer=tokenizer_path,
                tensor_parallel_size=torch.cuda.device_count(),
                dtype=str(dtype).replace('torch.', ''),
            )
        }
    else:
        return transformers.pipeline(
            'text-generation',
            model=model_path,
            tokenizer=transformers.AutoTokenizer.from_pretrained(tokenizer_path),
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map='auto'
        )

def run_inference(*, prompt, tokenizer_path, model_path, dtype, max_new_tokens, use_vllm):
    global model

    lock.acquire()

    evaluation.utils.switch_gpu_model_type('huggingface')

    if (model is None
            or model['tokenizer_path'] != tokenizer_path
            or model['model_path'] != model_path
            or model['dtype'] != dtype
            or model['max_new_tokens'] != max_new_tokens):
        unload_model(False)
        model = {
            'tokenizer_path': tokenizer_path,
            'model_path': model_path,
            'dtype': dtype,
            'model': create_model(model_path=model_path, tokenizer_path=tokenizer_path, dtype=dtype, use_vllm=use_vllm),
            'use_vllm': use_vllm,
            'max_new_tokens': max_new_tokens
        }

    condition = threading.Condition()
    current_batch.append({ 'prompt': prompt, 'condition': condition, 'model': model })

    lock.release()
    return wait_for_response(condition)

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
        max_new_tokens=400,
        use_vllm=True,
    ):
        if tokenizer_path is None:
            tokenizer_path = model_path

        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.dtype = self.__class__.get_dtype(model_path)

        self.prefix = prefix
        self.user = user
        self.assistant = assistant
        self.system = system
        self.default_system = default_system
        self.end = end

        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm

    @staticmethod
    def get_dtype(model_path: str):
        return torch.float16

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

    def reply(self, conversation):
        prompt = self._conversation_to_prompt(conversation)
        response = run_inference(prompt=prompt, tokenizer_path=self.tokenizer_path, model_path=self.model_path, dtype=self.dtype,
            max_new_tokens=self.max_new_tokens, use_vllm=self.use_vllm)
        response = response.split(self.user)[0] # some models continue to simulate the user and further assistant conversation
        return response.strip()
