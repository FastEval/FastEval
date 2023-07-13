import threading
import uuid
import gc
import asyncio

import torch
import transformers
import vllm

import evaluation.utils

lock = threading.Lock()
model = None
vllm_event_loop = None

def unload_model(use_lock=True):
    global model
    global vllm_event_loop

    if use_lock:
        lock.acquire()

    if model is not None:
        model = None
        gc.collect()

    if vllm_event_loop is not None:
        vllm_event_loop.stop()
        vllm_event_loop = None

    if use_lock:
        lock.release()

def execute_vllm_requests():
    global vllm_event_loop

    assert vllm_event_loop is None
    vllm_event_loop = asyncio.new_event_loop()
    vllm_event_loop.run_forever()

def create_model(*, model_path, tokenizer_path, dtype):
    engine = vllm.AsyncLLMEngine.from_engine_args(vllm.AsyncEngineArgs(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=str(dtype).replace('torch.', ''),
        disable_log_requests=True,
        trust_remote_code=True,
    ))

    executor_thread = threading.Thread(target=execute_vllm_requests)
    executor_thread.start()

    return { 'engine': engine, 'executor_thread': executor_thread }

async def vllm_respond_to_prompt(*, prompt, prompt_model, temperature):
    assert prompt_model is model

    if temperature is None:
        temperature = 1.0

    response_generator = prompt_model['model']['engine'].generate(prompt, vllm.SamplingParams(
        # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

        best_of=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        use_beam_search=False,
        max_tokens=prompt_model['max_new_tokens'],
    ), request_id=uuid.uuid4())

    response = None
    async for response_part in response_generator:
        if not response_part.finished:
            continue
        assert response is None
        outputs = response_part.outputs
        assert len(outputs) == 1
        response = outputs[0].text

    return response.replace(prompt_model['eos_token'], '')

def run_inference(*, prompt, tokenizer_path, model_path, dtype, max_new_tokens, temperature):
    global model

    lock.acquire()

    evaluation.utils.switch_gpu_model_type('vllm')

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
            'max_new_tokens': max_new_tokens,
            'eos_token': transformers.AutoTokenizer.from_pretrained(tokenizer_path).eos_token,
            'model': create_model(model_path=model_path, tokenizer_path=tokenizer_path, dtype=dtype),
        }

    future = asyncio.run_coroutine_threadsafe(vllm_respond_to_prompt(prompt=prompt, prompt_model=model, temperature=temperature), vllm_event_loop)
    lock.release()
    return future.result()
