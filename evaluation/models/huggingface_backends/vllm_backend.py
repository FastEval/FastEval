import threading
import uuid
import gc
import asyncio

import torch
import transformers
import vllm

import evaluation.models.models

lock = threading.Lock()
model = None

def unload_model(use_lock=True):
    global model

    if use_lock:
        lock.acquire()

    if model is None:
        if use_lock:
            lock.release()
        return

    model['event_loop'].stop()
    model = None
    gc.collect()

    if use_lock:
        lock.release()

def load_model(*, model_path, tokenizer_path, dtype):
    global model

    event_loop = asyncio.new_event_loop()

    engine = vllm.AsyncLLMEngine.from_engine_args(vllm.AsyncEngineArgs(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=str(dtype).replace('torch.', ''),
        disable_log_requests=True,
        trust_remote_code=True,
    ))

    condition = model

    model = {
        'tokenizer_path': tokenizer_path,
        'model_path': model_path,
        'dtype': dtype,
        'engine': engine,
        'event_loop': event_loop,
    }

    with condition:
        condition.notify_all()

    event_loop.run_forever()

def load_model_in_separate_thread(*, model_path, tokenizer_path, dtype):
    global model

    model = threading.Condition()

    model_thread = threading.Thread(target=load_model, kwargs={
        'model_path': model_path,
        'tokenizer_path': tokenizer_path,
        'dtype': dtype,
    })

    model_thread.start()

async def respond_to_prompt(*, prompt, prompt_model, temperature, max_new_tokens):
    assert prompt_model is model

    if temperature is None:
        temperature = 1.0

    if isinstance(prompt, tuple):
        if prompt[0] != 'tokens':
            raise Exception('Unknown prompt type')
        args = { 'prompt_token_ids': prompt[1], 'prompt': None }
    else:
        args = { 'prompt': prompt }

    response_generator = prompt_model['engine'].generate(**args, sampling_params=vllm.SamplingParams(
        # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

        best_of=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        use_beam_search=False,
        max_tokens=max_new_tokens,
    ), request_id=uuid.uuid4())

    response = None
    async for response_part in response_generator:
        if not response_part.finished:
            continue
        assert response is None
        outputs = response_part.outputs
        assert len(outputs) == 1
        response = outputs[0].text

    return response

def run_inference(*, prompt, tokenizer_path, model_path, dtype, max_new_tokens, temperature):
    global model

    lock.acquire()

    evaluation.models.models.switch_gpu_model_type('vllm')

    if (model is None
            or model['tokenizer_path'] != tokenizer_path
            or model['model_path'] != model_path
            or model['dtype'] != dtype):
        unload_model(False)
        load_model_in_separate_thread(model_path=model_path, tokenizer_path=tokenizer_path, dtype=dtype)

    model_or_model_condition = model
    if hasattr(model_or_model_condition, 'wait'):
        condition = model_or_model_condition
        with condition:
            condition.wait()

    current_model = model

    assert not hasattr(current_model, 'wait')

    future = asyncio.run_coroutine_threadsafe(respond_to_prompt(prompt=prompt, prompt_model=current_model,
        temperature=temperature, max_new_tokens=max_new_tokens), current_model['event_loop'])

    lock.release()

    return future.result()
