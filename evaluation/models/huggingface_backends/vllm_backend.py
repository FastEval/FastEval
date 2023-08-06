import uuid
import asyncio
import threading
import queue

from evaluation.models.huggingface_backends.data_parallel import DataParallelBackend

def create_model_in_separate_thread(*, model_path, tokenizer_path, dtype, resulting_model_queue):
    import torch
    import vllm

    event_loop = asyncio.new_event_loop()

    engine = vllm.AsyncLLMEngine.from_engine_args(vllm.AsyncEngineArgs(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=str(dtype).replace('torch.', ''),
        disable_log_requests=True,
        trust_remote_code=True,
    ))

    model = {
        'tokenizer_path': tokenizer_path,
        'model_path': model_path,
        'dtype': dtype,
        'engine': engine,
        'event_loop': event_loop,
    }

    resulting_model_queue.put(model)

    event_loop.run_forever()

def create_model(*, tokenizer_path, model_path, dtype):
    resulting_model_queue = queue.Queue()

    model_thread = threading.Thread(target=create_model_in_separate_thread, kwargs={
        'model_path': model_path,
        'tokenizer_path': tokenizer_path,
        'dtype': dtype,
        'resulting_model_queue': resulting_model_queue,
    })

    model_thread.start()

    return resulting_model_queue.get()

async def respond_to_prompt(*, model, prompt, temperature, max_new_tokens):
    import vllm

    if temperature is None:
        temperature = 1.0

    if isinstance(prompt, tuple):
        if prompt[0] != 'tokens':
            raise Exception('Unknown prompt type')
        args = { 'prompt_token_ids': prompt[1], 'prompt': None }
    else:
        args = { 'prompt': prompt }

    response_generator = model['engine'].generate(**args, sampling_params=vllm.SamplingParams(
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

def compute_model_response(*, model, item):
    future = asyncio.run_coroutine_threadsafe(respond_to_prompt(model=model, prompt=item['prompt'],
        temperature=item['temperature'], max_new_tokens=item['max_new_tokens']), model['event_loop'])
    return future.result()

def unload_worker_model(model):
    loop = model['event_loop']
    loop.call_soon_threadsafe(loop.stop)

backend = DataParallelBackend(
    backend_name='vllm',
    worker_functions={
        'create_model': create_model,
        'compute_model_response': compute_model_response,
        'unload_worker_model': unload_worker_model,
    },
    worker_is_blocking=False,
)

def run_inference(**kwargs):
    return backend.run_inference(**kwargs, max_batch_size=1)

def unload_model():
    return backend.unload_model()
