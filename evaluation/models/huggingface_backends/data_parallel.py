import multiprocessing
import time
import gc
import os
import random
import asyncio
import concurrent.futures

import evaluation.args
import evaluation.models.models

start_new_worker_lock = asyncio.Lock()

async def handle_item_async(compute_model_response, model, item):
    result_pipe = item['result_pipe']

    try:
        response = await compute_model_response(model=model, item=item)
        result_pipe.send(('response', response))
    except:
        import traceback
        result_pipe.send(('exception', traceback.format_exc()))

    result_pipe.close()

def handle_item_sync(compute_model_responses, model, batch):
    try:
        compute_model_responses(model=model, batch=batch)
    except:
        import traceback
        exception_stacktrace = traceback.format_exc()
        for batch_item in batch:
            result_pipe = batch_item['result_pipe']
            try:
                result_pipe.send(('exception', exception_stacktrace))
                result_pipe.close()
            except:
                pass

async def run_worker_process(*, tokenizer_path, model_path, dtype, queue, worker_functions, worker_is_blocking):
    assert ((worker_is_blocking and 'compute_model_responses' in worker_functions)
        or (not worker_is_blocking and 'compute_model_response' in worker_functions))

    try:
        model = worker_functions['create_model'](
            tokenizer_path=tokenizer_path,
            model_path=model_path,
            dtype=dtype,
        )
    except:
        import traceback
        queue.put(('error-when-creating-model', traceback.format_exc()))
        return

    ack_pipe_parent_conn, ack_pipe_child_conn = multiprocessing.Pipe()
    queue.put(('model-created', ack_pipe_child_conn))
    ack_pipe_parent_conn.recv()

    queue_wait_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    remaining_tasks = set()

    while True:
        item = await loop.run_in_executor(queue_wait_executor, queue.get)

        if item == 'unload-model':
            break

        if worker_is_blocking:
            handle_item_sync(worker_functions['compute_model_responses'], model, item)
        else:
            assert len(item) == 1
            task = asyncio.create_task(handle_item_async(worker_functions['compute_model_response'], model, item[0]))
            remaining_tasks.add(task)
            task.add_done_callback(remaining_tasks.discard)

    for task in remaining_tasks:
        await task

    if 'unload_worker_model' in worker_functions:
        worker_functions['unload_worker_model'](model)

    model = None
    gc.collect()

def run_worker_process_in_new_event_loop(**kwargs):
    asyncio.run(run_worker_process(**kwargs))

async def start_new_worker_process(*, tokenizer_path, model_path, dtype, queue, devices, worker_functions, worker_is_blocking):
    # This lock is needed because we modify the `CUDA_VISIBLE_DEVICES` environment variable
    # before starting the new child process. This environment variable is global for our whole process,
    # so we need to start the child processes sequentially
    await start_new_worker_lock.acquire()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        previous_cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    else:
        previous_cuda_visible_devices = None

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(device_id) for device_id in devices])

    multiprocessing.Process(target=run_worker_process_in_new_event_loop, kwargs={
        'tokenizer_path': tokenizer_path,
        'model_path': model_path,
        'dtype': dtype,
        'queue': queue,
        'worker_functions': worker_functions,
        'worker_is_blocking': worker_is_blocking,
    }).start()

    if previous_cuda_visible_devices is None:
        del os.environ['CUDA_VISIBLE_DEVICES']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = previous_cuda_visible_devices

    start_new_worker_lock.release()

class WorkerProcessManager:
    async def init(self, *, tokenizer_path, model_path, dtype, maximum_batch_size, num_devices_per_model, worker_functions, worker_is_blocking):
        import torch

        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.dtype = dtype
        self.maximum_batch_size = maximum_batch_size
        self.worker_is_blocking = worker_is_blocking

        self.next_batch = []
        self.timestamp_when_last_batch_item_was_added = None
        self.lock = asyncio.Lock()

        self.num_workers = torch.cuda.device_count() // num_devices_per_model

        if worker_is_blocking:
            self.queue = multiprocessing.Queue()
        else:
            self.queues = [multiprocessing.Queue() for _ in range(self.num_workers)]

        for i in range(self.num_workers):
            if worker_is_blocking:
                queue = self.queue
            else:
                queue = self.queues[i]

            devices = list(range(i * num_devices_per_model, (i + 1) * num_devices_per_model))
            await start_new_worker_process(tokenizer_path=self.tokenizer_path, model_path=self.model_path, dtype=self.dtype,
                queue=queue, devices=devices, worker_functions=worker_functions, worker_is_blocking=worker_is_blocking)

        ack_pipes = []
        for i in range(self.num_workers):
            if self.worker_is_blocking:
                model_creation_result = self.queue.get()
            else:
                model_creation_result = self.queues[i].get()

            if model_creation_result[0] == 'model-created':
                ack_pipes.append(model_creation_result[1])
            else:
                raise Exception('Model creation in worker failed: ' + model_creation_result[1])

        for pipe in ack_pipes:
            pipe.send('ack')

        self.models_are_loaded = True

    async def unload_model(self):
        await self.lock.acquire()

        assert self.models_are_loaded

        for i in range(self.num_workers):
            if self.worker_is_blocking:
                self.queue.put('unload-model')
            else:
                self.queues[i].put('unload-model')

        self.models_are_loaded = False

        self.lock.release()

    async def add_item_to_next_batch(self, item):
        await self.lock.acquire()
        self.next_batch.append(item)
        self.lock.release()

        time.sleep(0.05)

        await self.lock.acquire()

        if self.worker_is_blocking:
            queue = self.queue
        else:
            # TODO Keep track of workload and also how well the worker is handling
            # that workload, i.e. not just the submitted items but also the processed
            # items. Distribute new workload based on that.
            queue = random.choice(self.queues)

        queue.put(self.next_batch[:self.maximum_batch_size])
        self.next_batch = self.next_batch[self.maximum_batch_size:]

        self.lock.release()

async def pipe_receive_async(pipe):
    event = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_reader(pipe.fileno(), event.set)
    if not pipe.poll():
        await event.wait()
    result = pipe.recv()
    event.clear()
    return result

class DataParallelBackend:
    def __init__(self, *, backend_name, worker_functions, worker_is_blocking):
        self.backend_name = backend_name
        self.worker_functions = worker_functions
        self.worker_is_blocking = worker_is_blocking

        self.lock = asyncio.Lock()
        self.current_worker_process_manager = None

    async def run_inference(self, *, prompt, tokenizer_path, model_path, dtype, max_new_tokens, temperature, max_batch_size):
        import torch

        await self.lock.acquire()

        await evaluation.models.models.switch_inference_backend(self.backend_name)

        if (self.current_worker_process_manager is None
                or self.current_worker_process_manager.tokenizer_path != tokenizer_path
                or self.current_worker_process_manager.model_path != model_path
                or self.current_worker_process_manager.dtype != dtype
                or self.current_worker_process_manager.maximum_batch_size != max_batch_size):
            if self.current_worker_process_manager is not None:
                await self.current_worker_process_manager.unload_model()
            try:
                self.current_worker_process_manager = WorkerProcessManager()
                await self.current_worker_process_manager.init(
                    tokenizer_path=tokenizer_path,
                    model_path=model_path,
                    dtype=dtype,
                    maximum_batch_size=max_batch_size,
                    num_devices_per_model=evaluation.args.cmd_arguments.num_gpus_per_model or torch.cuda.device_count(),
                    worker_functions=self.worker_functions,
                    worker_is_blocking=self.worker_is_blocking,
                )
            except Exception as error:
                self.lock.release()
                raise error

        manager = self.current_worker_process_manager

        self.lock.release()

        result_pipe_parent_conn, result_pipe_child_conn = multiprocessing.Pipe()

        manager.add_item_to_next_batch({
            'prompt': prompt,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'result_pipe': result_pipe_child_conn,
        })

        result = await pipe_receive_async(result_pipe_parent_conn)

        if result[0] == 'response':
            return result[1]
        raise Exception('Error when running inference: ' + result[1])

    async def unload_model(self):
        await self.lock.acquire()
        if self.current_worker_process_manager is not None:
            await self.current_worker_process_manager.unload_model()
        self.current_worker_process_manager = None
        self.lock.release()
