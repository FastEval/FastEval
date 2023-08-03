import threading
import multiprocessing
import time
import gc
import os

import torch

import evaluation.args
import evaluation.models.models

def run_worker_process(tokenizer_path, model_path, dtype, queue, worker_functions):
    model = worker_functions['create_model'](
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        dtype=dtype,
    )

    while True:
        item = queue.get()
        if item == 'unload-model':
            break

        batch = item

        if 'compute_model_response' in worker_functions:
            for i in range(len(batch)):
                response = worker_functions['compute_model_response'](batch[i])
                result_pipe = batch[i]['result_pipe']
                result_pipe.send(response)
                result_pipe.close()
        elif 'compute_model_responses' in worker_functions:
            worker_functions['compute_model_responses'](
                model=model,
                batch=batch,
            )
        else:
            raise

    model = None
    gc.collect()

    queue.task_done()

start_new_worker_lock = threading.Lock()
def start_new_worker_process(*, tokenizer_path, model_path, dtype, queue, devices, worker_functions):
    start_new_worker_lock.acquire()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(device_id) for device_id in devices])
    multiprocessing.Process(target=run_worker_process, args=(tokenizer_path, model_path, dtype, queue, worker_functions)).start()
    del os.environ['CUDA_VISIBLE_DEVICES']
    start_new_worker_lock.release()

class WorkerProcessManager:
    def __init__(self, *, tokenizer_path, model_path, dtype, maximum_batch_size, num_devices_per_model, worker_functions):
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.dtype = dtype
        self.maximum_batch_size = maximum_batch_size

        self.next_batch = []
        self.timestamp_when_last_batch_item_was_added = None
        self.queue = multiprocessing.Queue()
        self.lock = threading.Lock()

        self.num_threads = torch.cuda.device_count() // num_devices_per_model
        for i in range(self.num_threads):
            devices = list(range(i * num_devices_per_model, (i + 1) * num_devices_per_model))
            start_new_worker_process(tokenizer_path=self.tokenizer_path, model_path=self.model_path, dtype=self.dtype,
                queue=self.queue, devices=devices, worker_functions=worker_functions)

        self.models_are_loaded = True

    def unload_model(self):
        self.lock.acquire()

        assert self.models_are_loaded

        for i in range(self.num_threads):
            self.queue.put('unload-model')

        self.models_are_loaded = False

        self.lock.release()

    def add_item_to_next_batch(self, item):
        self.lock.acquire()
        self.next_batch.append(item)
        self.lock.release()

        time.sleep(0.05)

        self.lock.acquire()
        self.queue.put(self.next_batch[:self.maximum_batch_size])
        self.next_batch = self.next_batch[self.maximum_batch_size:]
        self.lock.release()

class DataParallelBackend:
    def __init__(self, *, backend_name, worker_functions):
        self.backend_name = backend_name
        self.worker_functions = worker_functions

        self.lock = threading.Lock()
        self.current_worker_process_manager = None

    def run_inference(self, *, prompt, tokenizer_path, model_path, dtype, max_new_tokens, temperature, max_batch_size):
        self.lock.acquire()

        evaluation.models.models.switch_gpu_model_type(self.backend_name)

        if (self.current_worker_process_manager is None
                or self.current_worker_process_manager.tokenizer_path != tokenizer_path
                or self.current_worker_process_manager.model_path != model_path
                or self.current_worker_process_manager.dtype != dtype
                or self.current_worker_process_manager.maximum_batch_size != max_batch_size):
            if self.current_worker_process_manager is not None:
                self.current_worker_process_manager.unload_model()
            self.current_worker_process_manager = WorkerProcessManager(
                tokenizer_path=tokenizer_path,
                model_path=model_path,
                dtype=dtype,
                maximum_batch_size=max_batch_size,
                num_devices_per_model=evaluation.args.cmd_arguments.num_gpus_per_model,
                worker_functions=self.worker_functions,
            )

        manager = self.current_worker_process_manager

        self.lock.release()

        result_pipe_parent_conn, result_pipe_child_conn = multiprocessing.Pipe()

        manager.add_item_to_next_batch({
            'prompt': prompt,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'result_pipe': result_pipe_child_conn,
        })

        return result_pipe_parent_conn.recv()

    def unload_model(self):
        self.lock.acquire()
        if self.current_worker_process_manager is not None:
            self.current_worker_process_manager.unload_model()
        self.current_worker_process_manager = None
        self.lock.release()
