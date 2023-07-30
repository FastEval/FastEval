import threading
import multiprocessing
import time
import gc
import os

import torch
import transformers

import evaluation.models.models

global_lock = threading.Lock()
current_worker_process_manager = None

def unload_model():
    global global_lock
    global current_worker_process_manager

    global_lock.acquire()
    if current_worker_process_manager is not None:
        current_worker_process_manager.unload_model()
    current_worker_process_manager = None
    global_lock.release()

def compute_model_response(*, batch, tokenizer, model):
    sampling_parameters_to_batch_items = {}

    for i, batch_item in enumerate(batch):
        temperature = batch_item['temperature']
        if temperature is None:
            temperature = 1.0

        max_new_token = batch_item['max_new_tokens']
        assert max_new_token is not None

        sampling_parameters = (temperature, max_new_token)

        if sampling_parameters not in sampling_parameters_to_batch_items:
            sampling_parameters_to_batch_items[sampling_parameters] = []
        sampling_parameters_to_batch_items[sampling_parameters].append(batch_item)

    for (temperature, max_new_token), batch_items_with_specific_sampling_parameters in sampling_parameters_to_batch_items.items():
        prompts = [batch_item['prompt'] for batch_item in batch_items_with_specific_sampling_parameters]
        input_ids = tokenizer(prompts, return_tensors='pt').to('cuda')
        output_tokens = model.generate(
            **input_ids,

            generation_config=model.generation_config,

            # See the following link for more details & a list of the parameters
            # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig

            # Parameters that control the length of the output
            max_new_tokens=max_new_token,
            min_new_tokens=1,

            # Parameters that control the generation strategy used
            do_sample=temperature > 1e-8,
            num_beams=1,

            # Parameters for manipulation of the model output logits
            temperature=temperature,
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
        )

        for i in range(len(batch_items_with_specific_sampling_parameters)):
            response = output_tokens[i]
            response = response[len(input_ids[i]):]
            response = tokenizer.decode(response)
            result_pipe = batch_items_with_specific_sampling_parameters[i]['result_pipe']
            result_pipe.send(response)
            result_pipe.close()

def run_worker_process(tokenizer_path, model_path, dtype, queue):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = 'left'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map='auto',
    )

    while True:
        item = queue.get()
        if item == 'unload-model':
            break
        compute_model_response(batch=item, tokenizer=tokenizer, model=model)

    tokenizer = None
    model = None
    gc.collect()

    queue.task_done()

start_new_worker_lock = threading.Lock()
def start_new_worker_process(*, tokenizer_path, model_path, dtype, queue, devices):
    start_new_worker_lock.acquire()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(device_id) for device_id in devices])
    multiprocessing.Process(target=run_worker_process, args=(tokenizer_path, model_path, dtype, queue)).start()
    del os.environ['CUDA_VISIBLE_DEVICES']
    start_new_worker_lock.release()

class WorkerProcessManager:
    def __init__(self, *, tokenizer_path, model_path, dtype, maximum_batch_size, num_devices_per_model):
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
            start_new_worker_process(tokenizer_path=self.tokenizer_path, model_path=self.model_path, dtype=self.dtype, queue=self.queue, devices=devices)

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

def run_inference(*, prompt, tokenizer_path, model_path, dtype, max_new_tokens, temperature, max_batch_size):
    global global_lock
    global current_worker_process_manager

    global_lock.acquire()

    evaluation.models.models.switch_gpu_model_type('hf_transformers')

    if (current_worker_process_manager is None
            or current_worker_process_manager.tokenizer_path != tokenizer_path
            or current_worker_process_manager.model_path != model_path
            or current_worker_process_manager.dtype != dtype
            or current_worker_process_manager.maximum_batch_size != max_batch_size):
        if current_worker_process_manager is not None:
            current_worker_process_manager.unload_model()
        current_worker_process_manager = WorkerProcessManager(
            tokenizer_path=tokenizer_path,
            model_path=model_path,
            dtype=dtype,
            maximum_batch_size=max_batch_size,
            num_devices_per_model=1,
        )

    manager = current_worker_process_manager

    global_lock.release()

    result_pipe_parent_conn, result_pipe_child_conn = multiprocessing.Pipe()

    manager.add_item_to_next_batch({
        'prompt': prompt,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'result_pipe': result_pipe_child_conn,
    })

    return result_pipe_parent_conn.recv()
