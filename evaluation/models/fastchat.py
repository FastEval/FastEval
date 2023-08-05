import threading
import subprocess
import os
import re
import json

from .open_ai_base import OpenAIBase

import evaluation.models.models
from evaluation.models.utils import put_system_message_in_prompter_message
from evaluation.constants import NUM_THREADS_LOCAL_MODEL, DEFAULT_MAX_NEW_TOKENS

lock = threading.Lock()
server = None

def unload_model(use_lock=True):
    global server

    if use_lock:
        lock.acquire()

    if server is not None:
        for process in server['processes']:
            process.kill()
        server = None

    if use_lock:
        lock.release()

def should_filter_process_output(process_name, line):
    if process_name == 'model':
        if 'POST /worker_generate' in line and '200 OK' in line:
            return True
        if 'POST /count_token' in line and '200 OK' in line:
            return True
        if 'POST /model_details' in line and '200 OK' in line:
            return True
        if 'POST /worker_get_conv_template' in line and '200 OK' in line:
            return True
        if 'model_worker | Send heart beat. Models:' in line:
            return True
        if 'INFO | torch.distributed.distributed_c10d | Added key:' in line:
            return True
        if 'INFO | torch.distributed.distributed_c10d | Rank 0:' in line:
            return True
        if 'INFO | model_worker | Register to controller' in line:
            return True
    elif process_name == 'controller':
        if 'POST /get_worker_address' in line and '200 OK' in line:
            return True
        if 'controller | Receive heart beat.' in line:
            return True
        if 'POST /receive_heart_beat' in line and '200 OK' in line:
            return True
        if "INFO | controller | names: ['http://localhost:21002'], " in line and ", ret: http://localhost:21002" in line:
            return True
        if 'INFO | controller | args: Namespace' in line:
            return True
        if 'INFO | controller | Register a new worker:' in line:
            return True
        if 'INFO | controller | Register done:' in line:
            return True
        if 'POST /register_worker' in line and '200 OK' in line:
            return True

    common_filter = [
        'INFO:     Started server process',
        'INFO:     Waiting for application startup.',
        'INFO:     Application startup complete.',
        'INFO:     Uvicorn running on',
    ]

    for item in common_filter:
        if item in line:
            return True

    return False

def print_process_output_line(process_name, line):
    if should_filter_process_output(process_name, line):
        return
    print('[fastchat ' + process_name + ']', line, end='')

def print_process_output(process_name, process):
    for line in process.stderr:
        print_process_output_line(process_name, line)

def start_server(model_name, use_vllm):
    global server

    os.environ['FASTCHAT_WORKER_API_TIMEOUT'] = '1000000000'

    controller_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.controller', '--host', '127.0.0.1'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    for line in controller_process.stderr:
        print_process_output_line('controller', line)
        if 'Uvicorn running on' in line:
            break

    if use_vllm:
        worker_name = 'fastchat.serve.vllm_worker'
        if model_name in ['lmsys/vicuna-7b-v1.3', 'lmsys/vicuna-33b-v1.3']:
            additional_worker_args = ['--tokenizer', 'hf-internal-testing/llama-tokenizer']
        else:
            additional_worker_args = []
    else:
        worker_name = 'fastchat.serve.model_worker'
        additional_worker_args = []

    model_process = subprocess.Popen(['python3', '-m', worker_name, '--host', '127.0.0.1', '--model-path', model_name, *additional_worker_args],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    api_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', '127.0.0.1', '--port', '8000'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    for process_name, process in [('model', model_process), ('api', api_process)]:
        for line in process.stderr:
            print_process_output_line(process_name, line)
            if 'Uvicorn running on' in line:
                break

    for process_name, process in [('controller', controller_process), ('model', model_process), ('api', api_process)]:
        threading.Thread(target=print_process_output, args=(process_name, process)).start()

    server = {
        'model_name': model_name,
        'processes': [controller_process, model_process, api_process],
        'use_vllm': use_vllm,
    }

def ensure_model_is_loaded(model_name, use_vllm):
    lock.acquire()

    evaluation.models.models.switch_gpu_model_type('fastchat')

    if server is None:
        start_server(model_name, use_vllm)
    elif server['model_name'] != model_name or server['use_vllm'] != use_vllm:
        unload_model(False)
        start_server(model_name, use_vllm)

    lock.release()

class Fastchat(OpenAIBase):
    num_threads = NUM_THREADS_LOCAL_MODEL

    def __init__(self, model_name, *, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        self.use_vllm = evaluation.models.models.is_vllm_supported(model_name)
        super().__init__(model_name, max_new_tokens=max_new_tokens)

    def _reply(self, *, conversation, api_base, api_key, temperature, model_name, max_new_tokens):
        from openai.error import APIError

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        try:
            return super()._reply(conversation=conversation, api_base=api_base, api_key=api_key, temperature=temperature,
                model_name=model_name, max_new_tokens=max_new_tokens)
        except APIError as error:
            error_information = re.search("This model's maximum context length is ([0-9]+) tokens\. "
                + 'However, you requested ([0-9]+) tokens \([0-9]+ in the messages, [0-9]+ in the completion\)\. '
                + 'Please reduce the length of the messages or completion\.', json.loads(error.http_body)['message'])
            maximum_context_length = int(error_information.group(1))
            request_total_length = int(error_information.group(2))
            num_tokens_too_much = request_total_length - maximum_context_length
            reduced_max_new_tokens = max_new_tokens - num_tokens_too_much
            return super()._reply(conversation=conversation, api_base=api_base, api_key=api_key,
                max_new_tokens=reduced_max_new_tokens, temperature=temperature, model_name=model_name)

    def reply(self, conversation, temperature=None, max_new_tokens=None):
        conversation = put_system_message_in_prompter_message(conversation)
        ensure_model_is_loaded(self.model_name, use_vllm=self.use_vllm)
        return self._reply(
            conversation=conversation,
            temperature=temperature,
            api_base='http://localhost:8000/v1',
            api_key='EMPTY',
            model_name=self.model_name.split('/')[-1],
            max_new_tokens=max_new_tokens,
        )
