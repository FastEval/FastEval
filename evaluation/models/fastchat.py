import threading
import subprocess
import os
import re
import json

from .open_ai_base import OpenAIBase

import evaluation.models.models
from evaluation.models.utils import put_system_message_in_user_message
from evaluation.constants import DEFAULT_MAX_NEW_TOKENS

server = None
server_lock = threading.RLock()

def unload_model():
    global server

    server_lock.acquire()

    if server is not None:
        for process in server['processes']:
            process.kill()
        server = None

    server_lock.release()

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
        if 'POST /list_models' in line and '200 OK' in line:
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

def start_server(*, model_name, tokenizer_path=None, use_vllm):
    global server

    os.environ['FASTCHAT_WORKER_API_TIMEOUT'] = '1000000000'

    controller_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.controller', '--host', '127.0.0.1'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    for line in controller_process.stderr:
        print_process_output_line('controller', line)
        if 'Uvicorn running on' in line:
            break

    if use_vllm:
        import torch
        worker_name = 'fastchat.serve.vllm_worker'
        additional_worker_args = ['--num-gpus', str(torch.cuda.device_count())]
        if tokenizer_path is not None:
            additional_worker_args += ['--tokenizer', tokenizer_path]
    else:
        worker_name = 'fastchat.serve.model_worker'
        if tokenizer_path is not None:
            raise Exception('For fastchat models, the tokenizer can currently only be configured with the vLLM backend.')
        additional_worker_args = []

    model_process = subprocess.Popen(['python3', '-m', worker_name, '--host', '127.0.0.1', '--model-path', model_name,
        '--controller-address', 'http://127.0.0.1:21001', *additional_worker_args], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    api_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', '127.0.0.1', '--port', '8000',
        '--controller-address', 'http://127.0.0.1:21001'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

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

def ensure_model_is_loaded(*, model_name, use_vllm, tokenizer_path):
    server_lock.acquire()

    evaluation.models.models.switch_inference_backend('fastchat')

    if server is None:
        start_server(model_name=model_name, use_vllm=use_vllm, tokenizer_path=tokenizer_path)
    elif server['model_name'] != model_name or server['use_vllm'] != use_vllm:
        unload_model()
        start_server(model_name=model_name, use_vllm=use_vllm, tokenizer_path=tokenizer_path)

    server_lock.release()

class Fastchat(OpenAIBase):
    async def init(self, model_name, *, tokenizer=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, inference_backend):
        assert inference_backend in ['vllm', 'hf_transformers']
        self.use_vllm = inference_backend == 'vllm'
        self.tokenizer_path = tokenizer
        await super().init(model_name, max_new_tokens=max_new_tokens)

    async def reply(self, conversation, *, temperature=None, max_new_tokens=None):
        from openai.error import APIError

        conversation = put_system_message_in_user_message(conversation)
        ensure_model_is_loaded(model_name=self.model_name, use_vllm=self.use_vllm, tokenizer_path=self.tokenizer_path)

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        api_base = 'http://127.0.0.1:8000/v1'
        api_key = 'EMPTY'
        model_name = self.model_name.split('/')[-1]

        try:
            return await super().reply_single_try(conversation=conversation, api_base=api_base, api_key=api_key, temperature=temperature,
                model_name=model_name, max_new_tokens=max_new_tokens)
        except APIError as error:
            error_message = json.loads(error.http_body)['message']
            error_information = re.search("This model's maximum context length is ([0-9]+) tokens\. "
                + 'However, you requested ([0-9]+) tokens \([0-9]+ in the messages, [0-9]+ in the completion\)\. '
                + 'Please reduce the length of the messages or completion\.', error_message)
            if error_information is None:
                raise Exception('Fastchat Error: ' + error_message)
            maximum_context_length = int(error_information.group(1))
            request_total_length = int(error_information.group(2))
            num_tokens_too_much = request_total_length - maximum_context_length
            reduced_max_new_tokens = max_new_tokens - num_tokens_too_much
            if reduced_max_new_tokens <= 0:
                return ''
            return await super().reply_single_try(conversation=conversation, api_base=api_base, api_key=api_key,
                max_new_tokens=reduced_max_new_tokens, temperature=temperature, model_name=model_name)
