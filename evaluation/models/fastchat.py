import threading
import subprocess
import os
import re
import json

import torch
import openai
import tenacity

from .open_ai import OpenAI

import evaluation.utils

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

def print_process_output(process_name, process, output_type):
    for line in getattr(process, output_type):
        print('[fastchat ' + process_name + ']', line, end='')

def start_server(model_name):
    global server

    use_vllm = True # TODO: Get that from somewhere. Not all models are supported.

    print('[fastchat] Starting server for fastchat:' + model_name)

    os.environ['FASTCHAT_WORKER_API_TIMEOUT'] = '1000000000'

    controller_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.controller', '--host', '127.0.0.1'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in controller_process.stderr:
        print('[fastchat controller]', line, end='')
        if 'Uvicorn running on' in line:
            break

    print('[fastchat] Started controller. Starting model_worker & openai_api_server next.')

    if use_vllm:
        worker_name = 'fastchat.serve.vllm_worker'
    else:
        worker_name = 'fastchat.serve.model_worker'

    model_process = subprocess.Popen(['python3', '-m', worker_name, '--host', '127.0.0.1', '--model-path', model_name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    api_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', '127.0.0.1', '--port', '8000'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for process_name, process in [('model', model_process), ('api', api_process)]:
        for line in process.stderr:
            print('[fastchat ' + process_name + ']', line, end='')
            if 'Uvicorn running on' in line:
                break

    for process_name, process in [('controller', controller_process), ('model', model_process), ('api', api_process)]:
        for output_type in ['stdout', 'stderr']:
            threading.Thread(target=print_process_output, args=(process_name, process, output_type)).start()

    server = {
        'model_name': model_name,
        'processes': [controller_process, model_process, api_process],
    }

def ensure_model_is_loaded(model_name):
    lock.acquire()

    evaluation.utils.switch_gpu_model_type('fastchat')

    if server is None:
        start_server(model_name)
    elif server['model_name'] != model_name:
        unload_model(False)
        start_server(model_name)

    lock.release()

class Fastchat(OpenAI):
    def __init__(self, model_name, *, max_new_tokens=400):
        super().__init__(model_name, max_new_tokens=max_new_tokens)

    def _reply(self, conversation, model_name):
        try:
            return super()._reply(conversation, model_name)
        except openai.error.APIError as error:
            error_information = re.search("This model's maximum context length is ([0-9]+) tokens\. "
                + 'However, you requested ([0-9]+) tokens \([0-9]+ in the messages, [0-9]+ in the completion\)\. '
                + 'Please reduce the length of the messages or completion\.', json.loads(error.http_body)['message'])
            maximum_context_length = int(error_information.group(1))
            request_total_length = int(error_information.group(2))
            num_tokens_too_much = request_total_length - maximum_context_length
            reduced_max_new_tokens = self.max_new_tokens - num_tokens_too_much
            return super()._reply(conversation, model_name, max_new_tokens=reduced_max_new_tokens)

    def reply(self, conversation):
        ensure_model_is_loaded(self.model_name)
        openai.api_base = 'http://localhost:8000/v1'
        openai.api_key = 'EMPTY'
        return self._reply(conversation, self.model_name.split('/')[-1])
