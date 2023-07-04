import threading
import subprocess
import atexit
import signal

import openai
import tenacity

from .open_ai import OpenAI

lock = threading.Lock()
server = None

def stop_server():
    for process in server['processes']:
        process.kill()

def print_process_output(process_name, process, output_type):
    for line in getattr(process, output_type):
        print('[fastchat ' + process_name + ']', line, end='')

def start_server(model_name):
    global server

    print('[fastchat] Starting server for fastchat:' + model_name)

    controller_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.controller', '--host', '127.0.0.1'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in controller_process.stderr:
        print('[fastchat controller]', line, end='')
        if 'Uvicorn running on' in line:
            break

    print('[fastchat] Started controller. Starting model_worker & openai_api_server next.')

    model_process = subprocess.Popen(['python3', '-m', 'fastchat.serve.model_worker', '--host', '127.0.0.1', '--model-path', model_name],
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

    if server is None:
        start_server(model_name)
    elif server['model_name'] != model_name:
        stop_server()
        start_server(model_name)

    lock.release()

class Fastchat(OpenAI):
    def reply(self, conversation):
        ensure_model_is_loaded(self.model_name)
        openai.api_base = 'http://localhost:8000/v1'
        openai.api_key = 'EMPTY'
        return super()._reply(conversation, self.model_name.split('/')[-1])

def register_exit_handler():
    atexit.register(stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
