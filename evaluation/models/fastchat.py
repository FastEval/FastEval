import threading
import subprocess

import openai
import tenacity

from .open_ai import OpenAI

lock = threading.Lock()
server = None

def stop_server():
    for process in server['processes']:
        process.kill()

def start_server(model_name):
    print('Starting server for fastchat:' + model_name)

    p1 = subprocess.Popen(['python3', '-m', 'fastchat.serve.controller'])
    p2 = subprocess.Popen(['python3', '-m', 'fastchat.serve.model_worker', '--model-path', model_name])
    p3 = subprocess.Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', '8000'])

    import time
    time.sleep(120)

    server = {
        'model_name': model_name,
        'processes': [p1, p2, p3],
    }

    return

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
        super().reply(self, conversation)
