import subprocess
import threading
import requests

from .parameters import GENERATION_PARAMETERS

API_PORT = 5000
API_URI = 'http://localhost:' + str(API_PORT) + '/api/v1/chat'

server = None
server_lock = threading.Lock()

def ensure_server_is_started(model_name):
    global server

    server_lock.acquire()

    if server is not None and server['model_name'] == model_name:
        return

    if server is not None and server['model_name'] != model_name:
        previous_server_process = server['process']
        p.kill()
        server = None

    server_process = subprocess.Popen([
        'python', '-u', 'server.py',
        '--api',
        '--api-blocking-port', str(API_PORT),
        '--model', model_name,
        '--trust-remote-code',
        '--bf16',
    ], cwd='text-generation-webui', stdout=subprocess.PIPE, text=True)

    server = {
        'model_name': model_name,
        'process': server_process,
    }

    for line in server_process.stdout:
        if line == 'Starting API at http://127.0.0.1:5000/api\n':
            break

    server_lock.release()

class TextGenerationWebUI:
    def __init__(self, model_name):
        self.model_name = model_name

    def reply(self, conversation):
        ensure_server_is_started(self.model_name)

        import time
        time.sleep(10000000)

        # https://github.com/oobabooga/text-generation-webui/blob/main/api-examples/api-example-chat.py
        # https://github.com/oobabooga/text-generation-webui/blob/main/api-examples/api-example-model.py

        prompt = 'Hi!'
        history = {
            'internal': [],
            'visible': [],
        }

        request = {
            'mode': 'chat',
            'user_input': prompt,
            'history': history,

            'max_new_tokens': 400,

            'preset': 'None',
            **GENERATION_PARAMETERS,
        }

        response = requests.post(API_URI, json=request)
        if response.status_code != 200:
            print(response)
            raise
        result = response.json()['results'][0]['history']

        print(result)
        import sys
        sys.exit()
