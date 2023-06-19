import subprocess
import threading
import requests

from .parameters import GENERATION_PARAMETERS

API_PORT = 5000
API_URI = 'http://localhost:' + str(API_PORT) + '/api/v1/chat'

server = None
server_lock = threading.Lock()

def ensure_server_is_started():
    server_lock.acquire()

    if server is not None:
        return
    server = subprocess.Popen(['python', 'server.py', '--api', '--api-blocking-port', API_PORT], cwd='text-generation-webui')

    server_lock.release()

class OpenAI:
    def __init__(self, model_name):
        self.model_name = model_name

    def reply(self, conversation):
        ensure_server_is_started()

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
