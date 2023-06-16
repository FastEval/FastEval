import subprocess
import threading

import torch
import transformers
import text_generation

from .utils import put_system_message_in_prompter_message

server = None
server_lock = threading.Lock()

def start_server(model_path, dtype):
    global server

    # TODO: dtype currently not used
    server_process = subprocess.Popen([
        'docker', 'run',
        '--gpus', 'all',
        '--shm-size', '1g',
        '-p', '8080:80',
        '-v', '.docker-shared-volume:/data',
        'ghcr.io/huggingface/text-generation-inference:0.8',
        '--model-id', model_path,
    ])

    server = {
        'model_path': model_path,
        'dtype': dtype,
        'process': server_process,
    }

def stop_server():
    server['process'].kill()

def request(*, model_path, dtype, prompt):
    server_lock.acquire()
    if server is None:
        start_server()
    elif server['model_path'] != model_path or server['dtype'] != dtype:
        stop_server()
        start_server()
    server_lock.release()

    client = text_generation.Client('http://127.0.0.1:8080')
    print(client.generate(prompt, max_new_tokens=400))
    raise

class Huggingface:
    def __init__(
        self,
        model_path: str,
        *,
        prefix='',
        user: str,
        assistant: str,
        end: str,
    ):
        self.model_path = model_path
        self.dtype = self.__class__.get_dtype(model_path)

        self.prefix = prefix
        self.user = user
        self.assistant = assistant

        if end == 'tokenizer-eos-token':
            self.end = transformers.AutoTokenizer.from_pretrained(model_path).eos_token
        else:
            self.end = end

    @staticmethod
    def get_dtype(model_path: str):
        return torch.float16

    def _conversation_item_to_prompt(self, item_type, item):
        if item_type == 'assistant':
            return self.assistant + item + self.end
        elif item_type == 'user':
            return self.user + item + self.end
        else:
            raise

    def _conversation_to_prompt(self, conversation):
        conversation = put_system_message_in_prompter_message(conversation)
        return self.prefix + ''.join(self._conversation_item_to_prompt(item_type, item) for item_type, item in conversation) + self.assistant

    def reply(self, conversation):
        prompt = self._conversation_to_prompt(conversation)
        response = request(model_path=self.model_path, dtype=self.dtype, prompt=prompt)
        response = response[0]['generated_text'][len(prompt):]
        response = response.split(self.user)[0] # some models continue to simulate the user and further assistant conversation
        response = response.strip()
        return response
