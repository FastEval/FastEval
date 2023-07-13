import os
import threading
import subprocess

from text_generation import Client

import evaluation.models.models

lock = threading.Lock()
server_information = None

def unload_model(use_lock=True):
    global server_information

    if use_lock:
        lock.acquire()

    if server_information is not None:
        server_information['server']['process'].kill()
        server_information = None

    if use_lock:
        lock.release()

def start_server(*, model_path, tokenizer_path, dtype):
    global server_is_ready

    cwd = os.getcwd()

    new_environment = os.environ.copy()
    new_environment['USE_FLASH_ATTENTION'] = 'TRUE'
    new_environment['PATH'] = (os.path.join(cwd, 'text-generation-inference/.venv/bin') + ':'
        + os.path.join(cwd, 'text-generation-inference/target/release') + ':'
        + os.environ['PATH'])

    port = 1234

    process = subprocess.Popen([
        'text-generation-launcher',
        '--model-id', model_path,
        '--max-total-tokens', '4096',
        '--max-input-length', '2048',
        '--hostname', '127.0.0.1',
        '--port', str(port),
        '--huggingface-hub-cache', '/workspace/huggingface-cache',
        '--dtype', 'b-float16'
    ], env=new_environment, stdout=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line.replace('\n', ''))
        if 'text_generation_router' in line and 'Connected' in line:
            break

    import time
    time.sleep(5)

    return {
        'process': process,
        'port': port,
    }

def run_inference(*, prompt, tokenizer_path, model_path, dtype, max_new_tokens, temperature):
    global server_information

    lock.acquire()

    evaluation.models.models.switch_gpu_model_type('tgi')

    if (server_information is None
            or server_information['tokenizer_path'] != tokenizer_path
            or server_information['model_path'] != model_path
            or server_information['dtype'] != dtype):
        unload_model(False)
        server_information = {
            'tokenizer_path': tokenizer_path,
            'model_path': model_path,
            'dtype': dtype,
            'server': start_server(model_path=model_path, tokenizer_path=tokenizer_path, dtype=dtype),
        }

    client = Client('http://127.0.0.1:' + str(server_information['server']['port']), timeout=1_000_000)

    lock.release()

    if temperature > 1e-8:
        kwargs = { 'temperature': temperature, 'do_sample': True }
    else:
        kwargs = { 'do_sample': False }

    return client.generate(prompt,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        return_full_text=False,
        best_of=1,
        **kwargs,
    ).generated_text
