import os
import time
import random
import torch
import threading
import subprocess

import transformers
from text_generation import Client

import evaluation.args
import evaluation.models.models

lock = threading.Lock()
server_information = None

def unload_model(use_lock=True):
    global server_information

    if use_lock:
        lock.acquire()

    if server_information is not None:
        server_information['server']['process'].terminate()
        server_information = None

    if use_lock:
        lock.release()

def should_filter_process_output(line):
    if 'GenerateParameters' in line and 'text_generation_router' in line and 'Success' in line:
        return True

    return False

def print_process_output(stdout):
    for line in stdout:
        if not should_filter_process_output(line):
            print('[TGI]', line, end='')

def start_server(*, model_path, tokenizer_path, dtype):
    global server_is_ready

    cwd = os.getcwd()

    new_environment = os.environ.copy()
    new_environment['USE_FLASH_ATTENTION'] = 'TRUE'
    new_environment['PATH'] = (os.path.join(cwd, 'text-generation-inference/.venv/bin') + ':'
        + os.path.join(cwd, 'text-generation-inference/target/release') + ':'
        + os.environ['PATH'])

    port = random.randint(9_000, 10_000)

    if dtype == torch.float16:
        dtype_arg = 'float16'
    elif dtype == torch.bfloat16:
        dtype_arg = 'bfloat16'
    else:
        raise Exception('This dtype is not supported by text-generation-inference')

    if evaluation.args.cmd_arguments.tgi_max_batch_total_tokens is not None:
        additional_args = ['--max-batch-total-tokens', str(evaluation.args.cmd_arguments.tgi_max_batch_total_tokens)]
    else:
        additional_args = []

    process = subprocess.Popen([
        'text-generation-launcher',
        '--model-id', model_path,
        '--max-total-tokens', '4096',
        '--max-input-length', '2048',
        '--hostname', '127.0.0.1',
        '--port', str(port),
        '--dtype', dtype_arg,
        '--max-concurrent-requests', '1024',
        '--num-shard', str(torch.cuda.device_count()),
        *additional_args,
    ], env=new_environment, stdout=subprocess.PIPE, text=True)

    for line in process.stdout:
        if not should_filter_process_output(line):
            print('[TGI]', line, end='')
        if 'text_generation_router' in line and 'Connected' in line:
            break

    threading.Thread(target=print_process_output, args=(process.stdout, )).start()

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
            'tokenizer': transformers.AutoTokenizer.from_pretrained(tokenizer_path),
            'server': start_server(model_path=model_path, tokenizer_path=tokenizer_path, dtype=dtype),
        }

    current_server_information = server_information
    client = Client('http://127.0.0.1:' + str(current_server_information['server']['port']), timeout=1_000_000)

    lock.release()

    if temperature is None:
        temperature = 1.0
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
