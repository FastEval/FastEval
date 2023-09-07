import asyncio
import json
import os

import evaluation.models.models
from evaluation.constants import DEFAULT_MAX_NEW_TOKENS
from evaluation.models.utils import put_system_message_in_user_message

from .open_ai_base import OpenAIBase

server = None
server_lock = asyncio.Lock()


async def unload_model(already_aquired_server_lock=False):
    global server

    if not already_aquired_server_lock:
        await server_lock.acquire()

    if server is not None:
        for process in server["processes"]:
            process.kill()
        server = None

    if not already_aquired_server_lock:
        server_lock.release()


def should_filter_process_output(process_name, line):
    if len(line.strip()) == 0:
        return True

    if process_name == "model":
        if "POST /worker_generate" in line and "200 OK" in line:
            return True
        if "POST /count_token" in line and "200 OK" in line:
            return True
        if "POST /model_details" in line and "200 OK" in line:
            return True
        if "POST /worker_get_conv_template" in line and "200 OK" in line:
            return True
        if "model_worker | Send heart beat. Models:" in line:
            return True
        if "INFO | torch.distributed.distributed_c10d | Added key:" in line:
            return True
        if "INFO | torch.distributed.distributed_c10d | Rank 0:" in line:
            return True
        if "INFO | model_worker | Register to controller" in line:
            return True
    elif process_name == "controller":
        if "POST /get_worker_address" in line and "200 OK" in line:
            return True
        if "POST /list_models" in line and "200 OK" in line:
            return True
        if "controller | Receive heart beat." in line:
            return True
        if "POST /receive_heart_beat" in line and "200 OK" in line:
            return True
        if (
            "INFO | controller | names: ['http://localhost:21002'], " in line
            and ", ret: http://localhost:21002" in line
        ):
            return True
        if "INFO | controller | args: Namespace" in line:
            return True
        if "INFO | controller | Register a new worker:" in line:
            return True
        if "INFO | controller | Register done:" in line:
            return True
        if "POST /register_worker" in line and "200 OK" in line:
            return True

    common_filter = [
        "INFO:     Started server process",
        "INFO:     Waiting for application startup.",
        "INFO:     Application startup complete.",
        "INFO:     Uvicorn running on",
    ]

    for item in common_filter:
        if item in line:
            return True

    return False


def print_process_output_line(process_name, line):
    if should_filter_process_output(process_name, line):
        return
    print("[fastchat " + process_name + "]", line, end="")


async def print_process_output(process_name, process):
    while True:
        line = (await process.stderr.readline()).decode("utf-8")
        print_process_output_line(process_name, line)


async def start_server(*, model_name, tokenizer_path=None, use_vllm):
    global server

    os.environ["FASTCHAT_WORKER_API_TIMEOUT"] = "1000000000"

    controller_process = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        "fastchat.serve.controller",
        "--host",
        "127.0.0.1",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    while True:
        line = (await controller_process.stderr.readline()).decode("utf-8")
        print_process_output_line("controller", line)
        if "Uvicorn running on" in line:
            break

    if use_vllm:
        import torch

        worker_name = "fastchat.serve.vllm_worker"
        additional_worker_args = ["--num-gpus", str(torch.cuda.device_count())]
        if tokenizer_path is not None:
            additional_worker_args += ["--tokenizer", tokenizer_path]
    else:
        worker_name = "fastchat.serve.model_worker"
        if tokenizer_path is not None:
            raise Exception(
                "For fastchat models, the tokenizer can currently only be configured with the vLLM backend."
            )
        additional_worker_args = []

    model_process = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        worker_name,
        "--host",
        "127.0.0.1",
        "--model-path",
        model_name,
        "--controller-address",
        "http://127.0.0.1:21001",
        *additional_worker_args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    api_process = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        "fastchat.serve.openai_api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--controller-address",
        "http://127.0.0.1:21001",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    async def read_until_uvicorn_running(process_name, process):
        while True:
            line = (await process.stderr.readline()).decode("utf-8")
            print_process_output_line(process_name, line)
            if "Uvicorn running on" in line:
                return

    await asyncio.gather(
        read_until_uvicorn_running("model", model_process),
        read_until_uvicorn_running("api", api_process),
    )

    for process_name, process in [
        ("controller", controller_process),
        ("model", model_process),
        ("api", api_process),
    ]:
        asyncio.create_task(print_process_output(process_name, process))

    server = {
        "model_name": model_name,
        "processes": [controller_process, model_process, api_process],
        "use_vllm": use_vllm,
    }


async def ensure_model_is_loaded(*, model_name, use_vllm, tokenizer_path):
    await server_lock.acquire()

    await evaluation.models.models.switch_inference_backend("fastchat")

    if server is None:
        await start_server(
            model_name=model_name, use_vllm=use_vllm, tokenizer_path=tokenizer_path
        )
    elif server["model_name"] != model_name or server["use_vllm"] != use_vllm:
        await unload_model(already_aquired_server_lock=True)
        await start_server(
            model_name=model_name, use_vllm=use_vllm, tokenizer_path=tokenizer_path
        )

    server_lock.release()


class Fastchat(OpenAIBase):
    async def init(
        self,
        model_name,
        *,
        tokenizer=None,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        inference_backend
    ):
        assert inference_backend in ["vllm", "hf_transformers"]
        self.use_vllm = inference_backend == "vllm"
        self.tokenizer_path = tokenizer
        await super().init(model_name, max_new_tokens=max_new_tokens)

    async def reply(self, conversation, *, temperature=None, max_new_tokens=None):
        from openai.error import APIError

        conversation = put_system_message_in_user_message(conversation)
        await ensure_model_is_loaded(
            model_name=self.model_name,
            use_vllm=self.use_vllm,
            tokenizer_path=self.tokenizer_path,
        )

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        return await self.reply_two_attempts_with_different_max_new_tokens(
            conversation=conversation,
            api_base="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            temperature=temperature,
            model_name=self.model_name.split("/")[-1],
            max_new_tokens=max_new_tokens,
            too_many_tokens_error=APIError,
            get_error_message=lambda error: json.loads(error.http_body)["message"],
        )
