import asyncio
import os
import random
import subprocess

from evaluation.models.huggingface_backends.data_parallel import DataParallelBackend


def should_filter_process_output(line):
    if (
        "GenerateParameters" in line
        and "text_generation_router" in line
        and "Success" in line
    ):
        return True

    return False


async def print_process_output(stdout):
    while True:
        line = (await stdout.readline()).decode("utf-8")
        if not should_filter_process_output(line):
            print("[TGI]", line, end="")


async def create_model(*, model_path, tokenizer_path, dtype):
    # TODO: Actually use the tokenizer_path

    import torch

    cwd = os.getcwd()

    new_environment = os.environ.copy()
    new_environment["USE_FLASH_ATTENTION"] = "TRUE"
    new_environment["PATH"] = (
        os.path.join(cwd, "text-generation-inference/.venv/bin")
        + ":"
        + os.path.join(cwd, "text-generation-inference/target/release")
        + ":"
        + os.environ["PATH"]
    )

    port = random.randint(9_000, 10_000)

    if dtype == torch.float16:
        dtype_arg = "float16"
    elif dtype == torch.bfloat16:
        dtype_arg = "bfloat16"
    else:
        raise Exception("This dtype is not supported by text-generation-inference")

    process = await asyncio.create_subprocess_exec(
        "text-generation-launcher",
        "--model-id",
        model_path,
        "--max-total-tokens",
        "4096",
        "--max-input-length",
        "2048",
        "--hostname",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        dtype_arg,
        "--max-concurrent-requests",
        "8192",
        "--num-shard",
        str(torch.cuda.device_count()),
        env=new_environment,
        stdout=asyncio.subprocess.PIPE,
    )

    while True:
        line = (await process.stdout.readline()).decode("utf-8")
        if not should_filter_process_output(line):
            print("[TGI]", line, end="")
        if "text_generation_router" in line and "Connected" in line:
            break

    asyncio.create_task(print_process_output(process.stdout))

    await asyncio.sleep(5)

    return {
        "process": process,
        "port": port,
    }


async def compute_model_response(*, model, item):
    from text_generation import AsyncClient

    client = AsyncClient("http://127.0.0.1:" + str(model["port"]), timeout=1_000_000)

    temperature = item["temperature"]
    if temperature is None:
        temperature = 1.0
    if temperature > 1e-8:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}

    max_new_tokens = item["max_new_tokens"]
    assert max_new_tokens is not None

    return (
        await client.generate(
            item["prompt"],
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
            return_full_text=False,
            best_of=1,
            **kwargs,
        )
    ).generated_text


def unload_worker_model(model):
    model["process"].terminate()


backend = DataParallelBackend(
    backend_name="tgi",
    worker_functions={
        "create_model": create_model,
        "compute_model_response": compute_model_response,
        "unload_worker_model": unload_worker_model,
    },
    worker_is_blocking=False,
)


async def run_inference(**kwargs):
    return await backend.run_inference(**kwargs, max_batch_size=1)


async def unload_model():
    return await backend.unload_model()
