import atexit
import signal
from contextlib import contextmanager
import multiprocessing.pool

import torch
import tqdm
import transformers

import evaluation.models.fastchat
import evaluation.models.huggingface

from evaluation.models.open_ai import OpenAI
from evaluation.models.fastchat import Fastchat
from evaluation.models.open_assistant import OpenAssistant
from evaluation.models.guanaco import Guanaco
from evaluation.models.falcon_instruct import FalconInstruct
from evaluation.models.alpaca_without_prefix import AlpacaWithoutPrefix
from evaluation.models.alpaca_with_prefix import AlpacaWithPrefix
from evaluation.models.chatml import ChatML

def replace_model_name_slashes(model_name: str) -> str:
    """
    The model name can be something like OpenAssistant/oasst-sft-1-pythia-12b.
    The path where we store evaluation results should depend on the model name,
    but paths can't include '/', so we need to replace that.
    """

    return model_name.replace('/', '--')

def undo_replace_model_name_slashes(model_name: str) -> str:
    return model_name.replace('--', '/')

def get_model_class(model_type: str):
    model_classes = {
        'openai': OpenAI,
        'fastchat': Fastchat,
        'open-assistant': OpenAssistant,
        'guanaco': Guanaco,
        'falcon-instruct': FalconInstruct,
        'alpaca-without-prefix': AlpacaWithoutPrefix,
        'alpaca-with-prefix': AlpacaWithPrefix,
        'chatml': ChatML,
    }

    if model_type in model_classes:
        return model_classes[model_type]

    raise

def create_model(model_type: str, model_name: str, **kwargs):
    return get_model_class(model_type)(model_name, **kwargs)

config_dict_cache = {}
def get_config_dict(model_name):
    if model_name in config_dict_cache:
        return config_dict_cache[model_name]
    config_dict = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config_dict_cache[model_name] = config_dict
    return config_dict

def get_dtype(model_name: str):
    return get_config_dict(model_name).torch_dtype

def is_vllm_supported(model_name: str):
    model_type = get_config_dict(model_name).model_type
    if model_type in ['llama', 'gpt_neox', 'gpt_bigcode', 'mpt']:
        return True
    if model_type in ['RefinedWeb', 'RefinedWebModel']:
        return False
    raise

def compute_model_replies(model, conversations):
    if len(conversations) == 0:
        return []

    def reply(conversation_with_index):
        index, conversation = conversation_with_index
        reply = model.reply(conversation)
        return index, reply

    with multiprocessing.pool.ThreadPool(min(model.num_threads, len(conversations))) as pool:
        iterator = pool.imap_unordered(reply, enumerate(conversations))
        replies_with_indices = list(tqdm.tqdm(iterator, total=len(conversations)))

    return [reply_with_index[1] for reply_with_index in sorted(replies_with_indices, key=lambda item: item[0])]

def unload_model(*args, **kwargs):
    evaluation.models.fastchat.unload_model()
    evaluation.models.huggingface.unload_model()

def switch_gpu_model_type(new_model_type):
    if new_model_type == 'huggingface':
        evaluation.models.fastchat.unload_model()
    if new_model_type == 'fastchat':
        evaluation.models.huggingface.unload_model()

@contextmanager
def changed_exit_handlers():
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)

    atexit.register(unload_model)
    signal.signal(signal.SIGTERM, unload_model)
    signal.signal(signal.SIGINT, unload_model)

    yield

    atexit.unregister(unload_model)
    signal.signal(signal.SIGTERM, previous_sigterm)
    signal.signal(signal.SIGINT, previous_sigint)
