import os
import multiprocessing.pool

import torch
import transformers
import tqdm

import evaluation.models.fastchat
import evaluation.models.huggingface_backends.hf_transformers
import evaluation.models.huggingface_backends.vllm
import evaluation.models.huggingface_backends.tgi
from evaluation.models.open_ai import OpenAI
from evaluation.models.fastchat import Fastchat
from evaluation.models.open_assistant import OpenAssistant
from evaluation.models.guanaco import Guanaco
from evaluation.models.falcon_instruct import FalconInstruct
from evaluation.models.alpaca_without_prefix import AlpacaWithoutPrefix
from evaluation.models.alpaca_with_prefix import AlpacaWithPrefix
from evaluation.models.chatml import ChatML
from evaluation.models.starchat import Starchat

config_dict_cache = {}

def create_model(model_type: str, model_name: str, **kwargs):
    model_classes = {
        'openai': OpenAI,
        'fastchat': Fastchat,
        'open-assistant': OpenAssistant,
        'guanaco': Guanaco,
        'falcon-instruct': FalconInstruct,
        'alpaca-without-prefix': AlpacaWithoutPrefix,
        'alpaca-with-prefix': AlpacaWithPrefix,
        'chatml': ChatML,
        'starchat': Starchat,
    }

    if model_type not in model_classes:
        raise Exception('Unknown model type "' + model_type + '"')

    model_class = model_classes[model_type]

    return model_class(model_name, **kwargs)

def get_config_dict(model_name):
    if model_name in config_dict_cache:
        return config_dict_cache[model_name]
    config_dict = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config_dict_cache[model_name] = config_dict
    return config_dict

def get_dtype(model_name: str):
    return get_config_dict(model_name).torch_dtype

def is_vllm_supported(model_name: str):
    if 'starchat' in model_name:
        return False # https://github.com/vllm-project/vllm/issues/380
    if 'starcoder' in model_name:
        return False # https://github.com/vllm-project/vllm/issues/393

    model_type = get_config_dict(model_name).model_type
    if model_type in ['llama', 'gpt_neox', 'gpt_bigcode', 'mpt']:
        return True
    if model_type in ['RefinedWeb', 'RefinedWebModel', 'falcon']:
        return False

    raise Exception('Model "' + model_name + '" has unknown model type "' + model_type + '"')

def is_tgi_supported(model_name: str):
    if 'starchat' in model_name:
        return True
    if 'starcoder' in model_name:
        return True

    model_type = get_config_dict(model_name).model_type
    if model_type in ['RefinedWeb', 'RefinedWebModel', 'falcon']:
        return True

    raise Exception('Model "' + model_name + '" has unknown model type "' + model_type + '"')

def is_tgi_installed():
    return os.path.exists('text-generation-inference')

def get_huggingface_backend(model_path: str):
    if is_vllm_supported(model_path):
        return 'vllm'
    if is_tgi_supported(model_path):
        if is_tgi_installed():
            return 'tgi'
        else:
            print('WARNING: The model "' + model_path + '" can be greatly accelerated by text-generation-inference, but it is not installed.')
            return 'hf_transformers'
    return 'hf_transformers'

def compute_model_replies(model, conversations):
    if len(conversations) == 0:
        return []

    def reply(conversation_with_index):
        index, conversation = conversation_with_index

        if isinstance(conversation, list):
            reply = model.reply(conversation)
        elif isinstance(conversation, dict):
            reply = model.reply(conversation['conversation'], temperature=conversation['temperature'])
        else:
            raise

        return index, reply

    with multiprocessing.pool.ThreadPool(min(model.num_threads, len(conversations))) as pool:
        iterator = pool.imap_unordered(reply, enumerate(conversations))
        replies_with_indices = list(tqdm.tqdm(iterator, total=len(conversations)))

    return [reply_with_index[1] for reply_with_index in sorted(replies_with_indices, key=lambda item: item[0])]

def switch_gpu_model_type(new_model_type):
    unload_model_functions = {
        'hf_transformers': evaluation.models.huggingface_backends.hf_transformers.unload_model,
        'vllm': evaluation.models.huggingface_backends.vllm.unload_model,
        'fastchat': evaluation.models.fastchat.unload_model,
        'tgi': evaluation.models.huggingface_backends.tgi.unload_model,
    }

    for model_type, unload_model_function in unload_model_functions.items():
        if model_type == new_model_type:
            continue
        unload_model_function()

def unload_model(*args, **kwargs):
    switch_gpu_model_type(None)
