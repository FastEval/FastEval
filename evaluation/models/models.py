import os

import torch
import transformers

import evaluation.args
import evaluation.utils
import evaluation.models.fastchat
import evaluation.models.huggingface_backends.hf_transformers
import evaluation.models.huggingface_backends.vllm_backend
import evaluation.models.huggingface_backends.tgi
from evaluation.models.debug import Debug
from evaluation.models.open_ai import OpenAI
from evaluation.models.fastchat import Fastchat
from evaluation.models.open_assistant import OpenAssistant
from evaluation.models.guanaco import Guanaco
from evaluation.models.falcon_instruct import FalconInstruct
from evaluation.models.alpaca_without_prefix import AlpacaWithoutPrefix
from evaluation.models.alpaca_with_prefix import AlpacaWithPrefix
from evaluation.models.chatml import ChatML
from evaluation.models.starchat import Starchat
from evaluation.models.llama2_chat import Llama2Chat
from evaluation.models.free_willy2 import FreeWilly2
from evaluation.models.dolphin import Dolphin
from evaluation.models.openchat_llama2_v1 import OpenchatLlama2V1

config_dict_cache = {}

def create_model(model_type: str, model_name: str, model_args: dict[str, str], **kwargs):
    model_classes = {
        'debug': Debug,
        'openai': OpenAI,
        'fastchat': Fastchat,
        'open-assistant': OpenAssistant,
        'guanaco': Guanaco,
        'falcon-instruct': FalconInstruct,
        'alpaca-without-prefix': AlpacaWithoutPrefix,
        'alpaca-with-prefix': AlpacaWithPrefix,
        'chatml': ChatML,
        'starchat': Starchat,
        'llama2-chat': Llama2Chat,
        'free-willy2': FreeWilly2,
        'dolphin': Dolphin,
        'openchat-llama2-v1': OpenchatLlama2V1,
    }

    if model_type not in model_classes:
        raise Exception('Unknown model type "' + model_type + '"')

    model_class = model_classes[model_type]

    return model_class(model_name, **model_args, **kwargs)

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
    forced_backend = evaluation.args.cmd_arguments.force_backend
    if forced_backend is not None:
        return forced_backend

    if is_vllm_supported(model_path):
        return 'vllm'
    if is_tgi_supported(model_path):
        if is_tgi_installed():
            return 'tgi'
        else:
            print('WARNING: The model "' + model_path + '" can be greatly accelerated by text-generation-inference, but it is not installed.')
            return 'hf_transformers'
    return 'hf_transformers'

def compute_model_replies(model, conversations, *, desc=None):
    if len(conversations) == 0:
        return []

    def reply(conversation):
        if isinstance(conversation, list):
            reply = model.reply(conversation)
        elif isinstance(conversation, dict):
            reply = model.reply(conversation['conversation'], temperature=conversation['temperature'])
        else:
            raise

        return reply

    return evaluation.utils.process_with_thread_pool(
        num_threads=model.num_threads,
        items=conversations,
        process_function=reply,
        desc=desc,
    )

def switch_gpu_model_type(new_model_type):
    unload_model_functions = {
        'hf_transformers': evaluation.models.huggingface_backends.hf_transformers.unload_model,
        'vllm': evaluation.models.huggingface_backends.vllm_backend.unload_model,
        'fastchat': evaluation.models.fastchat.unload_model,
        'tgi': evaluation.models.huggingface_backends.tgi.unload_model,
    }

    for model_type, unload_model_function in unload_model_functions.items():
        if model_type == new_model_type:
            continue
        unload_model_function()

def unload_model(*args, **kwargs):
    switch_gpu_model_type(None)
