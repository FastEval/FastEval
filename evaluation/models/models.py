import os

import evaluation.utils
import evaluation.args

config_dict_cache = {}

def create_model(model_type: str, model_name: str, model_args: dict[str, str], **kwargs):
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
    import transformers

    if model_name in config_dict_cache:
        return config_dict_cache[model_name]
    config_dict = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config_dict_cache[model_name] = config_dict
    return config_dict

def get_dtype(model_name: str):
    return get_config_dict(model_name).torch_dtype

def get_supported_inference_backends(model_name: str):
    if 'starchat' in model_name:
        # vLLM currently does not support starchat.
        # See https://github.com/vllm-project/vllm/issues/380
        return ['tgi', 'hf_transformers']

    generally_supported_model_types = [
        'llama', # LLaMA & LLaMA-2
        'gpt_neox', # EleutherAI Pythia models
        'gpt_bigcode', # Starcoder
        'mpt', # MPT models from MosaicML

        # All of these are some variant of falcon from TII
        'RefinedWeb',
        'RefinedWebModel',
        'falcon',
    ]

    model_type = get_config_dict(model_name).model_type
    if model_type in generally_supported_model_types:
        return ['vllm', 'tgi', 'hf_transformers']

    return []

def is_tgi_installed():
    return os.path.exists('text-generation-inference')

def get_huggingface_backend(model_path: str):
    forced_backend = evaluation.args.cmd_arguments.force_backend
    if forced_backend is not None:
        return forced_backend

    supported_backends = get_supported_inference_backends()

    if 'vllm' in supported_backends:
        return 'vllm'

    if 'tgi' in supported_backends:
        if is_tgi_installed():
            return 'tgi'
        print('WARNING: The model "' + model_path + '" can be greatly accelerated by text-generation-inference, but it is not installed.')

    if 'hf_transformers' in supported_backends:
        return 'hf_transformers'

    raise Exception('Model "' + model_name + '" has unknown model type "' + model_type + '"')

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
    import evaluation.models.fastchat
    import evaluation.models.huggingface_backends.hf_transformers
    import evaluation.models.huggingface_backends.vllm_backend
    import evaluation.models.huggingface_backends.tgi

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

def unload_model():
    switch_gpu_model_type(None)
