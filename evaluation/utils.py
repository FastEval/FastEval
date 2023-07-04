import torch
import tqdm
import multiprocessing.pool

import evaluation.models.fastchat

from evaluation.models.open_ai import OpenAI
from evaluation.models.open_assistant import OpenAssistant
from evaluation.models.guanaco import Guanaco
from evaluation.models.falcon_instruct import FalconInstruct
from evaluation.models.alpaca_without_prefix import AlpacaWithoutPrefix
from evaluation.models.alpaca_with_prefix import AlpacaWithPrefix
from evaluation.models.chatml import ChatML
from evaluation.models.fastchat import Fastchat

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
    if model_type == 'openai':
        return OpenAI
    if model_type == 'open-assistant':
        return OpenAssistant
    if model_type == 'guanaco':
        return Guanaco
    if model_type == 'falcon-instruct':
        return FalconInstruct
    if model_type == 'alpaca-without-prefix':
        return AlpacaWithoutPrefix
    if model_type == 'alpaca-with-prefix':
        return AlpacaWithPrefix
    if model_type == 'chatml':
        return ChatML
    if model_type == 'fastchat':
        return Fastchat

def create_model(model_type: str, model_name: str):
    return get_model_class(model_type)(model_name)

def get_dtype(model_type: str, model_name: str):
    if model_type == 'base':
        if 'llama' in model_name:
            return torch.float16
        if 'falcon' in model_name:
            return torch.bfloat16
        if 'mpt' in model_name:
            return torch.bfloat16
        raise
    return get_model_class(model_type).get_dtype(model_name)

def compute_model_replies(model, conversations, *, num_threads=10):
    def reply(conversation_with_index):
        index, conversation = conversation_with_index
        reply = model.reply(conversation)
        return index, reply

    with multiprocessing.pool.ThreadPool(num_threads) as pool:
        iterator = pool.imap_unordered(reply, enumerate(conversations))
        replies_with_indices = list(tqdm.tqdm(iterator, total=len(conversations)))

    return [reply_with_index[1] for reply_with_index in sorted(replies_with_indices, key=lambda item: item[0])]

def register_exit_handlers():
    evaluation.models.fastchat.register_exit_handler()
