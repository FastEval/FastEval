import torch

from evaluation.models.open_ai import OpenAI
from evaluation.models.open_assistant import OpenAssistant
from evaluation.models.guanaco import Guanaco
from evaluation.models.falcon_instruct import FalconInstruct
from evaluation.models.alpaca import Alpaca

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
    if model_type == 'alpaca':
        return Alpaca

def create_model(model_type: str, model_name: str):
    return get_model_class(model_type)(model_name)

def get_dtype(model_type: str, model_name: str):
    if model_type == 'base':
        if 'llama' in model_name:
            return torch.float16
        raise
    return get_model_class(model_type).get_dtype(model_name)
