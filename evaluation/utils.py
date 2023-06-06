from evaluation.models.open_ai import OpenAI
from evaluation.models.open_assistant import OpenAssistant
from evaluation.models.guanaco import Guanaco
from evaluation.models.falcon import Falcon
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

def create_model(model_type: str, model_name: str):
    if model_type == 'openai':
        return OpenAI(model_name)
    if model_type == 'open-assistant':
        return OpenAssistant(model_name)
    if model_type == 'guanaco':
        return Guanaco(model_name)
    if model_type == 'falcon':
        return Falcon(model_name)
    if model_type == 'alpaca':
        return Alpaca(model_name)
