from evaluation.models.open_ai import OpenAI
from evaluation.models.open_assistant import OpenAssistant

def replace_model_name_slashes(model_name: str) -> str:
    """
    The model name can be something like OpenAssistant/oasst-sft-1-pythia-12b.
    The path where we store evaluation results should depend on the model name,
    but paths can't include '/', so we need to replace that.
    """

    return model_name.replace('/', '--')

def undo_replace_model_name_slashes(model_name: str) -> str:
    return model_name.replace('--', '/')

def create_model(model_name: str):
    if model_name == 'gpt-3.5-turbo':
        return OpenAI(model_name)
    return OpenAssistant(model_name)
