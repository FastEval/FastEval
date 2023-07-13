import atexit
import signal
from contextlib import contextmanager

import evaluation.models.models

def replace_model_name_slashes(model_name: str) -> str:
    """
    The model name can be something like OpenAssistant/oasst-sft-1-pythia-12b.
    The path where we store evaluation results should depend on the model name,
    but paths can't include '/', so we need to replace that.
    """

    return model_name.replace('/', '--')

def undo_replace_model_name_slashes(model_name: str) -> str:
    return model_name.replace('--', '/')

@contextmanager
def changed_exit_handlers():
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)

    atexit.register(evaluation.models.models.unload_model)
    signal.signal(signal.SIGTERM, evaluation.models.models.unload_model)
    signal.signal(signal.SIGINT, evaluation.models.models.unload_model)

    yield

    atexit.unregister(evaluation.models.models.unload_model)
    signal.signal(signal.SIGTERM, previous_sigterm)
    signal.signal(signal.SIGINT, previous_sigint)
