import atexit
import signal
import contextlib.contextmanager

def replace_model_name_slashes(model_name: str) -> str:
    """
    The model name can be something like OpenAssistant/oasst-sft-1-pythia-12b.
    The path where we store evaluation results should depend on the model name,
    but paths can't include '/', so we need to replace that.
    """

    return model_name.replace('/', '--')

def undo_replace_model_name_slashes(model_name: str) -> str:
    return model_name.replace('--', '/')

@contextlib.contextmanager
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
