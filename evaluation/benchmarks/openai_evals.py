import os
import sys
import json
from typing import Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from evals.api import CompletionFn, CompletionResult
from evals.registry import Registry
from evals.cli.oaieval import get_parser, run

from evaluation.utils import replace_model_name_slashes

def prompt_to_string(prompt: Union[str, list[dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer):
    """
    Converts a prompt in the OpenAI evals conversation format to a string for consumption by OpenAssistant.
    See https://github.com/openai/evals/blob/main/evals/registry/data/README.md for the OpenAI evals format.
    The prompt input may also already be a string, in which it is just the prompter prompt.
    """

    if isinstance(prompt, str):
        return '<|prompter|>' + prompt + tokenizer.eos_token + '<|assistant|>'

    prompt_string = ''
    previous_was_system_prompter = False
    for item in prompt:
        role = item['role']
        content = item['content']
        if role == 'system' and 'name' not in item:
            prompt_string += '<|prompter|>' + content
            previous_was_system_prompter = True
        elif role == 'assistant' or (role == 'system' and item['name'] == 'example_assistant'):
            prompt_string += '<|assistant|>' + content + tokenizer.eos_token
        elif (role == 'system' and item['name'] == 'example_user') or role == 'user':
            if previous_was_system_prompter:
                prompt_string += '\n\n' + content + tokenizer.eos_token
                previous_was_system_prompter = False
            else:
                prompt_string += '<|prompter|>' + content + tokenizer.eos_token
        else:
            raise

    return prompt_string + '<|assistant|>'

class OpenAssistantCompletionResult(CompletionResult):
    """
    I don't know why this thing is needed just to wrap a single reply string.
    But that's what OpenAI evals requires, so we need it.
    """

    def __init__(self, reply: str):
        self.reply = reply

    def get_completions(self) -> list[str]:
        return [self.reply.strip()]

class OpenAssistantCompletionFn(CompletionFn):
    """
    See https://github.com/openai/evals/blob/main/docs/completion-fns.md
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel):
        self.tokenizer = tokenizer
        self.model = model

    def model_output(self, prompt: str):
        # TODO: max_length should be taken from the model and not hardcoded.
        model_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2047 - 400).to(0)

        # TODO: What is this for? Is this needed? I just took it from https://github.com/Open-Assistant/oasst-model-eval
        if 'token_type_ids' in model_input:
            del model_input['token_type_ids']

        model_output = self.model.generate(
            **model_input,
            early_stopping=True, # TODO: Why? Isn't this only for beam search? Also taken from https://github.com/Open-Assistant/oasst-model-eval
            min_new_tokens=1,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.2,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]

        # TODO: What's that truncation for? Also just taken from https://github.com/Open-Assistant/oasst-model-eval I think
        output_decoded = self.tokenizer.decode(model_output, truncate_before_pattern=[r'\n\n^#', "^'''", '\n\n\n'])
        reply = output_decoded.split('<|assistant|>')[-1].replace(self.tokenizer.eos_token, '').strip()
        return reply

    def __call__(
        self,
        prompt: Union[str, list[dict[str, str]]],
        **kwargs,
    ) -> OpenAssistantCompletionResult:
        prompt_as_string = prompt_to_string(prompt, self.tokenizer)
        model_reply = self.model_output(prompt_as_string)
        return OpenAssistantCompletionResult(model_reply)

class RegistryWithOpenAssistant(Registry):
    """
    OpenAI evals uses the `Registry` for keeping track of models that can be evaluated.
    While we could also register OpenAssistant models to that registry, we don't do that
    because that would require a specific project structure and also it wouldn't work
    without an OpenAI API key, even if OpenAI models are not actually needed.
    Therefore we overwrite this class to use OpenAssistant models.
    """

    def __init__(self):
        super().__init__()
        self.tokenizers: dict[str, transformers.PreTrainedTokenizerBase] = {}
        self.models: dict[str, transformers.PreTrainedModel] = {}

    def make_completion_fn(self, model_name: str) -> OpenAssistantCompletionFn:
        if model_name not in self.tokenizers:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            # TODO: Is the `torch_dtype` actually needed?
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto').eval()

        return OpenAssistantCompletionFn(self.tokenizers[model_name], self.models[model_name])

    # Prevent errors about OpenAI API key missing even if we don't use OpenAI models
    api_model_ids = []

def run_single_eval(registry: Registry, model_name: str, eval_name: str):
    """
    Evaluate the specified model on the single specified eval from OpenAI evals
    """

    run(get_parser().parse_args([
        model_name,
        eval_name,
        '--record_path', os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name), eval_name + '.json'),
    ]), registry)

def run_multiple_evals(registry: Registry, model_name: str, evals: list[str]):
    """
    Evaluate the specified model on the specified evals from OpenAI evals
    """

    non_working_evals = [
        'best.dev.v0', # Compares multiple models
        'positive-binary-operations.test.v1', # KeyError: 'sample'
        'spider-sql.dev.v0', # TypeError: ModelBasedClassify.__init__() missing 1 required positional argument: 'modelgraded_spec'
        'svg_understanding.v0', # CUDA out of memory
        'stock-options-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_options_iron_butteryfly_spread.jsonl
        'stock-option-terms-inverse-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_option_terms_inverse_iron_butteryfly_spread.jsonl
        'joke-fruits-v2.dev.v0', # Buggy in openai/evals itself due to removed format_type feature that is still used by this eval
    ]

    for eval in evals:
        if os.path.exists(os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name), eval + '.json')):
            continue
        if eval in non_working_evals:
            continue
        print('Now evaluating', eval)
        run_single_eval(registry, model_name, eval)

def create_reports_index_file(model_name: str):
    """
    Create a single file in reports/openai-evals/<model_name>/__index__.json that contains information about all the evals
    that this model was evaluated on. This index file is used for showing information on the website.
    """

    model_reports_path = os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name))

    reports_metadata = {}
    for report_filename in os.listdir(model_reports_path):
        if report_filename == '__index__.json':
            continue
        with open(os.path.join(model_reports_path, report_filename), 'r') as report_file:
            report_metadata = report_file.read().split('\n')[:2]
        reports_metadata[report_filename] = {
            'spec': json.loads(report_metadata[0])['spec'],
            'final_report': json.loads(report_metadata[1])['final_report'],
        }

    with open(os.path.join(model_reports_path, '__index__.json'), 'w') as reports_index_file:
        json.dump(reports_metadata, reports_index_file, indent=4)

def evaluate_model(model_name: str):
    """
    Evaluate the specified model on all the evals in OpenAI evals
    """

    os.environ['EVALS_THREADS'] = '1'
    os.environ['EVALS_THREAD_TIMEOUT'] = '999999'
    # os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    if model_name == 'gpt-3.5-turbo':
        registry = Registry()
    else:
        registry = RegistryWithOpenAssistant()

    run_multiple_evals(registry, model_name, [eval.key for eval in registry.get_evals(['*'])])
    create_reports_index_file(model_name)

def main():
    model_name = sys.argv[1]
    evaluate_model(model_name)

if __name__ == '__main__':
    main()
