#!/usr/bin/env python3

import os
import json
from typing import Union
from evals.api import CompletionFn, CompletionResult
from evals.registry import Registry
from evals.cli.oaieval import get_parser, run

open_assistant_models = [
    'OpenAssistant/oasst-rlhf-3-llama-30b-5k-steps',
    'oasst-sft-7-llama-30b',
    'oasst-sft-7e3-llama-30b',
    'OpenAssistant/llama-30b-sft-v8-2.5k-steps',
    'OpenAssistant/pythia-12b-sft-v8-7k-steps',
]

def replace_model_name_slashes(model_name):
    return model_name.replace('/', '--')

def prompt_to_string(prompt, tokenizer):
    if isinstance(prompt, str):
        return '<|prompter|>' + prompt + tokenizer.eos_token + '<|assistant|>'

    prompt_str = ''
    previous_was_system_prompter = False
    for item in prompt:
        role = item['role']
        content = item['content']
        if role == 'system' and 'name' not in item:
            prompt_str += '<|prompter|>' + content
            previous_was_system_prompter = True
        elif role == 'assistant' or (role == 'system' and item['name'] == 'example_assistant'):
            prompt_str += '<|assistant|>' + content + tokenizer.eos_token
        elif (role == 'system' and item['name'] == 'example_user') or role == 'user':
            if previous_was_system_prompter:
                prompt_str += '\n\n' + content + tokenizer.eos_token
                previous_was_system_prompter = False
            else:
                prompt_str += '<|prompter|>' + content + tokenizer.eos_token
        else:
            raise
    prompt_str += '<|assistant|>'
    return prompt_str

class OpenAssistantCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]

class OpenAssistantCompletionFn(CompletionFn):
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def model_output(self, prompt_str):
        inputs = self.tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True, max_length=2047 - 400).to(0)

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        outputs = self.model.generate(
            **inputs,
            early_stopping=True,
            max_new_tokens=400,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.2,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=2048,
        )

        output = self.tokenizer.decode(outputs[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        reply = output.split('<|assistant|>')[-1].replace(self.tokenizer.eos_token, '').strip()
        return reply

    def __call__(
        self,
        prompt: Union[str, list[dict[str, str]]],
        **kwargs,
    ) -> OpenAssistantCompletionResult:
        prompt_string = prompt_to_string(prompt, self.tokenizer)
        generated_model_output = self.model_output(prompt_string)
        return OpenAssistantCompletionResult(generated_model_output)

class RegistryWithOpenAssistant(Registry):
    def __init__(self):
        super().__init__()

        self.tokenizers = {}
        self.models = {}

    def make_completion_fn(self, model_name: str) -> CompletionFn:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert model_name in open_assistant_models

        if model_name not in self.tokenizers:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval().cuda()

        return OpenAssistantCompletionFn(self.tokenizers[model_name], self.models[model_name])

    api_model_ids = []

def run_single_eval(registry, model_name, eval_name):
    parser = get_parser()
    args = parser.parse_args([
        model_name,
        eval_name,
        '--record_path', os.path.join('reports', replace_model_name_slashes(model_name), eval_name + '.json'),
    ])

    import logging
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        filename=args.log_to_file if args.log_to_file else None,
    )

    import openai
    logging.getLogger("openai").setLevel(logging.WARN)
    if hasattr(openai.error, "set_display_cause"):
        openai.error.set_display_cause()

    run(args, registry)

def run_multiple_evals(registry, model_name, evals):
    ignored_evals = [
        'best.dev.v0', # Compares multiple models
        'positive-binary-operations.test.v1', # KeyError: 'sample'
        'spider-sql.dev.v0', # TypeError: ModelBasedClassify.__init__() missing 1 required positional argument: 'modelgraded_spec'
        'svg_understanding.v0', # CUDA out of memory
        'stock-options-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_options_iron_butteryfly_spread.jsonl
        'stock-option-terms-inverse-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_option_terms_inverse_iron_butteryfly_spread.jsonl
        'joke-fruits-v2.dev.v0', # Buggy in openai/evals itself due to removed format_type feature that is still used by this eval
    ]

    for eval in evals:
        if os.path.exists(os.path.join('reports', replace_model_name_slashes(model_name), eval.key + '.json')):
            continue
        if eval.key in ignored_evals:
            continue
        print('Now evaluating', eval.key)
        run_single_eval(registry, model_name, eval.key)

def run_eval_set(registry, model_name, eval_set_name):
    run_multiple_evals(registry, model_name, registry.get_evals(registry.get_eval_set(eval_set_name).evals))

def run_all_evals(registry, model_name):
    run_multiple_evals(registry, model_name, registry.get_evals(['*']))

def build_reports_index(model_name):
    specs_and_final_reports = {}
    for filename in os.listdir(os.path.join('reports', replace_model_name_slashes(model_name))):
        if filename == '__index__.json':
            continue
        with open(os.path.join('reports', replace_model_name_slashes(model_name), filename), 'r') as f:
            spec_and_final_report = f.read().split('\n')[:2]
            spec = spec_and_final_report[0]
            final_report = spec_and_final_report[1]
            specs_and_final_reports[filename] = { 'spec': json.loads(spec)['spec'], 'final_report': json.loads(final_report)['final_report'] }
    with open(os.path.join('reports', replace_model_name_slashes(model_name), '__index__.json'), 'w') as f:
        json.dump(specs_and_final_reports, f, indent=4)

def evaluate_model(model_name):
    os.environ['EVALS_THREADS'] = '1'
    os.environ['EVALS_THREAD_TIMEOUT'] = '999999'

    if model_name in open_assistant_models:
        registry = RegistryWithOpenAssistant()
    else:
        registry = Registry()

    run_all_evals(registry, model_name)
    build_reports_index(model_name)

if __name__ == '__main__':
    evaluate_model('OpenAssistant/oasst-rlhf-3-llama-30b-5k-steps')
