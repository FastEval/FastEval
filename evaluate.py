#!/usr/bin/env python3

import os
import json
import argparse
import threading
import uuid

import evaluation.args
from evaluation import benchmarks
from evaluation.utils import changed_exit_handlers
from evaluation.models.models import unload_model

def merge_models_and_benchmarks_to_evaluate(existing_models_and_benchmarks, new_model_type, new_model_name, new_benchmarks, model_args):
    if new_model_name is None:
        return existing_models_and_benchmarks

    model_args = { k: v for k, v in model_args.items() if v is not None }

    inserted_into_existing_entry = False
    for item in existing_models_and_benchmarks:
        if item['model_name'] != new_model_name:
            continue
        if item['model_type'] != new_model_type:
            raise # TODO: Only for now. Later we allow this.
        if item['model_args'] != model_args:
            raise # TODO: Only for now. Later we allow this.
        for benchmark in new_benchmarks:
            if benchmark not in item['benchmarks']:
                item['benchmarks'].insert(0, benchmark)
        inserted_into_existing_entry = True

    if inserted_into_existing_entry:
        return existing_models_and_benchmarks

    existing_models_and_benchmarks.insert(0, {
        'id': str(uuid.uuid4()),
        'model_type': new_model_type,
        'model_name': new_model_name,
        'benchmarks': new_benchmarks,
        'model_args': model_args,
    })

    return existing_models_and_benchmarks

def main():
    all_benchmarks = ['mt-bench', 'lm-evaluation-harness', 'human-eval-plus', 'cot']

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmarks', choices=['all'] + all_benchmarks, nargs='*', default='all')
    parser.add_argument('-t', '--model-type')
    parser.add_argument('-m', '--model-name')
    parser.add_argument('--model-tokenizer')
    parser.add_argument('--model-default-system-message')
    parser.add_argument('--force-backend', choices=['hf_transformers', 'tgi', 'vllm'], required=False)
    parser.add_argument('--tgi-max-batch-total-tokens', type=int, default=None)
    args = parser.parse_args()

    evaluation.args.cmd_arguments = args

    if 'all' in args.benchmarks:
        args.benchmarks = all_benchmarks

    model_args = {
        'tokenizer': args.model_tokenizer,
        'default_system_message': args.model_default_system_message,
    }

    if os.path.exists('reports/__index__.json'):
        with open('reports/__index__.json') as f:
            models_and_benchmarks = merge_models_and_benchmarks_to_evaluate(json.load(f), args.model_type, args.model_name, args.benchmarks, model_args)

    evaluation_functions = [
        ('mt-bench', benchmarks.mt_bench.evaluate_model),
        ('human-eval-plus', benchmarks.human_eval_plus.evaluate_model),
        ('cot', benchmarks.cot.evaluate_model),
    ]

    with changed_exit_handlers():
        for item in models_and_benchmarks:
            for benchmark_name, evaluation_function in evaluation_functions:
                if benchmark_name in item['benchmarks']:
                    evaluation_function(item['model_type'], item['model_name'], item['model_args'])
            unload_model()

    for item in models_and_benchmarks:
        if 'lm-evaluation-harness' in item['benchmarks']:
            benchmarks.lm_evaluation_harness.evaluate_model(item['model_type'], item['model_name'], item['model_args'])

    for thread in threading.enumerate():
        if thread.daemon:
            continue

        try:
            thread.join()
        except RuntimeError as error:
            if 'cannot join current thread' in error.args[0]: # main thread
                pass
            else:
                raise

    with open(os.path.join('reports', '__index__.json'), 'w') as f:
        json.dump(models_and_benchmarks, f, indent=4)

if __name__ == '__main__':
    main()
