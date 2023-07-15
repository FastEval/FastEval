#!/usr/bin/env python3

import os
import json
import argparse

import evaluation.args
from evaluation import benchmarks
from evaluation.utils import changed_exit_handlers
from evaluation.models.models import unload_model

def merge_models_and_benchmarks_to_evaluate(existing_models_and_benchmarks, new_models, new_benchmarks):
    additional_models = set()
    for model_type, model_name in new_models:
        for benchmark in new_benchmarks:
            for item in existing_models_and_benchmarks:
                if item['model_name'] == model_name:
                    assert item['model_type'] == model_type
                    if benchmark in item['benchmarks']:
                        break
                    item['benchmarks'].insert(0, benchmark)
                    break
            else:
                additional_models.add((model_type, model_name))
    for model_type, model_name in additional_models:
        existing_models_and_benchmarks.insert(0, {
            'model_type': model_type,
            'model_name': model_name,
            'benchmarks': new_benchmarks,
        })
    return existing_models_and_benchmarks

def main():
    all_benchmarks = ['openai-evals', 'mt-bench', 'lm-evaluation-harness', 'human-eval-plus', 'cot']

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmarks', choices=['all'] + all_benchmarks, nargs='*', default='all')
    parser.add_argument('-m', '--models', nargs='+')
    parser.add_argument('--force-backend', choices=['hf_transformers', 'tgi', 'vllm'], required=False)
    parser.add_argument('--tgi-max-batch-total-tokens', type=int, default=16000)
    args = parser.parse_args()

    evaluation.args.cmd_arguments = args

    if 'all' in args.benchmarks:
        args.benchmarks = all_benchmarks
    if args.models is None:
        args.models = []

    args.models = [model.split(':') for model in args.models]

    if os.path.exists('reports/__index__.json'):
        with open('reports/__index__.json') as f:
            models_and_benchmarks = merge_models_and_benchmarks_to_evaluate(json.load(f), args.models, args.benchmarks)

    evaluation_functions = {
        'cot': benchmarks.cot.evaluate_model,
        'human-eval-plus': benchmarks.human_eval_plus.evaluate_model,
        'openai-evals': benchmarks.openai_evals.evaluate_model,
        'mt-bench': benchmarks.mt_bench.evaluate_model,
    }

    with changed_exit_handlers():
        for item in models_and_benchmarks:
            for benchmark_name, evaluation_function in evaluation_functions.items():
                if benchmark_name in item['benchmarks']:
                    evaluation_function(item['model_type'], item['model_name'])
            unload_model()

    for item in models_and_benchmarks:
        if 'lm-evaluation-harness' in item['benchmarks']:
            benchmarks.lm_evaluation_harness.evaluate_model(item['model_type'], item['model_name'])

    with open(os.path.join('reports', '__index__.json'), 'w') as f:
        json.dump(models_and_benchmarks, f, indent=4)

if __name__ == '__main__':
    main()
