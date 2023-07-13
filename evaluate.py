#!/usr/bin/env python3

import os
import json
import argparse

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
    all_benchmarks = ['openai-evals', 'vicuna', 'mt-bench', 'lm-evaluation-harness', 'human-eval-plus', 'cot']

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmarks', choices=['all'] + all_benchmarks, nargs='*', default='all')
    parser.add_argument('-m', '--models', nargs='+')
    parser.add_argument('-e', '--exclude-vicuna-reviews', default=False, action='store_true')
    args = parser.parse_args()

    if 'all' in args.benchmarks:
        args.benchmarks = all_benchmarks
    if args.models is None:
        args.models = []

    args.models = [model.split(':') for model in args.models]

    if os.path.exists('reports/__index__.json'):
        with open('reports/__index__.json') as f:
            models_and_benchmarks = merge_models_and_benchmarks_to_evaluate(json.load(f), args.models, args.benchmarks)

    with changed_exit_handlers():
        benchmarks.cot.evaluate_models([(item['model_type'], item['model_name'])
            for item in models_and_benchmarks if 'cot' in item['benchmarks']])
        benchmarks.human_eval_plus.evaluate_models([(item['model_type'], item['model_name'])
            for item in models_and_benchmarks if 'human-eval-plus' in item['benchmarks']])
        benchmarks.openai_evals.evaluate_models([(item['model_type'], item['model_name'])
            for item in models_and_benchmarks if 'openai-evals' in item['benchmarks']])
        benchmarks.mt_bench.evaluate_models([(item['model_type'], item['model_name'])
            for item in models_and_benchmarks if 'mt-bench' in item['benchmarks']])

    benchmarks.lm_evaluation_harness.evaluate_models([(item['model_type'], item['model_name'])
        for item in models_and_benchmarks if 'lm-evaluation-harness' in item['benchmarks']])

    with changed_exit_handlers():
        benchmarks.vicuna.evaluate_models([(item['model_type'], item['model_name'])
            for item in models_and_benchmarks if 'vicuna' in item['benchmarks']], exclude_reviews=args.exclude_vicuna_reviews)

    unload_model()

    with open(os.path.join('reports', '__index__.json'), 'w') as f:
        json.dump(models_and_benchmarks, f, indent=4)

if __name__ == '__main__':
    main()
