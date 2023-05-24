#!/usr/bin/env python3

import os
import json
import argparse

from evaluation import benchmarks

def main():
    all_benchmarks = ['openai-evals', 'vicuna', 'lm-evaluation-harness']

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmarks', choices=['all'] + all_benchmarks, nargs='*', default='all')
    parser.add_argument('-m', '--models', nargs='+', required=True)
    args = parser.parse_args()

    if 'all' in args.benchmarks:
        args.benchmarks = all_benchmarks

    if 'openai-evals' in args.benchmarks:
        for model in args.models:
            benchmarks.openai_evals.evaluate_model(model)
    if 'vicuna' in args.benchmarks:
        benchmarks.vicuna.evaluate_models(args.models)
    if 'lm-evaluation-harness' in args.benchmarks:
        benchmarks.lm_evaluation_harness.evaluate_models(args.models)

    with open(os.path.join('reports', '__index__.json'), 'w') as f:
        json.dump(args.models, f, indent=4)

if __name__ == '__main__':
    main()
