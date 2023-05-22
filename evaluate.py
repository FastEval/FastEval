#!/usr/bin/env python3

import argparse

from evaluation import benchmarks

def main():
    all_benchmarks = ['openai-evals', 'vicuna']

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

if __name__ == '__main__':
    main()
