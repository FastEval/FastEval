#!/usr/bin/env python3

import argparse

from evaluation import benchmarks

def main():
    all_benchmarks = ['openai-evals', 'vicuna']

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', choices=['all'] + all_benchmarks, nargs='*', default='all')
    parser.add_argument('-m', '--model', nargs='+')
    args = parser.parse_args()

    if 'all' in args.benchmark:
        args.benchmark = all_benchmarks

    if 'openai-evals' in args.benchmark:
        for model in args.model:
            benchmarks.openai_evals.evaluate_model(model)
    if 'vicuna' in args.benchmark:
        benchmarks.vicuna.evaluate_models(args.model)

if __name__ == '__main__':
    main()
