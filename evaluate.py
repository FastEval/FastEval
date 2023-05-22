#!/usr/bin/env python3

import argparse

from evaluation.benchmarks.openai_evals import evaluate_model
from evaluation.benchmarks.vicuna import evaluate_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', choices=['openai-evals', 'vicuna', 'all'], nargs='*', default='all')
    parser.add_argument('-m', '--model', nargs='+')
    args = parser.parse_args()

    if args.benchmark == 'openai-evals':
        for model in args.model:
            evaluate_model(model)
    elif args.benchmark == 'vicuna':
        evaluate_models([(('open-ai' if model == 'gpt-3.5-turbo' else 'open-assistant'), model) for model in args.model])

if __name__ == '__main__':
    main()

# ./main.py -b openai-evals -m gpt-3.5-turbo -m OpenAssistant/oasst-sft-1-pythia-12b
