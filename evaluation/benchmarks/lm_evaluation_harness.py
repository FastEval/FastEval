import os
import json

import lm_eval.evaluator

from evaluation.utils import get_model_type

def evaluate_model(model):
    output_path = os.path.join('./reports/lm-evaluation-harness', model.replace('/', '--') + '.json')
    if os.path.exists(output_path):
        return

    tasks = ['openbookqa', 'arc_easy', 'winogrande', 'hellaswag', 'arc_challenge', 'piqa', 'boolq']

    model_type = get_model_type(model)

    print('lm-evaluation-harness: Evaluating', model, model_type)

    if model_type == 'huggingface':
        results = lm_eval.evaluator.simple_evaluate('hf-causal-experimental', model_args='pretrained=' + model + ',dtype="float16"', tasks=tasks)
    elif model_type == 'openai':
        return # TODO

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def evaluate_models(models):
    for model_name in models:
        evaluate_model(model_name)
