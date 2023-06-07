import os
import json

import lm_eval.evaluator

from ..utils import replace_model_name_slashes

def evaluate_model(model_type, model_name):
    output_path = os.path.join('./reports/lm-evaluation-harness', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(output_path):
        return

    tasks = ['openbookqa', 'arc_easy', 'winogrande', 'hellaswag', 'arc_challenge', 'piqa', 'boolq']

    if model_type == 'openai':
        return

    print('lm-evaluation-harness: Evaluating', model_name)
    if model_type == 'falcon-instruct':
        dtype = 'bfloat16'
    else:
        dtype = 'float16'
    results = lm_eval.evaluator.simple_evaluate('hf-causal-experimental', tasks=tasks,
        model_args='pretrained=' + model_name + ',dtype="' + dtype + '",trust_remote_code=True,use_accelerate=True')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def evaluate_models(models):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
