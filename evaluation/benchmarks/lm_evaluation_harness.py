import os
import json

import lm_eval.evaluator

def evaluate_model(model_type, model_name):
    output_path = os.path.join('./reports/lm-evaluation-harness', model_name.replace('/', '--') + '.json')
    if os.path.exists(output_path):
        return

    tasks = ['openbookqa', 'arc_easy', 'winogrande', 'hellaswag', 'arc_challenge', 'piqa', 'boolq']

    if model_type in ['open-assistant', 'guanaco', 'falcon']:
        print('lm-evaluation-harness: Evaluating', model_name)
        results = lm_eval.evaluator.simple_evaluate('hf-causal-experimental', tasks=tasks,
            model_args='pretrained=' + model_name + ',dtype="float16",trust_remote_code=True')
    elif model_type == 'openai':
        return

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def evaluate_models(models):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
