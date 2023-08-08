import os
import json
import statistics

from evaluation.benchmarks.utils import model_name_to_filename
from evaluation.models.models import get_dtype

def run_evaluation(*, model_name, model_args):
    import lm_eval.evaluator

    tasks = ['openbookqa', 'arc_easy', 'winogrande', 'hellaswag', 'arc_challenge', 'piqa', 'boolq']

    kwargs = {}
    if 'tokenizer' in model_args:
        kwargs['tokenizer'] = model_args['tokenizer']

    lm_eval_model_args = ','.join([k + '=' + str(v) for k, v in ({
        'pretrained': model_name,
        'dtype': str(get_dtype(model_name)).replace('torch.', ''),
        'trust_remote_code': True,
        'use_accelerate': True,
        **kwargs,
    }).items()])

    print(model_name + ' :: LM-Eval :: Evaluating')

    return lm_eval.evaluator.simple_evaluate('hf-causal-experimental', tasks=tasks, model_args=lm_eval_model_args)

def evaluate_model(model_type, model_name, model_args, evaluation_id):
    if model_type == 'openai':
        return

    output_folder = os.path.join('./reports/lm-evaluation-harness', model_name_to_filename(model_name), evaluation_id)
    os.makedirs(output_folder, exist_ok=True)

    gpt4all_output_filepath = os.path.join(output_folder, 'gpt4all.json')
    if os.path.exists(gpt4all_output_filepath):
        with open(gpt4all_output_filepath) as f:
            results = json.load(f)
    else:
        results = run_evaluation(model_name=model_name, model_args=model_args)
        with open(gpt4all_output_filepath, 'w') as f:
            json.dump(results, f, indent=4)

    total_scores_filepath = os.path.join(output_folder, 'total.json')
    if os.path.exists(total_scores_filepath):
        return

    scores = { 'tasks': {} }
    for task_name in results['results'].keys():
        task_results = results['results'][task_name]
        scores['tasks'][task_name] = task_results['acc_norm'] if 'acc_norm' in task_results else task_results['acc']

    scores['average'] = statistics.mean(scores['tasks'].values())

    with open(total_scores_filepath, 'w') as f:
        json.dump(scores, f, indent=4)
