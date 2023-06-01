import os
import json

import tqdm
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from human_eval.data import HUMAN_EVAL
from evaluation.utils import replace_model_name_slashes, create_model

def postprocess_model_reply(model_reply):
    model_reply = model_reply.split('```python')[-1].split('```py')[-1].split('```Python')[-1]
    if model_reply.count('```') >= 2:
        model_reply =  model_reply.split('```')[-1]
    model_reply = model_reply.split('```')[0]
    model_reply = model_reply.strip('\n')
    return model_reply

def score_model_replies(tmpfile):
    return evaluate_functional_correctness(tmpfile, [1], 4, 3.0, HUMAN_EVAL)

def evaluate_model(model_type, model_name):
    output_path = os.path.join('./reports/human-eval', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(output_path):
        return

    model = create_model(model_type, model_name)

    dataset = read_problems()
    samples = []
    for task_id in tqdm.tqdm(dataset):
        prompt = dataset[task_id]['prompt']
        reply = model.reply([
            ('user',
                'Please complete the following Python code. '
                'Do NOT provide any additional explanation or tests. '
                'Also do NOT output things like "Sure!" or "Here you go!" or similar things. '
                'Just provide the code without anything else.'
                '\n\n'
                + prompt),
        ])

        reply = postprocess_model_reply(reply)
        print('@@@@@@@@@@@@@@@@@@@@\n' + reply + '\n@@@@@@@@@@@@@@@@@@@@')
        samples.append({ 'task_id': task_id, 'completion': reply })

    with open('tmp', 'w') as f:
        f.write('\n'.join([json.dumps(sample) for sample in samples]))

    scores = score_model_replies('tmp')
    os.remove('tmp')
    with open('tmp_results.jsonl') as f:
        content = [json.loads(line) for line in f.read().split('\n') if line != '']
    for sample in content:
        sample['prompt'] = dataset[sample['task_id']]['prompt']
    output = {
        'replies': content,
        'scores': scores
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    os.remove('tmp_results.jsonl')

def evaluate_models(models):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
