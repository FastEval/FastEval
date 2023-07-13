import os
import json
import subprocess
import ast

from evalplus.data import get_human_eval_plus, write_jsonl

from evaluation.utils import replace_model_name_slashes
from evaluation.models.models import create_model, compute_model_replies

def postprocess_model_reply(model_reply):
    for item in ['```Python', '```python', '```py', '```']:
        if item in model_reply:
            model_reply = model_reply.split(item)[1].split('```')[0]
            break

    new_lines = []
    inside_function = False
    for line in model_reply.split('\n'):
        line = line.replace('\r', '')
        if line == '':
            new_lines.append(line)
        elif line.startswith('import ') or line.startswith('from '):
            new_lines.append(line)
        elif line.startswith('def '):
            new_lines.append(line)
            inside_function = True
        elif inside_function and (line.startswith(' ') or line.startswith('\t')):
            new_lines.append(line)
        else:
            inside_function = False

    model_reply = '\n'.join(new_lines)
    model_reply = model_reply.strip('\n')
    return model_reply

def create_conversation(prompt):
    return [
        ('user',
            'Please complete the following Python code. '
            'Provide the complete function implementation including the part that is already given as input. '
            'Do not provide anything else except the function code and implementation. '
            'Do not provide explanation, tests or example usage.'
            '\n\n'
            + prompt),
    ]

def evaluate_model(model_type, model_name):
    output_path = os.path.join('./reports/human-eval-plus', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(output_path):
        return

    model = create_model(model_type, model_name)

    dataset = get_human_eval_plus()
    task_ids = dataset.keys()
    prompts = [dataset[task_id]['prompt'] for task_id in task_ids]
    raw_replies = compute_model_replies(model, [create_conversation(prompt) for prompt in prompts])
    processed_replies = [{ 'task_id': task_id, 'completion': postprocess_model_reply(raw_replies[i]) }
        for i, task_id in enumerate(task_ids)]

    write_jsonl('human-eval-plus-tmp.jsonl', processed_replies)
    process_output = subprocess.run([
        'evalplus.evaluate',
        '--parallel', str(os.cpu_count()),
        '--min-time-limit', '10',
        '--gt-time-limit-factor', '10',
        '--dataset', 'humaneval',
        '--samples', 'human-eval-plus-tmp.jsonl'
    ], capture_output=True, text=True).stdout
    os.remove('human-eval-plus-tmp.jsonl')

    with open('human-eval-plus-tmp_eval_results.json') as f:
        results = json.load(f)['eval']

    samples_with_results = []
    for i, processed_reply in enumerate(processed_replies):
        task_id = processed_reply['task_id']
        result = results[task_id]['plus'][0][0]
        if result == 'failed':
            success = False
        elif result == 'success':
            success = True
        else:
            raise
        samples_with_results.append({
            'task_id': task_id,
            'prompt': prompts[i],
            'completion_processed': processed_reply['completion'],
            'completion_raw': raw_replies[i],
            'success': success,
        })

    os.remove('human-eval-plus-tmp_eval_results.json')

    process_output_lines = [line for line in process_output.split('\n') if line != '']
    assert process_output_lines[-2] == 'Base + Extra'
    score = ast.literal_eval(process_output_lines[-1])['pass@1']
    output = { 'replies': samples_with_results, 'score': score }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

def evaluate_models(models):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
