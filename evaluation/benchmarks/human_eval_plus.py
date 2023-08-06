import os
import json
import subprocess
import ast
import uuid

from evalplus.data import get_human_eval_plus, write_jsonl

from evaluation.benchmarks.utils import model_name_to_filename
from evaluation.models.models import create_model, compute_model_replies
from evaluation.constants import HUMAN_EVAL_PLUS_TEMPERATURE

N = 3

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

def compute_model_answers(*, model_type, model_name, model_args, output_folder):
    output_file = os.path.join(output_folder, 'answers.json')
    if os.path.exists(output_file):
        return

    model = create_model(model_type, model_name, model_args)

    dataset = get_human_eval_plus()
    task_ids = list(dataset.keys()) * N
    prompts = [dataset[task_id]['prompt'] for task_id in task_ids]
    raw_replies = compute_model_replies(model, [{
        'conversation': create_conversation(prompt),
        'temperature': HUMAN_EVAL_PLUS_TEMPERATURE,
    } for prompt in prompts], progress_bar_description=model_name + ' :: HumanEval+ :: Computing model replies')

    processed_replies = [{
        'task_id': task_id,
        'prompt': prompts[i],
        'completion_processed': postprocess_model_reply(raw_replies[i]),
        'completion_raw': raw_replies[i],
    } for i, task_id in enumerate(task_ids)]

    with open(output_file, 'w') as f:
        json.dump(processed_replies, f, indent=4)

def compute_scores(*, output_folder):
    output_file = os.path.join(output_folder, 'scores.json')
    if os.path.exists(output_file):
        return

    answers_file = os.path.join(output_folder, 'answers.json')
    with open(answers_file) as f:
        answers = json.load(f)

    human_eval_plus_input = [{
        'task_id': item['task_id'],
        'completion': item['completion_processed'],
    } for item in answers]

    tmp_folder = os.path.join('.tmp', 'human-eval-plus')
    os.makedirs(tmp_folder, exist_ok=True)

    tmp_uuid = str(uuid.uuid4())
    tmp_file_input = os.path.join(tmp_folder, tmp_uuid + '.jsonl')
    write_jsonl(tmp_file_input, human_eval_plus_input)

    process_output = subprocess.run([
        'evalplus.evaluate',
        '--parallel', str(os.cpu_count()),
        '--min-time-limit', '10',
        '--gt-time-limit-factor', '10',
        '--dataset', 'humaneval',
        '--samples', tmp_file_input,
    ], capture_output=True, text=True).stdout

    os.remove(tmp_file_input)

    tmp_file_output = os.path.join(tmp_folder, tmp_uuid + '_eval_results.json')
    with open(tmp_file_output) as f:
        results = json.load(f)['eval']
    os.remove(tmp_file_output)

    NUM_TASKS = 164

    assert len(answers) == NUM_TASKS * N

    processed_results = []
    for i, item in enumerate(answers):
        task_id = item['task_id']
        assert int(task_id.split('/')[1]) == i % NUM_TASKS
        num_sample = i // NUM_TASKS

        human_eval_result = results[task_id]['base'][num_sample][0]
        human_eval_plus_result = results[task_id]['plus'][num_sample][0]

        result_to_bool = {
            'failed': False,
            'success': True,
        }

        processed_results.append({
            'task_id': task_id,
            'success': {
                'base': result_to_bool[human_eval_result],
                'plus': result_to_bool[human_eval_plus_result],
            }
        })

    process_output_lines = [line for line in process_output.split('\n') if line != '']
    assert process_output_lines[-4] == 'Base'
    assert process_output_lines[-2] == 'Base + Extra'
    human_eval_score = ast.literal_eval(process_output_lines[-3])['pass@1']
    human_eval_plus_score = ast.literal_eval(process_output_lines[-1])['pass@1']

    output = {
        'answers': processed_results,
        'scores': {
            'base': human_eval_score,
            'plus': human_eval_plus_score
        }
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

def evaluate_model(model_type, model_name, model_args, evaluation_id):
    output_folder = os.path.join('reports/human-eval-plus', model_name_to_filename(model_name), evaluation_id)
    os.makedirs(output_folder, exist_ok=True)
    compute_model_answers(model_type=model_type, model_name=model_name, model_args=model_args, output_folder=output_folder)
    compute_scores(output_folder=output_folder)
