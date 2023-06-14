import re
import os
import json

import datasets
import tqdm

from evaluation.utils import create_model, replace_model_name_slashes

def reply(model, answer_format, question):
    return model.reply([
        ('user',
            'Please answer the following question step-by-step. '
            'Do not output the answer immediately. '
            'Instead first explain your reasoning step-by-step. '
            'Only afterwards output the answer. '
            'The final line should contain the answer ' + answer_format + ' without anything else.'
            '\n\n'
            + question),
    ])

def evaluate_model_on_dataset(*, name, model, data, question_column, answer_column, answer_format, is_correct):
    num_correct = 0
    num_total = 0
    print('Evaluating model on ', name)
    for item in tqdm.tqdm(data):
        question = item[question_column]
        correct_answer = item[answer_column]
        model_answer = reply(model, answer_format, question)
        model_answer_is_correct = is_correct(model_answer=model_answer.split('\n')[-1], correct_answer=correct_answer)
        if model_answer_is_correct:
            num_correct += 1
        num_total += 1
        if num_total >= 2:
            break # TODO: Only for testing. Remove.
    return num_correct / num_total

def evaluate_model_on_gsm8k(model):
    def is_correct(model_answer, correct_answer):
        model_answer_matches = re.findall(r'(\d+(,\d+)*(\.\d*)?)', model_answer)
        if len(model_answer_matches) == 0:
            return False
        model_answer_processed = float(model_answer_matches[0][0].replace(',', ''))
        correct_answer_processed = float(correct_answer.split('\n')[-1].split('####')[1].strip())
        return abs(model_answer_processed - correct_answer_processed) < 1e-8

    return evaluate_model_on_dataset(
        name='gsm8k',
        model=model,
        data=datasets.load_dataset('gsm8k', 'main')['test'],
        question_column='question',
        answer_column='answer',
        answer_format='as a single number',
        is_correct=is_correct,
    )

def evaluate_model_on_bbh(model):
    def is_correct(model_answer, correct_answer):
        model_answer_matches = re.findall(r'\([ABCDEFGHIJKLMNOPQRSTUVWXYZ]\)', model_answer)
        if len(model_answer_matches) == 0:
            return False
        return model_answer_matches[-1] == correct_answer

    tasks = [
        'date_understanding',
        'disambiguation_qa',
        'geometric_shapes',
        'hyperbaton',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'logical_deduction_three_objects',
        'movie_recommendation',
        'penguins_in_a_table',
        'reasoning_about_colored_objects',
        'ruin_names',
        'salient_translation_error_detection',
        'snarks',
        'temporal_sequences',
        'tracking_shuffled_objects_five_objects',
        'tracking_shuffled_objects_seven_objects',
        'tracking_shuffled_objects_three_objects'
    ]

    accuracies = {
        task: evaluate_model_on_dataset(
            name='BBH ' + task,
            model=model,
            data=datasets.load_dataset('lukaemon/bbh', task)['test'],
            question_column='input',
            answer_column='target',
            answer_format='as a single letter with parenthesis',
            is_correct=is_correct,
        ) for task in tasks
    }

    return {
        'tasks': accuracies,
        'average': sum(accuracies.values()) / len(accuracies.values())
    }


def evaluate_model(model_type, model_name):
    output_path = os.path.join('./reports/cot', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(output_path):
        return

    model = create_model(model_type, model_name)

    output = {
        'gsm8k': evaluate_model_on_gsm8k(model),
        'bbh': evaluate_model_on_bbh(model),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

def evaluate_models(models):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
