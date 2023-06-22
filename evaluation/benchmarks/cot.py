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

def evaluate_model_on_dataset(*, name, model, data, question_column, answer_column, answer_format, is_correct, output_path):
    output_file_path = os.path.join(output_path, name + '.json')
    if os.path.exists(output_file_path):
        with open(output_file_path) as f:
            return json.load(f)['score']

    num_correct = 0
    num_total = 0
    model_outputs = []
    print('Evaluating model on ', name)
    for item in tqdm.tqdm(data):
        question = item[question_column]
        correct_answer = item[answer_column]
        model_answer = reply(model, answer_format, question)
        model_answer_is_correct = is_correct(model_answer=model_answer.split('\n')[-1], correct_answer=correct_answer)
        model_outputs.append({ 'id': num_total, 'question': question, 'correct_answer': correct_answer,
            'model_answer': model_answer, 'correct': model_answer_is_correct })
        if model_answer_is_correct:
            num_correct += 1
        num_total += 1
        if num_total >= 2:
            break # TODO: Only for testing. Remove.

    score = num_correct / num_total

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    print(os.path.basename(output_file_path))
    with open(output_file_path, 'w') as f:
        json.dump({
            'score': score,
            'model_outputs': model_outputs,
        }, f, indent=4)

    return score

def evaluate_model_on_gsm8k(model, output_path):
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
        output_path=output_path,
    )

def evaluate_model_on_bbh(model, output_path):
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
            name='bbh/' + task,
            model=model,
            data=datasets.load_dataset('lukaemon/bbh', task)['test'],
            question_column='input',
            answer_column='target',
            answer_format='as a single letter with parenthesis',
            is_correct=is_correct,
            output_path=output_path,
        ) for task in tasks
    }

    return {
        'tasks': accuracies,
        'average': sum(accuracies.values()) / len(accuracies.values())
    }

def evaluate_model(model_type, model_name):
    output_folder = os.path.join('reports', 'cot', replace_model_name_slashes(model_name))
    final_scores_file = os.path.join(output_folder, 'scores.json')
    if os.path.exists(final_scores_file):
        return

    model = create_model(model_type, model_name)

    tasks_path = os.path.join(output_folder, 'tasks')

    gsm8k_score = evaluate_model_on_gsm8k(model, tasks_path)
    bbh_scores = evaluate_model_on_bbh(model, tasks_path)
    scores = [gsm8k_score, bbh_scores['average']]

    output = {
        'gsm8k': gsm8k_score,
        'bbh': bbh_scores,
        'average': sum(scores) / len(scores),
    }

    os.makedirs(os.path.dirname(final_scores_file), exist_ok=True)
    with open(final_scores_file, 'w') as f:
        json.dump(output, f, indent=4)

def evaluate_models(models):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
