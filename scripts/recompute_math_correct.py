#!/usr/bin/env python3

import os
import json

from evaluation.benchmarks.cot_math_equivalence import is_math_correct, extract_model_answer

#print(extract_model_answer('The answer is that there is no value of $B$ that satisfies the given condition.'))
#import sys
#sys.exit()

reports_path = 'reports/cot'
models = os.listdir(reports_path)
evaluations_paths = []
for model in models:
    model_path = os.path.join(reports_path, model)
    for evaluation in os.listdir(model_path):
        evaluation_path = os.path.join(model_path, evaluation)
        evaluations_paths.append((model, evaluation_path))

total_score = 0
for model, evaluation_path in evaluations_paths:
    #if model != 'WizardLM--WizardLM-70B-V1.0':
    #    continue

    scores_file = os.path.join(evaluation_path, 'scores.json')
    if os.path.exists(scores_file):
        os.remove(scores_file)

    math_file = os.path.join(evaluation_path, 'tasks/math.json')

    with open(math_file) as f:
        math_data = json.load(f)

    model_outputs = math_data['model_outputs']

    num_correct = 0
    for i, model_output in enumerate(model_outputs):
        is_correct = is_math_correct(correct_answer=model_output['correct_answer'], model_answer=model_output['model_answer'].split('\n')[-1])
        model_output['correct'] = is_correct
        if is_correct:
            num_correct += 1

    average_score = num_correct / len(model_outputs)
    math_data['score'] = average_score
    total_score += average_score

    with open(math_file, 'w') as f:
        json.dump(math_data, f, indent=4)

    print(model, average_score)

print(total_score)
