import os
import json

from evaluation.benchmarks.utils import model_name_to_filename
from evaluation.constants import WEIGHTS

def compute_total_scores(model_name, evaluation_id):
    benchmarks = ['cot', 'human-eval-plus', 'lm-evaluation-harness', 'mt-bench']
    scores = { 'benchmarks': {} }
    for benchmark_name in benchmarks:
        benchmark_folder = os.path.join('reports', benchmark_name, model_name_to_filename(model_name), evaluation_id)

        if benchmark_name == 'lm-evaluation-harness':
            scores_filename = 'total.json'
        else:
            scores_filename = 'scores.json'
        scores_filepath = os.path.join(benchmark_folder, scores_filename)

        if not os.path.exists(scores_filepath):
            continue # The model has not been evaluated on all of the benchmarks

        with open(scores_filepath) as f:
            scores_file_content = json.load(f)

        if benchmark_name == 'cot':
            if not 'total' in scores_file_content:
                continue
            benchmark_score = scores_file_content['total']
        elif benchmark_name == 'human-eval-plus':
            benchmark_score = scores_file_content['scores']['plus']
        elif benchmark_name == 'lm-evaluation-harness':
            benchmark_score = scores_file_content['average']
        elif benchmark_name == 'mt-bench':
            benchmark_score = scores_file_content['average']

        scores['benchmarks'][benchmark_name] = benchmark_score

    if 'cot' in scores['benchmarks'] and 'human-eval-plus' in scores['benchmarks'] and 'mt-bench' in scores['benchmarks']:
        scores['total'] = 0
        max_value = 0
        for benchmark_name in scores['benchmarks'].keys():
            weight, benchmark_max_value = WEIGHTS[benchmark_name]
            scores['total'] += weight * scores['benchmarks'][benchmark_name]
            max_value += weight * benchmark_max_value
        scores['total'] /= max_value
        scores['total'] *= 100

    output_filename = os.path.join('reports', 'total', model_name_to_filename(model_name), evaluation_id, 'scores.json')
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(scores, f, indent=4)

def get_total_scores(model_name, evaluation_id):
    scores_filepath = os.path.join('reports', 'total', model_name_to_filename(model_name), evaluation_id, 'scores.json')
    with open(scores_filepath) as f:
        return json.load(f)
