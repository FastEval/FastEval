import os
import json

from evaluation.benchmarks.utils import model_name_to_filename

def compute_total_scores(model_name, evaluation_id):
    output_filename = os.path.join('reports', 'total', model_name_to_filename(model_name), evaluation_id, 'scores.json')
    if os.path.exists(output_filename):
        return

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
            return # The model has not been evaluated on all of the benchmarks

        with open(scores_filepath) as f:
            scores_file_content = json.load(f)

        if benchmark_name == 'cot':
            benchmark_score = scores_file_content['total']
        elif benchmark_name == 'human-eval-plus':
            benchmark_score = scores_file_content['scores']['plus']
        elif benchmark_name == 'lm-evaluation-harness':
            benchmark_score = scores_file_content['average']
        elif benchmark_name == 'mt-bench':
            benchmark_score = scores_file_content['average']

        scores['benchmarks'][benchmark_name] = benchmark_score

    # https://github.com/FastEval/FastEval/issues/61#issuecomment-1668562791
    scores['total'] = (
        2.258328740981252 * scores['benchmarks']['mt-bench']
        + 15.877679229809127 * scores['benchmarks']['cot']
        + 15.128786199627087 * scores['benchmarks']['human-eval-plus']
        + 46.41024716075128 * scores['benchmarks']['lm-evaluation-harness']
    )

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(scores, f, indent=4)
