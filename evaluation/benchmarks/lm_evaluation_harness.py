import os
import json

import lm_eval.evaluator

from ..utils import replace_model_name_slashes
from evaluation.models.models import get_dtype, create_model

def evaluate_model(model_type, model_name, model_args):
    if model_type == 'openai':
        return

    output_path = os.path.join('./reports/lm-evaluation-harness', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(output_path):
        return

    tokenizer_path = create_model(model_type, model_name, model_args).tokenizer_path

    tasks = ['openbookqa', 'arc_easy', 'winogrande', 'hellaswag', 'arc_challenge', 'piqa', 'boolq']

    lm_eval_model_args = ','.join([k + '=' + str(v) for k, v in ({
        'pretrained': model_name,
        'dtype': str(get_dtype(model_name)).replace('torch.', ''),
        'trust_remote_code': True,
        'use_accelerate': True,
        'tokenizer': tokenizer_path,
    }).items()])

    print(model_name + ' :: LM-Eval :: Evaluating')

    results = lm_eval.evaluator.simple_evaluate('hf-causal-experimental', tasks=tasks, model_args=lm_eval_model_args)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
