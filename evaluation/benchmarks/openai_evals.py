import os
import json
import typing

import evals
import evals.registry
import evals.cli.oaieval

from evaluation.utils import replace_model_name_slashes, create_model

def convert_conversation(prompt: typing.Union[str, list[dict[str, str]]]):
    if isinstance(prompt, str):
        return [('user', prompt)]

    conversation = []
    for item in prompt:
        role = item['role']
        content = item['content']
        if role == 'system' and 'name' not in item:
            conversation.append(('system', content))
        elif role == 'assistant' or (role == 'system' and item['name'] == 'example_assistant'):
            conversation.append(('assistant', content))
        elif (role == 'system' and item['name'] == 'example_user') or role == 'user':
            conversation.append(('user', content))
        else:
            raise

    return conversation

class CompletionResult(evals.api.CompletionResult):
    """
    I don't know why this thing is needed just to wrap a single reply string.
    But that's what OpenAI evals requires, so we need it.
    """

    def __init__(self, reply: str):
        self.reply = reply

    def get_completions(self) -> list[str]:
        return [self.reply.strip()]

class CompletionFn(evals.api.CompletionFn):
    """
    See https://github.com/openai/evals/blob/main/docs/completion-fns.md
    """

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        prompt: typing.Union[str, list[dict[str, str]]],
        **kwargs,
    ) -> CompletionResult:
        return CompletionResult(self.model.reply(convert_conversation(prompt)))

class Registry(evals.registry.Registry):
    def __init__(self):
        super().__init__()
        self.previous_model_name = None
        self.previous_model = None

    def make_completion_fn(self, model_type_and_name: str) -> CompletionFn:
        model_type, model_name = model_type_and_name.split(':')
        if model_type == 'openai':
            return CompletionFn(create_model(model_type, model_name))
        if self.previous_model_name != model_name:
            self.previous_model_name = model_name
            self.previous_model = create_model(model_type, model_name)
        return CompletionFn(self.previous_model)

    # Prevent errors about OpenAI API key missing even if we don't use OpenAI models
    api_model_ids = []

def run_single_eval(registry: Registry, model_type: str, model_name: str, eval):
    models = model_type + ':' + model_name
    if eval.cls == 'evals.elsuite.modelgraded.classify:ModelBasedClassify':
        models += ',openai:gpt-3.5-turbo'
    evals.cli.oaieval.run(evals.cli.oaieval.get_parser().parse_args([
        models,
        eval.key,
        '--record_path', os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name), eval.key + '.json'),
    ]), registry)

def run_multiple_evals(registry: Registry, model_type: str, model_name: str, evals):
    non_working_evals = [
        'best.dev.v0', # Compares multiple models
        'positive-binary-operations.test.v1', # KeyError: 'sample'
        'spider-sql.dev.v0', # TypeError: ModelBasedClassify.__init__() missing 1 required positional argument: 'modelgraded_spec'
        'svg_understanding.v0', # CUDA out of memory
        'stock-options-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_options_iron_butteryfly_spread.jsonl
        'stock-option-terms-inverse-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_option_terms_inverse_iron_butteryfly_spread.jsonl
        'joke-fruits-v2.dev.v0', # Buggy in openai/evals itself due to removed format_type feature that is still used by this eval
        'illinois-law.v0', # RuntimeError: The size of tensor a (2048) must match the size of tensor b (2287) at non-singleton dimension 3
    ]

    evals_where_all_models_get_zero_score = [
        'actors-sequence.dev.match-v1',
        'complex-replace-characters.dev.v0',
        'finance.dev.v0',
        'knot-theory-code-conversion.dev.v0',
        'naughty_strings.test.v1',
        'reverse-string.s1.simple-v0',
    ]

    other_excluded_evals = [
        # Requires sacrebleu>=2.3.1 which conflicts with lm-evaluation-harness dependencies
        'manga-translation-bubble.dev.v0',
        'manga-translation-page.dev.v0',
        'manga-translation-panel.dev.v0',

        # Doesn't give a score
        'categorize-with-distractors.dev.v0',
        'coqa-fact-expl.dev.v0',
        'coqa-fact.dev.v0',
        'logic-fact.dev.v0',
        'logic-liar-paradox.dev.v0',
        'loss-logic-fact.dev.v0',
        'naughty_strings_graded_meta.test.v1',
        'rap-animals-vs-fruits.dev.v0',

        # calculate scores incorrectly
        'joke-animals-vs-fruits.dev.v0',
        'joke-fruits-ans-meta.dev.v0',
        'joke-fruits-expl-meta.dev.v0',
        'joke-fruits-likert.dev.v0',
        'joke-fruits-meta.dev.v0',
        'joke-fruits.dev.v0',
    ]

    ignored_evals = non_working_evals + evals_where_all_models_get_zero_score + other_excluded_evals

    for eval in evals:
        if os.path.exists(os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name), eval.key + '.json')):
            continue
        if eval.key in ignored_evals:
            continue
        print('Now evaluating', eval.key)
        run_single_eval(registry, model_type, model_name, eval)

def create_reports_index_file(model_name: str):
    """
    Create a single file in reports/openai-evals/<model_name>/__index__.json that contains information about all the evals
    that this model was evaluated on. This index file is used for showing information on the website.
    """

    model_reports_path = os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name))

    reports_metadata = {}
    for report_filename in os.listdir(model_reports_path):
        if report_filename == '__index__.json':
            continue
        with open(os.path.join(model_reports_path, report_filename), 'r') as report_file:
            report_metadata = report_file.read().split('\n')[:2]
        reports_metadata[report_filename] = {
            'spec': json.loads(report_metadata[0])['spec'],
            'final_report': json.loads(report_metadata[1])['final_report'],
        }

    with open(os.path.join(model_reports_path, '__index__.json'), 'w') as reports_index_file:
        json.dump(reports_metadata, reports_index_file, indent=4)

def evaluate_model(model_type: str, model_name: str):
    os.environ['EVALS_THREADS'] = '1'
    os.environ['EVALS_THREAD_TIMEOUT'] = '999999'

    registry = Registry()
    run_multiple_evals(registry, model_type, model_name, [eval for eval in registry.get_evals(['*'])])
    create_reports_index_file(model_name)

def evaluate_models(models: list[dict[str, str]]):
    for model_type, model_name in models:
        evaluate_model(model_type, model_name)
