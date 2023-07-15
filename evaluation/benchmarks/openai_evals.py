import os
import json
import typing
import uuid

import evals
import evals.registry
import evals.cli.oaieval

from evaluation.utils import replace_model_name_slashes
from evaluation.models.models import create_model
from evaluation.constants import OPENAI_EVALS_JUDGE_MAX_NEW_TOKENS, OPENAI_EVALS_JUDGE

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

    def make_completion_fn(self, model_type_and_name: str) -> CompletionFn:
        model_type, model_name = model_type_and_name.split(':')
        # TODO: The reviewer has a larger number of tokens than normally.
        # However, we currently don't check whether it's the reviewer model but whether it's the same model as the reviewer.
        # Meaning that if we now evaluate the same model as OPENAI_EVALS_JUDGE, then we will also increase the number of tokens
        # for that model which is unfair. Fix that.
        if model_type == OPENAI_EVALS_JUDGE[0] and model_name == OPENAI_EVALS_JUDGE[1]:
            return CompletionFn(create_model(model_type, model_name, max_new_tokens=OPENAI_EVALS_JUDGE_MAX_NEW_TOKENS))
        return CompletionFn(create_model(model_type, model_name))

    # Prevent errors about OpenAI API key missing even if we don't use OpenAI models
    api_model_ids = []

def run_single_eval(registry: Registry, model_type: str, model_name: str, eval):
    models = model_type + ':' + model_name
    if eval.cls == 'evals.elsuite.modelgraded.classify:ModelBasedClassify':
        models += ',' + OPENAI_EVALS_JUDGE[0] + ':' + OPENAI_EVALS_JUDGE[1]

    tmpfile = os.path.join('.tmp', 'openai-evals', str(uuid.uuid4()) + '.json')
    os.makedirs(os.path.dirname(tmpfile), exist_ok=True)

    evals.cli.oaieval.run(evals.cli.oaieval.get_parser().parse_args([
        models,
        eval.key,
        '--record_path', tmpfile,
    ]), registry)

    outfile = os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name), eval.key + '.json')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    os.link(tmpfile, outfile)

def run_multiple_evals(registry: Registry, model_type: str, model_name: str, evals):
    non_working_evals = [
        'best.dev.v0', # Compares multiple models
        'positive-binary-operations.test.v1', # KeyError: 'sample'
        'spider-sql.dev.v0', # TypeError: ModelBasedClassify.__init__() missing 1 required positional argument: 'modelgraded_spec'
        'svg_understanding.v0', # CUDA out of memory
        'stock-options-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_options_iron_butteryfly_spread.jsonl
        'stock-option-terms-inverse-iron-butteryfly-spread.dev.v0', # RuntimeError: Failed to open: stock_options/stock_option_terms_inverse_iron_butteryfly_spread.jsonl
        'joke-fruits-v2.dev.v0', # Buggy in openai/evals itself due to removed format_type feature that is still used by this eval
    ]

    evals_where_all_models_get_zero_score = [
        'actors-sequence.dev.match-v1',
        'complex-replace-characters.dev.v0',
        'finance.dev.v0',
        'knot-theory-code-conversion.dev.v0',
        'naughty_strings.test.v1',
        'reverse-string.s1.simple-v0',
        'decrypt-caesar-cipher.dev.v0',
        'rot13.s1.simple-v0',
        'stock-options-bull-call-spread.dev.v0',
        'stock-options-inverse-iron-butterfly-spread.dev.v0',
        'stock-options-inverse-iron-condor-spread.dev.v0',
        'stock-options-iron-condor-spread.dev.v0',
        'unified-patch.dev.v0',
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

        # required context size too large
        'illinois-law.v0',
        'qa.dev.v0',

        # MMLU is already separated as part of CoT
        'mmlu-abstract-algebra.val.ab-v1',
        'mmlu-anatomy.val.ab-v1',
        'mmlu-astronomy.val.ab-v1',
        'mmlu-business-ethics.val.ab-v1',
        'mmlu-clinical-knowledge.val.ab-v1',
        'mmlu-college-biology.val.ab-v1',
        'mmlu-college-chemistry.val.ab-v1',
        'mmlu-college-computer-science.val.ab-v1',
        'mmlu-college-mathematics.val.ab-v1',
        'mmlu-college-medicine.val.ab-v1',
        'mmlu-college-physics.val.ab-v1',
        'mmlu-computer-security.val.ab-v1',
        'mmlu-conceptual-physics.val.ab-v1',
        'mmlu-econometrics.val.ab-v1',
        'mmlu-electrical-engineering.val.ab-v1',
        'mmlu-elementary-mathematics.val.ab-v1',
        'mmlu-formal-logic.val.ab-v1',
        'mmlu-global-facts.val.ab-v1',
        'mmlu-high-school-biology.val.ab-v1',
        'mmlu-high-school-chemistry.val.ab-v1',
        'mmlu-high-school-computer-science.val.ab-v1',
        'mmlu-high-school-european-history.val.ab-v1',
        'mmlu-high-school-geography.val.ab-v1',
        'mmlu-high-school-government-and-politics.val.ab-v1',
        'mmlu-high-school-macroeconomics.val.ab-v1',
        'mmlu-high-school-mathematics.val.ab-v1',
        'mmlu-high-school-microeconomics.val.ab-v1',
        'mmlu-high-school-physics.val.ab-v1',
        'mmlu-high-school-psychology.val.ab-v1',
        'mmlu-high-school-statistics.val.ab-v1',
        'mmlu-high-school-us-history.val.ab-v1',
        'mmlu-high-school-world-history.val.ab-v1',
        'mmlu-human-aging.val.ab-v1',
        'mmlu-human-sexuality.val.ab-v1',
        'mmlu-international-law.val.ab-v1',
        'mmlu-jurisprudence.val.ab-v1',
        'mmlu-logical-fallacies.val.ab-v1',
        'mmlu-machine-learning.val.ab-v1',
        'mmlu-management.val.ab-v1',
        'mmlu-marketing.val.ab-v1',
        'mmlu-medical-genetics.val.ab-v1',
        'mmlu-miscellaneous.val.ab-v1',
        'mmlu-moral-disputes.val.ab-v1',
        'mmlu-moral-scenarios.val.ab-v1',
        'mmlu-nutrition.val.ab-v1',
        'mmlu-philosophy.val.ab-v1',
        'mmlu-prehistory.val.ab-v1',
        'mmlu-professional-accounting.val.ab-v1',
        'mmlu-professional-law.val.ab-v1',
        'mmlu-professional-medicine.val.ab-v1',
        'mmlu-professional-psychology.val.ab-v1',
        'mmlu-public-relations.val.ab-v1',
        'mmlu-security-studies.val.ab-v1',
        'mmlu-sociology.val.ab-v1',
        'mmlu-us-foreign-policy.val.ab-v1',
        'mmlu-virology.val.ab-v1',
        'mmlu-world-religions.val.ab-v1',
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
    reports_index_path = os.path.join(model_reports_path, '__index__.json')
    if os.path.exists(reports_index_path):
        return

    reports_metadata = {}
    for report_filename in os.listdir(model_reports_path):
        if report_filename == '__index__.json':
            continue
        with open(os.path.join(model_reports_path, report_filename), 'r') as report_file:
            report_data = report_file.read().split('\n')
        report_data = [json.loads(line) for line in report_data if line != '']
        reports_metadata[report_filename] = {
            'spec': [item for item in report_data if 'spec' in item][0]['spec'],
            'final_report': [item for item in report_data if 'final_report' in item][0]['final_report'],
        }

    with open(reports_index_path, 'w') as reports_index_file:
        json.dump(reports_metadata, reports_index_file, indent=4)

def evaluate_model(model_type: str, model_name: str):
    os.environ['EVALS_THREAD_TIMEOUT'] = '999999'

    model = create_model(model_type, model_name)
    os.environ['EVALS_THREADS'] = str(min(model.num_threads, 20))

    registry = Registry()
    run_multiple_evals(registry, model_type, model_name, [eval for eval in registry.get_evals(['*'])])
    create_reports_index_file(model_name)
