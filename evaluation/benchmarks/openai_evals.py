import os
import json
import typing
import uuid

import evals
import evals.registry
import evals.cli.oaieval

from evaluation.utils import replace_model_name_slashes, process_with_thread_pool
from evaluation.models.models import create_model
from evaluation.constants import OPENAI_EVALS_JUDGE_MAX_NEW_TOKENS, OPENAI_EVALS_JUDGE

ignored_evals = [
    # Compares multiple models
    'best.dev.v0',

    # KeyError: 'sample'
    'positive-binary-operations.test.v1',

    # RuntimeError: Failed to open: stock_options/stock_options_iron_butteryfly_spread.jsonl
    'stock-options-iron-butteryfly-spread.dev.v0',

    # RuntimeError: Failed to open: stock_options/stock_option_terms_inverse_iron_butteryfly_spread.jsonl
    'stock-option-terms-inverse-iron-butteryfly-spread.dev.v0',

    # Buggy in openai/evals itself due to removed format_type feature that is still used by this eval
    'joke-fruits-v2.dev.v0',

    # Just for testing OpenAI Evals itself
    'test-match.s1.simple-v0',
    'test-includes-ignore-case.s1.simple-v0.json',
    'test-includes.s1.simple-v0.json',
    'test-fuzzy-match.s1.simple-v0.json',

    # Measures something we don't care about
    'diversity.dev.v0',

    # All models get a score of 0
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

    # Requires sacrebleu>=2.3.1 which conflicts with lm-evaluation-harness dependencies
    'manga-translation-bubble.dev.v0',
    'manga-translation-page.dev.v0',
    'manga-translation-panel.dev.v0',
    'chinese_hard_translations.dev.v0',

    # Uses ModelBasedClassify with answers from A-D.
    # Doesn't directly give a score.
    # Maybe I could eventually calculate one, but for now exclude it.
    'Unfamiliar-Chinese-Character.dev.v0',
    'allergen-information.dev.v0',
    'chinese_idioms.dev.v0',
    'consensus_summary.dev.v0',
    'euler_problems.dev.v0',
    'event-categories.dev.v0',
    'logic-container.dev.v0',
    'logic-riddles.dev.v0',
    'population_span_extraction.dev.v0',
    'reasoning_with_contradictory_statements.dev.v0',
    'security_guide.dev.v0',
    'categorize-with-distractors.dev.v0',
    'coqa-fact-expl.dev.v0',
    'coqa-fact.dev.v0',
    'logic-fact.dev.v0',
    'logic-liar-paradox.dev.v0',
    'loss-logic-fact.dev.v0',
    'soc_codes.dev.v0',
    'superficial-patterns.dev.v0',
    'ukraine_electronic_petitions.val.v0',
    'unwanted-rhyming.dev.v0',
    'naughty_strings_graded_meta.test.v1',

    # Meta evals (as far as I understand, they are for evaluating the quality of an eval itself)
    'coq-editing-meta.dev.v0',
    'arithmetic-expression-meta.dev.v0',
    'iambic-pentameter.dev.v0',
    'linear-regression-meta.dev.v0',
    'non-compound-names-meta.dev.v0',
    'translation-meta.dev.v0',

    # Calculates scores incorrectly
    'joke-animals-vs-fruits.dev.v0',
    'joke-fruits-ans-meta.dev.v0',
    'joke-fruits-expl-meta.dev.v0',
    'joke-fruits-likert.dev.v0',
    'joke-fruits-meta.dev.v0',
    'joke-fruits.dev.v0',

    # Required too large context size
    'illinois-law.v0',
    'qa.dev.v0',
    'svg_understanding.v0',

    # MMLU is already separately evaluated as part of CoT
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

    # I'm too lazy to implement more complexity for this single eval that needs its own eval class.
    # If I want to evaluate on this, I'm going to do it separately outside OpenAI evals.
    'lambada.oaitest.v1',
]

additionally_ignored_evals = None
def get_additionally_ignored_evals():
    # Filter for evals that have descriptions.
    # Going to use that as an indicator for higher-quality evals
    # and to remove evals that might just be used for testing OpenAI evals itself
    # and stuff like that.

    global additionally_ignored_evals
    if additionally_ignored_evals is not None:
        return additionally_ignored_evals

    with open('data/openai-evals/evals-descriptions.json') as f:
        evals_descriptions = json.load(f)
        additionally_ignored_evals = [k for k, v in evals_descriptions.items() if v is None]

    return additionally_ignored_evals

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
        return [self.reply]

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

    try:
        evals.cli.oaieval.run(evals.cli.oaieval.get_parser().parse_args([
            models,
            eval.key,
            '--record_path', tmpfile,
        ]), registry)
    except RuntimeError as exception:
        raise Exception('OpenAI Evals: Error evaluating ' + eval.key + ': ' + str(exception))

    outfile = os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name), eval.key + '.json')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    os.link(tmpfile, outfile)

def run_all_evals(model_type: str, model_name: str):
    os.environ['EVALS_SHOW_EVAL_PROGRESS'] = ''

    registry = Registry()

    additionally_ignored_evals = get_additionally_ignored_evals()

    evals = [eval for eval in registry.get_evals(['*']) if eval.key not in ignored_evals and eval.key not in additionally_ignored_evals
        and not os.path.exists(os.path.join('reports', 'openai-evals', replace_model_name_slashes(model_name), eval.key + '.json'))]

    if len(evals) == 0:
        return

    unique_evals = []
    unique_evals_keys = []
    for eval in evals:
        if eval.key in unique_evals_keys:
            continue
        unique_evals.append(eval)
        unique_evals_keys.append(eval.key)

    assert_no_duplicate_evals(evals)

    model_num_threads = create_model(model_type, model_name).num_threads
    num_threads_per_eval = min(model_num_threads, 20)
    os.environ['EVALS_THREADS'] = str(num_threads_per_eval)

    def evaluate(eval):
        run_single_eval(registry, model_type, model_name, eval)

    process_with_thread_pool(
        num_threads=(model_num_threads // num_threads_per_eval) * 4,
        items=unique_evals,
        process_function=evaluate,
        desc=model_name + ' :: OpenAI Evals :: Evaluating',
    )

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
    if model_type != 'debug':
        return

    run_all_evals(model_type, model_name)
    create_reports_index_file(model_name)
