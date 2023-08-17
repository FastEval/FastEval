import os
import json
import re
import ast
import statistics
import textwrap

from evaluation.benchmarks.utils import model_name_to_filename

from evaluation.models.models import create_model, compute_model_replies

JUDGE_MODEL_MAX_NEW_TOKENS = 2048

def generate_assistant_replies(*, model_type, model_name, model_args, evaluation_id, conversations_with_references, data_hash):
    answers_filepath = os.path.join('reports', 'custom', model_name_to_filename(model_name), evaluation_id, data_hash, 'answers.json')
    if os.path.exists(answers_filepath):
        return

    model = create_model(model_type, model_name, model_args)

    conversations_with_ids = [(conversation_id, conversation['conversation'])
        for conversation_id, conversation in conversations_with_references.items()]
    conversations = [conversation for conversation_id, conversation in conversations_with_ids]

    model_replies = compute_model_replies(
        model,
        conversations,
        progress_bar_description=model_name + ' :: Custom :: Computing model replies',
    )

    all_replies = { item[0]: model_replies[i] for i, item in enumerate(conversations_with_ids) }

    os.makedirs(os.path.dirname(answers_filepath), exist_ok=True)
    with open(answers_filepath, 'w') as f:
        json.dump(all_replies, f, indent=4)

def create_judge_conversation(*, conversations_with_references, model_replies, conversation_id):
    conversation_and_reference = conversations_with_references[conversation_id]
    conversation = conversation_and_reference['conversation']
    reference = conversation_and_reference['reference']
    model_reply = model_replies[conversation_id]

    system_message = textwrap.dedent("""\
    Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question.
    Your evaluation should consider correctness and helpfulness.
    """)

    if len(conversation) == 1:
        system_message += "You will be given a user question, the assistant's answer as well as a reference answer.\n"
    elif len(conversation) > 1:
        system_message += textwrap.dedent("""\
        You will be given an past conversation as context, a current user question, the assistant's answer as well as a reference answer.
        You evaluation should focus on the assistant's answer to the current user question.
        Do not evaluate the assistant's answers in the previous conversation context before.
        """)

    system_message += textwrap.dedent("""\
    Begin your evaluation by comparing the assistant's answer with the reference answer.
    Identify and correct any mistakes. Be as objective as possible.
    AFTER providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
    DO NOT output the rating immediately. Only provide the final rating AFTER providing your reasoning step-by-step.
    Also make sure to to use the correct output format for the final rating including the brackets: [[rating]].""")

    judge_prompt = ''

    if len(conversation) > 1:
        conversation_context = conversation[:-1]
        judge_prompt += '<|The Begin Of The Previous Conversation Context|>\n\n'
        for role, content in conversation_context:
            judge_prompt += '### ' + role.capitalize() + ':\n' + content + '\n\n'
        judge_prompt += '<|The End Of The Previous Conversation Context|>\n\n'

    assert conversation[-1][0] == 'user'
    judge_prompt += "### Current User Question:\n" + conversation[-1][1] + '\n\n'

    judge_prompt += "### Assistant's Answer:\n" + model_reply + '\n\n'

    judge_prompt += '### Reference Answer:\n' + reference + '\n\n'

    judge_prompt += ('Please now compare the assistant\'s answer with the reference answer. '
        + 'DO NOT output the rating immediately. Only provide the final rating AFTER providing your reasoning step-by-step.')

    return [
        ('system', system_message),
        ('user', judge_prompt),
    ]

def compute_judge_replies(*, model_name, evaluation_id, conversations_with_references, judge_model_type, judge_model_name, judge_model_args, data_hash):
    judge_replies_filepath = os.path.join('reports', 'custom', model_name_to_filename(model_name), evaluation_id, data_hash, 'judge-replies.json')
    if os.path.exists(judge_replies_filepath):
        return

    with open(os.path.join('reports/custom', model_name_to_filename(model_name), evaluation_id, data_hash, 'answers.json')) as f:
        answers = json.load(f)

    judge_conversations = [{
        'conversation_id': conversation_id,
        'conversation': create_judge_conversation(conversations_with_references=conversations_with_references, model_replies=answers, conversation_id=conversation_id),
    } for conversation_id in conversations_with_references.keys()]

    judge_model = create_model(judge_model_type, judge_model_name, judge_model_args, max_new_tokens=JUDGE_MODEL_MAX_NEW_TOKENS)

    judge_replies = compute_model_replies(judge_model, [{
        'conversation': item['conversation'],
        'temperature': 0,
    } for item in judge_conversations], progress_bar_description=model_name + ' :: Custom :: Judging with ' + judge_model_name)

    judge_replies = { judge_conversations[i]['conversation_id']: judge_reply for i, judge_reply in enumerate(judge_replies) }

    os.makedirs(os.path.dirname(judge_replies_filepath), exist_ok=True)
    with open(judge_replies_filepath, 'w') as f:
        json.dump(judge_replies, f, indent=4)

def compute_model_score(*, model_name, evaluation_id, data_hash):
    scores_filepath = os.path.join('reports', 'custom', model_name_to_filename(model_name), evaluation_id, data_hash, 'scores.json')
    if os.path.exists(scores_filepath):
        return

    judge_replies_filepath = os.path.join('reports', 'custom', model_name_to_filename(model_name), evaluation_id, data_hash, 'judge-replies.json')
    with open(judge_replies_filepath) as f:
        judge_replies = json.load(f)

    ratings = []
    for conversation_id, judge_reply in judge_replies.items():
        match = re.search('\[\[(\d+\.?\d*)\]\]', judge_reply)
        if not match:
            match = re.search('\[(\d+\.?\d*)\]', judge_reply)
        if not match:
            continue

        # TODO: Why is this used (in original fastchat) instead of just parsing string to float?
        rating = ast.literal_eval(match.groups()[0])
        ratings.append(rating)

    if len(ratings) == 0:
        average_rating = None
    else:
        average_rating = statistics.mean(ratings)

    scores = { 'average': average_rating }

    os.makedirs(os.path.dirname(scores_filepath), exist_ok=True)
    with open(scores_filepath, 'w') as f:
        json.dump(scores, f, indent=4)

def evaluate_model_on_single_data_file(model_type, model_name, model_args, evaluation_id, *, data_hash):
    with open(os.path.join('data', 'custom', data_hash + '.json')) as f:
        conversations_with_references = json.load(f)

    judge_model_type = 'openchat-llama2-v1'
    judge_model_name = 'Open-Orca/OpenOrcaxOpenChat-Preview2-13B'
    judge_model_args = {
        'dtype': 'bfloat16',
        'inference_backend': 'vllm',
    }

    generate_assistant_replies(
        model_type=model_type,
        model_name=model_name,
        model_args=model_args,
        evaluation_id=evaluation_id,
        conversations_with_references=conversations_with_references,
        data_hash=data_hash,
    )

    compute_judge_replies(
        model_name=model_name,
        evaluation_id=evaluation_id,
        conversations_with_references=conversations_with_references,
        judge_model_type=judge_model_type,
        judge_model_name=judge_model_name,
        judge_model_args=judge_model_args,
        data_hash=data_hash,
    )

    compute_model_score(
        model_name=model_name,
        evaluation_id=evaluation_id,
        data_hash=data_hash,
    )

def evaluate_model(model_type, model_name, model_args, evaluation_id, *, data_hashes):
    for data_hash in data_hashes:
        evaluate_model_on_single_data_file(model_type, model_name, model_args, evaluation_id, data_hash=data_hash)
