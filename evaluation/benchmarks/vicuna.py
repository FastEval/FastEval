import os
import json
import random
import numpy as np

from evaluation.utils import replace_model_name_slashes, undo_replace_model_name_slashes, create_model, compute_model_replies
from evaluation.constants import JUDGE_MAX_NEW_TOKENS, VICUNA_JUDGE

def generate_assistant_replies(model_type, model_name):
    answers_filepath = os.path.join('reports', 'vicuna', 'answers', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(answers_filepath):
        return

    model = create_model(model_type, model_name)

    with open('data/vicuna/questions.json') as f:
        questions = json.load(f)

    questions_items = questions.items()
    replies = compute_model_replies(model, [[('user', question)] for _question_id, question in questions_items])
    answers = { question_id: replies[i] for i, (question_id, _question) in enumerate(questions_items) }

    os.makedirs(os.path.dirname(answers_filepath), exist_ok=True)
    with open(answers_filepath, 'w') as f:
        json.dump(answers, f, indent=4)

def create_reviewer_prompt(question, answer1, answer2):
    system_message = 'You are a helpful and precise assistant for checking the quality of the answer.'

    # https://medium.com/@geronimo7/open-source-chatbots-in-the-wild-9a44d7a41a48
    # https://arxiv.org/abs/2305.17926
    # https://twitter.com/natolambert/status/1665757105788432384
    prompter_message = ('[Question]\n'
        + question + '\n'
        + '\n'
        + "[The Start of Assistant 1's Answer]\n"
        + answer1 + '\n'
        + "[The End of Assistant 1's Answer]\n"
        + '\n'
        + "[The Start of Assistant 2's Answer]\n"
        + answer2 + '\n'
        + "[The End of Assistant 2's Answer]\n"
        + '\n'
        + '[System]\n'
        + 'We would like to request your feedback on the performance of the two AI assistants (Assistant 1 and Assistant 2) in response to the user question displayed above. '
        + 'First, rate important aspects like relevance, helpfulness, conciseness and especially accuracy of their responses. '
        + 'Second, output who provided the best answer. '
        + 'Do not output the final result immediately. Output the reasoning first in order to think about it step by step. '
        + "If both answers are equally good and it's hard to decide on a winner then please call it a tie. "
        + "The final line after the step-by-step reasoning should look like this: 'Winner: Assistant 1' or 'Winner: Assistant 2' or 'Tie'. \n"
        + '\n')

    return system_message, prompter_message

def find_winner(line):
    possible_matches = [
        'winner of this round is assistant',
        'winner: assistant',
        'winner is assistant',
        'winner to be assistant',
        'winner is: assistant',
        'winner is "assistant',
        'winner for this question is assistant',
        'winner of this round is: assistant',
        'winner of this competition is assistant',
    ]

    winner_model = None
    for possible_match in possible_matches:
        if (possible_match + ' 1') in line:
            if winner_model is not None:
                return None
            winner_model = '1'
        elif (possible_match + ' 2') in line:
            if winner_model is not None:
                return None
            winner_model = '2'

    ties = [
        'winner: tie',
        'is a tie',
        'it a tie',
        "it's a tie",
        'winner: none',
        'winner is tie',
        ' tie.',
        "'tie'",
        "'tie.'",
    ]

    for tie in ties:
        if tie in line:
            if winner_model is not None:
                return None
            winner_model = 'tie'
            break
    if line in ['Tie.']:
        if winner_model is not None:
            return None
        winner_model = 'tie'

    return winner_model

def mle(pmat, max_iter=1000):
    # https://datascience.stackexchange.com/a/19079

    n = pmat.shape[0]
    wins = np.sum(pmat, axis=0)
    params = np.ones(n, dtype=float)
    for i in range(max_iter):
        tiled = np.tile(params, (n, 1))
        combined = 1.0 / (tiled + tiled.T)
        np.fill_diagonal(combined, 0)
        nxt = wins / np.sum(combined, axis=0)
        nxt = nxt / np.mean(nxt)
        if np.linalg.norm(nxt - params, ord=np.inf) < 1e-6:
            return nxt
        params = nxt
    raise RuntimeError('did not converge')

def compute_ranks(model_names, matches):
    num_wins = np.zeros((len(model_names), len(model_names)))
    model_name_to_index = { model_name: index for index, model_name in enumerate(model_names) }

    for model1, model2, winner_model in matches:
        model1_index = model_name_to_index[model1]
        model2_index = model_name_to_index[model2]
        if winner_model == '1':
            num_wins[model1_index, model2_index] += 1
        elif winner_model == '2':
            num_wins[model2_index, model1_index] += 1
        elif winner_model == 'tie':
            num_wins[model1_index, model2_index] += 0.5
            num_wins[model2_index, model1_index] += 0.5

    ranking = -mle(num_wins)
    min_rank = ranking.min()
    max_rank = ranking.max()
    ranking = { model_names[index]: ((rank - min_rank) / (max_rank - min_rank)) for index, rank in enumerate(ranking) }

    return ranking

def generate_reviews():
    with open('data/vicuna/questions.json') as f:
        questions = json.load(f)

    answers = {}
    answers_base_directory = os.path.join('reports', 'vicuna', 'answers')
    for model_filename in os.listdir(answers_base_directory):
        model_name = undo_replace_model_name_slashes(model_filename.replace('.json', ''))
        with open(os.path.join(answers_base_directory, model_filename)) as f:
            answers[model_name] = json.load(f)
    models = list(answers.keys())

    reviewer = create_model(*VICUNA_JUDGE, max_new_tokens=JUDGE_MAX_NEW_TOKENS)

    reviews_filepath = os.path.join('reports', 'vicuna', 'reviews.json')
    if os.path.exists(reviews_filepath):
        with open(reviews_filepath) as f:
            reviews = json.load(f)
    else:
        os.makedirs(os.path.dirname(reviews_filepath), exist_ok=True)
        reviews = []

    review_count = { (model_name1, model_name2): 0  for model_name1 in models for model_name2 in models if model_name1 != model_name2 }
    for review in reviews:
        review_count[(review['model1'], review['model2'])] += 1

    while True:
        question_id = random.choice(list(questions.keys()))
        model_name1, model_name2 = min(review_count, key=review_count.get)
        if review_count[(model_name1, model_name2)] >= 50:
            break
        reviews.append({ 'question_id': question_id, 'model1': model_name1, 'model2': model_name2 })
        review_count[(model_name1, model_name2)] += 1

    conversation_review_indices = {}
    conversations = []
    for i, review in enumerate(reviews):
        if 'review' in review:
            continue
        question_id = review['question_id']
        system_message, prompter_message = create_reviewer_prompt(
            questions[question_id],
            answers[review['model1']][question_id],
            answers[review['model2']][question_id]
        )
        conversation_review_indices[i] = len(conversations)
        conversations.append([
            ('system', system_message),
            ('user', prompter_message),
        ])

    ranks_filepath = os.path.join('reports', 'vicuna', 'ranks.json')
    if len(conversations) == 0 and os.path.exists(ranks_filepath):
        return False

    replies = compute_model_replies(reviewer, conversations)

    for i, review in enumerate(reviews):
        if 'review' in review:
            continue
        model_name1 = review['model1']
        model_name2 = review['model2']

        reply = replies[conversation_review_indices[i]]
        review['review'] = reply

        winner_model = find_winner(reply.split('\n')[-1].lower())
        review['winner_model'] = winner_model
        if winner_model is None:
            continue

    reviews = [review for review in reviews if review['winner_model'] is not None]

    ranks = compute_ranks(models, [(review['model1'], review['model2'], review['winner_model']) for review in reviews])

    with open(reviews_filepath, 'w') as f:
        json.dump(reviews, f, indent=4)
    with open(ranks_filepath, 'w') as f:
        json.dump(ranks, f, indent=4)

    return True

def evaluate_models(models, exclude_reviews):
    for model_type, model_name in models:
        generate_assistant_replies(model_type, model_name)
    if not exclude_reviews:
        while generate_reviews():
            pass
