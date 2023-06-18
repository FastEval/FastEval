import os
import json
import random

from evaluation.utils import replace_model_name_slashes, undo_replace_model_name_slashes, create_model, compute_model_replies

def generate_assistant_replies(model_type, model_name):
    answers_filepath = os.path.join('reports', 'vicuna', 'answers', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(answers_filepath):
        return False

    model = create_model(model_type, model_name)

    with open('data/vicuna/questions.json') as f:
        questions = json.load(f)

    questions_items = questions.items()
    replies = compute_model_replies(model, [[('user', question)] for _question_id, question in questions_items])
    answers = { question_id: replies[i] for i, (question_id, _question) in enumerate(questions_items) }

    os.makedirs(os.path.dirname(answers_filepath), exist_ok=True)
    with open(answers_filepath, 'w') as f:
        json.dump(answers, f, indent=4)

    return True

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
                print(line)
                return None
            winner_model = '1'
        elif (possible_match + ' 2') in line:
            if winner_model is not None:
                print(line)
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
                print(line)
                return None
            winner_model = 'tie'
            break
    if line in ['Tie.']:
        if winner_model is not None:
            print(line)
            return None
        winner_model = 'tie'

    return winner_model

def compute_elo_ranks_single_seed(model_names, matches):
    SCALE = 400
    BASE = 10
    K = 32

    model_ranks = dict([(model, 1000) for model in model_names])
    for model1, model2, winner_model in matches:
        model1_rank = model_ranks[model1]
        model2_rank = model_ranks[model2]
        e1 = 1 / (1 + BASE ** ((model2_rank - model1_rank) / SCALE))
        e2 = 1 / (1 + BASE ** ((model1_rank - model2_rank) / SCALE))
        if winner_model == '1':
            sa = 1
        elif winner_model == '2':
            sa = 0
        elif winner_model == 'tie':
            sa = 0.5
        model_ranks[model1] += K * (sa - e1)
        model_ranks[model2] += K * (1 - sa - e2)

    return model_ranks

def compute_elo_ranks(model_names, matches):
    model_ranks = dict([(model, 0) for model in model_names])
    num_seeds = 1000
    for _ in range(num_seeds):
        random.shuffle(matches)
        for model_name, model_rank in compute_elo_ranks_single_seed(model_names, matches).items():
            model_ranks[model_name] += model_rank / num_seeds
    return model_ranks

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

    reviewer = create_model('openai', 'gpt-3.5-turbo-0301')

    reviews = []
    models_results = dict([(model, {
        'num_matches': 0,
        'num_wins': 0,
        'num_ties': 0,
    }) for model in models])

    for i in range(1000):
        question_id = random.choice(list(questions.keys()))
        model_name1, model_name2 = random.sample(models, 2)
        reviews.append({
            'question_id': question_id,
            'model1': model_name1,
            'model2': model_name2,
        })

    conversations = []
    for review in reviews:
        question_id = review['question_id']
        system_message, prompter_message = create_reviewer_prompt(
            questions[question_id],
            answers[review['model1']][question_id],
            answers[review['model2']][question_id]
        )
        conversations.append([
            ('system', system_message),
            ('user', prompter_message),
        ])

    replies = compute_model_replies(reviewer, conversations)

    for i, review in enumerate(reviews):
        model_name1 = review['model1']
        model_name2 = review['model2']

        reply = replies[i]
        review['review'] = reply

        winner_model = find_winner(reply.split('\n')[-1].lower())
        review['winner_model'] = winner_model
        if winner_model is None:
            print(reply)
            continue

        models_results[model_name1]['num_matches'] += 1
        models_results[model_name2]['num_matches'] += 1
        if winner_model == '1':
            models_results[model_name1]['num_wins'] += 1
        elif winner_model == '2':
            models_results[model_name2]['num_wins'] += 1
        elif winner_model == 'tie':
            models_results[model_name1]['num_ties'] += 1
            models_results[model_name2]['num_ties'] += 1

    reviews = [review for review in reviews if review['winner_model'] is not None]

    elo_ranks = compute_elo_ranks(models, [(review['model1'], review['model2'], review['winner_model']) for review in reviews])
    for model_name, model_results in models_results.items():
        model_results['elo_rank'] = elo_ranks[model_name]

    reviews_filepath = os.path.join('reports', 'vicuna', 'reviews.json')
    os.makedirs(os.path.dirname(reviews_filepath), exist_ok=True)
    with open(reviews_filepath, 'w') as f:
        json.dump({ 'reviews': reviews, 'models': models_results }, f, indent=4)

def evaluate_models(models, exclude_reviews):
    did_evaluate_some_model = False
    for model_type, model_name in models:
        if generate_assistant_replies(model_type, model_name):
            did_evaluate_some_model = True
    if not exclude_reviews and (did_evaluate_some_model or not os.path.exists('reports/vicuna/reviews.json')):
        generate_reviews()
