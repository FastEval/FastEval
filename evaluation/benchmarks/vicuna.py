import os
import json
import random

import tqdm

from evaluation.utils import replace_model_name_slashes, undo_replace_model_name_slashes, create_model

def generate_assistant_replies(model_type, model_name):
    answers_filepath = os.path.join('reports', 'vicuna', 'answers', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(answers_filepath):
        return

    model = create_model(model_type, model_name)

    with open('questions.json') as f:
        questions = json.load(f)

    answers = dict([(question_id, model.reply([('user', question)])) for question_id, question in tqdm.tqdm(questions.items())])

    os.makedirs(os.path.dirname(answers_filepath), exist_ok=True)
    with open(answers_filepath, 'w') as f:
        json.dump(answers, f, indent=4)

def create_reviewer_prompt(question, answer1, answer2):
    system_message = 'You are a helpful and precise assistant for checking the quality of the answer.'

    # https://medium.com/@geronimo7/open-source-chatbots-in-the-wild-9a44d7a41a48
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
        + 'Please rate the helpfulness, relevance, accuracy, level of details of their responses. '
        + "Please output who provided the best answer. If both answers are equally good and it's hard to decide on a winner then please call it a tie. "
        + "Your output should look like this: 'Winner: Assistant 1' or 'Winner: Assistant 2' or 'Tie'. "
        + 'Do not output anything else.\n'
        + "\n")

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

def save_reviews(reviews, models_results):
    reviews_filepath = os.path.join('reports', 'vicuna', 'reviews.json')
    os.makedirs(os.path.dirname(reviews_filepath), exist_ok=True)
    with open(reviews_filepath, 'w') as f:
        json.dump({ 'reviews': reviews, 'models': models_results }, f, indent=4)

def generate_reviews():
    with open('questions.json') as f:
        questions = json.load(f)

    answers = {}
    answers_base_directory = os.path.join('reports', 'vicuna', 'answers')
    for model_filename in os.listdir(answers_base_directory):
        model_name = undo_replace_model_name_slashes(model_filename.replace('.json', ''))
        with open(os.path.join(answers_base_directory, model_filename)) as f:
            answers[model_name] = json.load(f)
    models = list(answers.keys())

    reviewer = create_model('gpt-3.5-turbo')

    reviews = []
    models_results = dict([(model, {
        'num_matches': 0,
        'num_wins': 0,
        'num_ties': 0,
        'elo_rank': 1000,
    }) for model in models])

    for i in tqdm.tqdm(range(1000)):
        question_id, question = random.choice(list(questions.items()))
        model_name1, model_name2 = random.sample(models, 2)
        system_message, prompter_message = create_reviewer_prompt(question, answers[model_name1][question_id], answers[model_name2][question_id])

        review = reviewer.reply([
            ('system', system_message),
            ('user', prompter_message),
        ])

        winner_model = find_winner(review.split('\n')[-1].lower())
        if winner_model is None:
            winner_model = find_winner(review.split('\n')[0].lower())
        if winner_model is None:
            print(review)
            continue

        reviews.append({
            'question_id': question_id,
            'model1': model_name1,
            'model2': model_name2,
            'review': review,
            'winner_model': winner_model
        })

        models_results[model_name1]['num_matches'] += 1
        models_results[model_name2]['num_matches'] += 1
        if winner_model == '1':
            models_results[model_name1]['num_wins'] += 1
        elif winner_model == '2':
            models_results[model_name2]['num_wins'] += 1
        elif winner_model == 'tie':
            models_results[model_name1]['num_ties'] += 1
            models_results[model_name2]['num_ties'] += 1

        SCALE = 400
        BASE = 10
        K = 32
        model1_rank = models_results[model_name1]['elo_rank']
        model2_rank = models_results[model_name2]['elo_rank']
        e1 = 1 / (1 + BASE ** ((model2_rank - model1_rank) / SCALE))
        e2 = 1 / (1 + BASE ** ((model1_rank - model2_rank) / SCALE))
        if winner_model == '1':
            sa = 1
        elif winner_model == '2':
            sa = 0
        elif winner_model == 'tie':
            sa = 0.5
        models_results[model_name1]['elo_rank'] += K * (sa - e1)
        models_results[model_name2]['elo_rank'] += K * (1 - sa - e2)

        if i != 0 and i % 10 == 0:
            save_reviews(reviews, models_results)

    save_reviews(reviews, models_results)

def evaluate_models(models):
    for model_type, model_name in models:
        generate_assistant_replies(model_type, model_name)
    generate_reviews()
