import os
import json

from evaluation.utils import replace_model_name_slashes, create_model, compute_model_replies

def get_temperature(category):
    return ({
        'writing': 0.7,
        'roleplay': 0.7,
        'extraction': 0.0,
        'math': 0.0,
        'coding': 0.0,
        'reasoning': 0.0,
        'stem': 0.1,
        'humanities': 0.1,
    })[category]

def generate_assistant_replies(model_type, model_name):
    answers_filepath = os.path.join('reports', 'mt-bench', 'answers', replace_model_name_slashes(model_name) + '.json')
    if os.path.exists(answers_filepath):
        return

    model = create_model(model_type, model_name)

    with open('data/mt-bench/questions.json') as f:
        questions = json.load(f)

    # TODO Actually use those temperatures
    for question in questions.values():
        question['temperature'] = get_temperature(question['category'])

    questions_items = questions.items()

    first_turn_conversations = [[('user', question['turns'][0])] for _question_id, question in questions_items]
    first_turn_replies = compute_model_replies(model, first_turn_conversations)
    first_turn_replies = { question_id: first_turn_replies[i] for i, (question_id, _question) in enumerate(questions_items) }

    second_turn_conversations = [[
        ('user', question['turns'][0]),
        ('assistant', first_turn_replies[question_id]),
        ('user', question['turns'][1]),
    ] for question_id, question in questions_items]

    second_turn_replies = compute_model_replies(model, second_turn_conversations)
    second_turn_replies = { question_id: second_turn_replies[i] for i, (question_id, _question) in enumerate(questions_items) }

    all_replies = { question_id: [first_turn_replies[question_id], second_turn_replies[question_id]] for question_id in questions.keys() }

    os.makedirs(os.path.dirname(answers_filepath), exist_ok=True)
    with open(answers_filepath, 'w') as f:
        json.dump(all_replies, f, indent=4)

def evaluate_models(models):
    for model_type, model_name in models:
        generate_assistant_replies(model_type, model_name)
