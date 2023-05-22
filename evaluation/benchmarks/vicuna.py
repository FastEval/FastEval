import os
import json

import tqdm

from evaluation.utils import replace_model_name_slashes
from evaluation.models.open_ai import OpenAI
from evaluation.models.open_assistant import OpenAssistant

def generate_assistant_replies(general_model_class, specific_model_id):
    models = {
        'open-assistant': OpenAssistant,
        'open-ai': OpenAI,
    }

    model = models[general_model_class](specific_model_id)

    with open('questions.json') as f:
        questions = json.load(f)

    answers = dict([(question_id, model.reply(question)) for question_id, question in tqdm.tqdm(questions.items())])

    with open(os.path.join('reports', 'vicuna', 'answers', replace_model_name_slashes(specific_model_id) + '.json', 'w')) as f:
        json.dump(answers, f)

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

def generate_reviews(model_id1, model_id2):
    with open('questions.json') as f:
        questions = json.load(f)
    with open(os.path.join('reports', 'vicuna', 'answers', replace_model_name_slashes(model_id1) + '.json')) as f:
        answers1 = json.load(f)
    with open(os.path.join('reports', 'vicuna', 'answers', replace_model_name_slashes(model_id2) + '.json')) as f:
        answers2 = json.load(f)

    reviewer = OpenAI('gpt-3.5-turbo')

    reviews = {}
    for question_id, question in questions.items():
        answer1 = answers1[question_id]
        answer2 = answers2[question_id]
        system_message, prompter_message = create_reviewer_prompt(question, answer1, answer2)
        reviews[question_id] = reviewer.reply(prompter_message, system_message)

    with open(os.path.join('reports', 'vicuna', 'reviews', replace_model_name_slashes(model_id1) + ' vs. ' + replace_model_name_slashes(model_id2) + '.json')) as f:
        json.dump(reviews, f)

def evaluate_models(models):
    for general_model_class, specific_model_id in models:
        generate_assistant_replies(general_model_class, specific_model_id)
    # generate_reviews('gpt-3.5-turbo', 'OpenAssistant/oasst-sft-1-pythia-12b')
