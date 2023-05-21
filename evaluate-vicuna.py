import os
import json
import tqdm
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM

class OpenAssistant:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto').eval()

    def reply(self, question):
        prompt = '<|prompter|>' + question + self.tokenizer.eos_token + '<|assistant|>'

        # TODO: max_length should be taken from the model and not hardcoded.
        model_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2047 - 400).to(0)

        # TODO: What is this for? Is this needed? I just took it from https://github.com/Open-Assistant/oasst-model-eval
        if 'token_type_ids' in model_input:
            del model_input['token_type_ids']

        model_output = self.model.generate(
            **model_input,
            early_stopping=True, # TODO: Why? Isn't this only for beam search? Also taken from https://github.com/Open-Assistant/oasst-model-eval
            min_new_tokens=1,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.2,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]

        # TODO: What's that truncation for? Also just taken from https://github.com/Open-Assistant/oasst-model-eval I think
        output_decoded = self.tokenizer.decode(model_output, truncate_before_pattern=[r'\n\n^#', "^'''", '\n\n\n'])
        return output_decoded.split('<|assistant|>')[-1].replace(self.tokenizer.eos_token, '').strip()

class OpenAI:
    def __init__(self, model_name):
        self.model_name = model_name

    def reply(self, question, system_message='You are a helpful assistant.'):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                { 'role': 'system', 'content': system_message },
                { 'role': 'user', 'content': question },
            ],
            max_tokens=1024,
        )['choices'][0]['message']['content']

def generate_assistant_replies(general_model_class, specific_model_id):
    models = {
        'open-assistant': OpenAssistant,
        'open-ai': OpenAI,
    }

    model = models[general_model_class](specific_model_id)

    with open('questions.json') as f:
        questions = json.load(f)

    answers = dict([(question_id, model.reply(question)) for question_id, question in tqdm.tqdm(questions.items())])

    with open(os.path.join('reports', 'vicuna', 'answers', specific_model_id.replace('/', '--') + '.json', 'w')) as f:
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
    with open(os.path.join('reports', 'vicuna', 'answers', model_id1.replace('/', '--') + '.json')) as f:
        answers1 = json.load(f)
    with open(os.path.join('reports', 'vicuna', 'answers', model_id2.replace('/', '--') + '.json')) as f:
        answers2 = json.load(f)

    reviewer = OpenAI('gpt-3.5-turbo')

    reviews = {}
    for question_id, question in questions.items():
        answer1 = answers1[question_id]
        answer2 = answers2[question_id]
        system_message, prompter_message = create_reviewer_prompt(question, answer1, answer2)
        reviews[question_id] = reviewer.reply(prompter_message, system_message)

    with open(os.path.join('reports', 'vicuna', 'reviews', model_id1.replace('/', '--') + ' vs. ' + model_id2.replace('/', '--') + '.json')) as f:
        json.dump(reviews, f)

if __name__ == '__main__':
    generate_assistant_replies('open-ai', 'gpt-3.5-turbo')
    generate_assistant_replies('open-assistant', 'OpenAssistant/oasst-sft-1-pythia-12b')
    generate_reviews('gpt-3.5-turbo', 'OpenAssistant/oasst-sft-1-pythia-12b')
