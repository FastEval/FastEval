import json
import tqdm
import shortuuid
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

def generate_replies(general_model_class, specific_model_id):
    models = {
        'open-assistant': OpenAssistant,
        'open-ai': OpenAI,
    }

    model = models[general_model_class](specific_model_id)

    with open('data/input/questions.json') as f:
        questions = json.load(f)

    answers = [{
        'question_id': question['question_id'],
        'answer_id': shortuuid.uuid(),
        'text': model.reply(question['input']),
        'metadata': {},
    } for question in tqdm.tqdm(questions)]

    with open('data/output/answers/' + specific_model_id.replace('/', '--') + '.json', 'w') as f:
        json.dump(answers, f)

def gen_prompt(reviewers, reviewer_prompts, category, question, answer1, answer2):
    reviewer_idx = 0 # Default to general category (index = 0)
    for idx, reviewer in enumerate(reviewers):
        if reviewer['category'] == category:
            reviewer_idx = idx
            break

    prompt_id = reviewers[reviewer_idx]['prompt_id']
    prompt_json = reviewer_prompts[prompt_id - 1]
    assert prompt_json['prompt_id'] == prompt_id

    system_message = prompt_json['system_prompt']
    prompt_template = prompt_json['prompt_template']
    defaults = prompt_json['defaults']
    prompter_message = prompt_template.format(question=question, answer_1=answer1, answer_2=answer2, **defaults)

    return system_message, prompter_message, reviewer_idx + 1

def generate_reviews(model_id1, model_id2):
    with open('data/input/questions.json') as f:
        questions = json.load(f)
    with open('data/output/answers/' + model_id1.replace('/', '-') + '.json') as f:
        answers1 = json.load(f)
    with open('data/output/answers/' + model_id2.replace('/', '-') + '.json') as f:
        answers2 = json.load(f)

    assert len(questions) == len(answers1) == len(answers2)

    with open('data/input/reviewers.json') as f:
        reviewers = json.load(f)
    with open('data/input/reviewers_prompts.json') as f:
        reviewer_prompts = json.load(f)

    openai = OpenAI()

    reviews = []
    for i in range(len(questions)):
        assert questions[i]['question_id'] == answers1[i]['question_id'] == answers2[i]['question_id']

        category = questions[i]['category']
        question = questions[i]['text']
        answer1 = answers1[i]['text']
        answer2 = answers2[i]['text']

        system_message, prompter_message, reviewer_id = gen_prompt(reviewers, reviewer_prompts, category, question, answer1, answer2)

        review = {
            'review_id': shortuuid.uuid(),
            'question_id': questions[i]['question_id'],
            'answer1_id': answers1[i]['answer_id'],
            'answer2_id': answers2[i]['answer_id'],
            'reviewer_id': reviewer_id,
            'metadata': {},
        }

        review_output = openai.reply(prompter_message, system_message)
        review['text'] = review_output

        score_pair = review_output.split('\n')[0].replace(',', ' ').split(' ')
        if len(score_pair) == 2:
            review['score'] = [float(score_pair[0]), float(score_pair[1])]
        else:
            raise

        reviews.append(review)

    with open('data/output/reviews/' + model_id1.replace('/', '-') + ' vs. ' + model_id2.replace('/', '-') + '.json') as f:
        json.dump(reviews, f)

if __name__ == '__main__':
    generate_replies('open-assistant', 'OpenAssistant/oasst-sft-1-pythia-12b')
    generate_replies('open-ai', 'gpt-3.5-turbo')
    generate_reviews('OpenAssistant/oasst-sft-1-pythia-12b', 'gpt-3.5-turbo')
