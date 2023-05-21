import json
import openai
import shortuuid

def get_eval(sys_prompt, user_prompt: str, max_tokens: int):
    return openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            { 'role': 'system', 'content': sys_prompt },
            { 'role': 'user', 'content': user_prompt },
        ],
        temperature=0.2, # TODO: figure out which temperature is best for evaluation
        max_tokens=max_tokens,
    )['choices'][0]['message']['content']

def parse_score(review):
    score_pair = review.split('\n')[0].replace(',', ' ').split(' ')
    if len(score_pair) == 2:
        return [float(score_pair[0]), float(score_pair[1])]
    raise

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

def load(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def main():
    questions = load('table/question.jsonl')
    answers1 = load('table/answer/answer_gpt35.jsonl')
    answers2 = load('table/answer/answer_pythia.jsonl')
    reviewers = load('table/reviewer.jsonl')
    reviewer_prompts = load('table/prompt.jsonl')

    assert len(questions) == len(answers1) == len(answers2)

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

        review_output = get_eval(system_message, prompter_message, 1024)
        review['text'] = review_output
        review['score'] = parse_score(review_output)

        reviews.append(review)

    with open('table/review/review.jsonl', 'w') as f:
        f.write([json.dumps(review) for review in reviews].join('\n'))

if __name__ == '__main__':
    main()
