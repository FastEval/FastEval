import json
import os
import time

import openai
import ray

import shortuuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 5

@ray.remote(num_cpus=4)
def get_eval(sys_prompt, user_prompt: str, max_tokens: int):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]

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
