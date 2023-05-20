import json
import os
import openai
import shortuuid
import tqdm

def get_answer(question_id: int, question: str):
    ans = { 'answer_id': shortuuid.uuid(), 'model_id': 'gpt-3.5-turbo', 'question_id': question_id }

    ans['text'] = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            { 'role': 'system', 'content': 'You are a helpful assistant.' },
            { 'role': 'user', 'content': question },
        ],
        max_tokens=1024,
    )['choices'][0]['message']['content']

    return ans

def main():
    questions_dict = {}
    with open('table/question.jsonl') as f:
        for line in f:
            if not line:
                continue
            q = json.loads(line)
            questions_dict[q['question_id']] = q['text']

    answers = []
    for question_id, question in tqdm.tqdm(questions_dict.items()):
        answers.append(get_answer(question_id, question))

    answers.sort(key=lambda x: x['question_id'])

    with open(os.path.expanduser('table/answer/answer_gpt35.jsonl'), 'w') as f:
        table = [json.dumps(ans) for ans in answers]
        f.write('\n'.join(table))

if __name__ == '__main__':
    main()
