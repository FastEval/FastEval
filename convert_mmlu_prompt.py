#!/usr/bin/env python3

import json

with open('data/mmlu-prompt.json') as f:
    data = json.load(f)

def convert(prompt):
    lines = prompt.split('\n')
    lines = lines[2:]
    few_shot_items = '\n'.join(lines).split('\nQ: ')

    assert few_shot_items[0].startswith('Q: ')
    few_shot_items[0] = few_shot_items[0][3:]

    few_shot_items_structured = []
    for item in few_shot_items:
        first_split = item.split('\n(A) ')
        question = first_split[0]
        second_part = '(A) ' + ('\n(A) '.join(first_split[1:]))
        qa = second_part.split("\nA: Let's think step by step. ")
        assert len(qa) == 2

        options = qa[0]
        first_option, remaining = options.split('(B) ')
        second_option, remaining = remaining.split('(C) ')
        third_option, remaining = remaining.split('(D) ')
        fourth_option = remaining

        assert first_option.startswith('(A) ')
        first_option = first_option[4:]

        options_array = [
            first_option.strip(),
            second_option.strip(),
            third_option.strip(),
        ]

        if ' (E) ' in fourth_option:
            fourth_option, fifth_option = remaining.split(' (E) ')
            options_array.append(fourth_option.strip())
            options_array.append(fifth_option.strip())
        else:
            options_array.append(fourth_option.strip())

        answer = qa[1]
        few_shot_items_structured.append({ 'question': question, 'options': options_array, 'answer': answer })

    return few_shot_items_structured

for k, v in data.items():
    data[k] = convert(v)

with open('data/mmlu-prompt-structured.json', 'w') as f:
    json.dump(data, f, indent=4)
