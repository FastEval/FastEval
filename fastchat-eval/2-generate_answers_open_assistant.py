from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import shortuuid

def generate(prompt, tokenizer, model):
    # TODO: max_length should be taken from the model and not hardcoded.
    model_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2047 - 400).to(0)

    # TODO: What is this for? Is this needed? I just took it from https://github.com/Open-Assistant/oasst-model-eval
    if 'token_type_ids' in model_input:
        del model_input['token_type_ids']

    model_output = model.generate(
        **model_input,
        early_stopping=True, # TODO: Why? Isn't this only for beam search? Also taken from https://github.com/Open-Assistant/oasst-model-eval
        min_new_tokens=1,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )[0]

    # TODO: What's that truncation for? Also just taken from https://github.com/Open-Assistant/oasst-model-eval I think
    output_decoded = tokenizer.decode(model_output, truncate_before_pattern=[r'\n\n^#', "^'''", '\n\n\n'])
    reply = output_decoded.split('<|assistant|>')[-1].replace(tokenizer.eos_token, '').strip()
    return reply

def main():
    model_path = 'OpenAssistant/oasst-sft-1-pythia-12b'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto').eval()

    with open('table/question.jsonl', 'r') as f:
        questions = [json.load(l) for l in f]

    answers = [{
        'question_id': question['question_id'],
        'text': generate('<|prompter|>' + question['text'] + tokenizer.eos_token + '<|assistant|>', tokenizer, model),
        'answer_id': shortuuid.uuid(),
        'metadata': {},
    } for question in questions]

    with open('table/answer/answer_pythia.jsonl', 'w') as f:
        f.write([json.dumps(answer) for answer in answers].join('\n'))

if __name__ == '__main__':
    main()
