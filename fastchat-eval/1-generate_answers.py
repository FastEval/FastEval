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

    def reply(self, question):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                { 'role': 'system', 'content': 'You are a helpful assistant.' },
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

    with open('data/input/questions.jsonl', 'r') as f:
        questions = [json.load(l) for l in f]

    answers = [{
        'question_id': question['question_id'],
        'answer_id': shortuuid.uuid(),
        'text': model.reply(question['input']),
        'metadata': {},
    } for question in tqdm.tqdm(questions)]

    with open('data/output/answers/' + specific_model_id.replace('/', '--') + '.jsonl', 'w') as f:
        f.write([json.dumps(answer) for answer in answers].join('\n'))

if __name__ == '__main__':
    generate_replies('open-assistant', 'OpenAssistant/oasst-sft-1-pythia-12b')
    generate_replies('open-ai', 'gpt-3.5-turbo')
