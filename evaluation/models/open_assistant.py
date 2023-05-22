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
