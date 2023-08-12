from evaluation.models.huggingface_backends.data_parallel import DataParallelBackend

def create_model(*, tokenizer_path, model_path, dtype):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = 'left'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map='auto',
    )

    return {
        'tokenizer': tokenizer,
        'model': model,
    }

def compute_model_responses(*, model, batch):
    import torch

    tokenizer = model['tokenizer']
    model = model['model']

    sampling_parameters_to_batch_items = {}

    for i, batch_item in enumerate(batch):
        temperature = batch_item['temperature']
        if temperature is None:
            temperature = 1.0

        max_new_token = batch_item['max_new_tokens']
        assert max_new_token is not None

        sampling_parameters = (temperature, max_new_token)

        if sampling_parameters not in sampling_parameters_to_batch_items:
            sampling_parameters_to_batch_items[sampling_parameters] = []
        sampling_parameters_to_batch_items[sampling_parameters].append(batch_item)

    for (temperature, max_new_token), batch_items_with_specific_sampling_parameters in sampling_parameters_to_batch_items.items():
        prompts = [batch_item['prompt'] for batch_item in batch_items_with_specific_sampling_parameters]

        input_ids = []
        attention_masks = []
        for prompt in prompts:
            if isinstance(prompt, str):
                tokens = tokenizer(prompt)
                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])
            elif isinstance(prompt, tuple):
                assert len(prompt) == 2 and prompt[0] == 'tokens'
                input_ids.append(prompt[1])
                attention_masks.append([1] * len(prompt[1]))
            else:
                raise

        input_ids = torch.tensor(input_ids, device='cuda')
        attention_masks = torch.tensor(attention_masks, device='cuda')

        generation_kwargs = {}
        for token_name, token_type in [('BOS', 'bos_token_id'), ('EOS', 'eos_token_id')]:
            if getattr(model.generation_config, token_type) is None and getattr(model.config, token_type) is None:
                continue
            if getattr(model.generation_config, token_type) is None and getattr(model.config, token_type) is not None:
                generation_kwargs[token_type] = getattr(model.config, token_type)
                continue
            if getattr(model.generation_config, token_type) is not None and getattr(model.config, token_type) is None:
                generation_kwargs[token_type] = getattr(model.generation_config, token_type)
                continue
            if getattr(model.generation_config, token_type) == getattr(model.config, token_type):
                generation_kwargs[token_type] = getattr(model.generation_config, token_type)
                continue
            print('WARNING: ' + token_name + ' token id in generation_config.json is different than to config.json. Using config.json.')
            generation_kwargs[token_type] = getattr(model.config, token_type)

        output_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,

            generation_config=model.generation_config,

            # See the following link for more details & a list of the parameters
            # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig

            # Parameters that control the length of the output
            max_new_tokens=max_new_token,
            min_new_tokens=1,

            # Parameters that control the generation strategy used
            do_sample=temperature > 1e-8,
            num_beams=1,

            # Parameters for manipulation of the model output logits
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            typical_p=1.0,
            epsilon_cutoff=0.0,
            eta_cutoff=0.0,
            diversity_penalty=0.0,
            repetition_penalty=1.0,
            encoder_repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
            renormalize_logits=False,

            **generation_kwargs,
        )

        for i in range(len(batch_items_with_specific_sampling_parameters)):
            response = output_tokens[i]
            response = response[len(input_ids[i]):]
            response = tokenizer.decode(response)
            result_pipe = batch_items_with_specific_sampling_parameters[i]['result_pipe']
            result_pipe.send(('response', response))
            result_pipe.close()

backend = DataParallelBackend(
    backend_name='hf_transformers',
    worker_functions={
        'create_model': create_model,
        'compute_model_responses': compute_model_responses,
    },
    worker_is_blocking=True,
)

def run_inference(**kwargs):
    return backend.run_inference(**kwargs)

def unload_model():
    return backend.unload_model()
