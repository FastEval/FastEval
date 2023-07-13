import threading
import time
import gc

import transformers

import evaluation.utils

lock = threading.Lock()
model = None
current_batch = []

def unload_model(use_lock=True):
    global model
    global current_batch

    if use_lock:
        lock.acquire()

    if model is not None:
        model = None
        gc.collect()

    current_batch = []

    if use_lock:
        lock.release()

def process_current_batch():
    global current_batch

    time.sleep(0.05)

    lock.acquire()

    if len(current_batch) == 0:
        lock.release()
        return

    # We just store a reference to the model here to make sure that the prompt is evaluated with the correct model
    # It could theoretically happen (in the rest of the code), that while we are waiting for some responses to be computed,
    # the underlying model should be changed. This is not something that should currently happen since we evaluate one
    # model at a time and wait for all the responses before switching the model.
    # But just to prevent future possible bugs, we make really sure that we have the correct model here.
    for batch_item in current_batch:
        assert batch_item['model'] is model

    temperatures = [batch_item['temperature'] if batch_item['temperature'] is not None else 1.0 for batch_item in current_batch]
    temperatures_to_batch_items = { temperature: [] for temperature in set(temperatures) }
    for i, batch_item in enumerate(current_batch):
        temperature = batch_item['temperature']
        temperatures_to_batch_items[temperature].append(batch_item)

    for temperature, batch_items_with_specific_temperature in temperatures_to_batch_items.items():
        prompts = [batch_item['prompt'] for batch_item in batch_items_with_specific_temperature]
        responses = model['pipeline'](
            prompts,

            # See the following link for more details & a list of the parameters
            # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig

            # Parameters that control the length of the output
            max_new_tokens=model['max_new_tokens'],
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

            batch_size=model['max_batch_size'],
        )

        responses = [responses[i][0]['generated_text'][len(prompts[i]):] for i in range(len(batch_items_with_specific_temperature))]
        for i in range(len(batch_items_with_specific_temperature)):
            batch_items_with_specific_temperature[i]['response'] = responses[i]

    for item in current_batch:
        item['obtained_response'] = False
        with item['condition']:
            item['condition'].notify_all()

    while not all(item['obtained_response'] for item in current_batch):
        time.sleep(0.01)

    current_batch = []

    lock.release()

def wait_for_response(condition):
    thread = threading.Thread(target=process_current_batch)
    thread.start()

    with condition:
        condition.wait()

    for item in current_batch:
        if item['condition'] != condition:
            continue
        response = item['response']
        item['obtained_response'] = True
        return response

    raise

def run_inference(*, prompt, tokenizer_path, model_path, dtype, max_new_tokens, temperature, max_batch_size):
    global model

    lock.acquire()

    evaluation.utils.switch_gpu_model_type('hf_transformers')

    if (model is None
            or model['tokenizer_path'] != tokenizer_path
            or model['model_path'] != model_path
            or model['dtype'] != dtype
            or model['max_new_tokens'] != max_new_tokens):
        unload_model(False)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.padding_side = 'left'
        model = {
            'tokenizer_path': tokenizer_path,
            'model_path': model_path,
            'dtype': dtype,
            'max_new_tokens': max_new_tokens,
            'max_batch_size': max_batch_size,
            'pipeline': transformers.pipeline(
                'text-generation',
                model=model_path,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map='auto'
            ),
        }

    condition = threading.Condition()
    current_batch.append({ 'prompt': prompt, 'condition': condition, 'model': model, 'temperature': temperature })
    lock.release()
    return wait_for_response(condition)
