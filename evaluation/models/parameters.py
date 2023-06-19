GENERATION_PARAMETERS = {
    # See the following link for more details & a list of the parameters
    # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig

    # Parameters that control the length of the output
    'max_new_tokens': 400,
    'min_new_tokens': 1,

    # Parameters that control the generation strategy used
    'do_sample': True,
    'num_beams': 1,

    # Parameters for manipulation of the model output logits
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 1.0,
    'typical_p': 1.0,
    'epsilon_cutoff': 0.0,
    'eta_cutoff': 0.0,
    'diversity_penalty': 0.0,
    'repetition_penalty': 1.0,
    'encoder_repetition_penalty': 1.0,
    'length_penalty': 1.0,
    'no_repeat_ngram_size': 0,
    'renormalize_logits': False,
}
