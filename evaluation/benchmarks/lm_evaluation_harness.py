from evaluation.utils import get_model_type

def evaluate_model(model):
    model_type = get_model_type(model)
    print('lm-evaluation-harness: Evaluating', model, model_type)

def evaluate_models(models):
    for model_name in models:
        evaluate_model(model_name)
