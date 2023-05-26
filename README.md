# Finetuned language model evaluation

This repository contains tools to automatically evaluate finetuned language models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the [website to view those reports](https://tju01.github.io/oasst-automatic-model-eval/).

Right now, the following benchmarks are supported:
- [OpenAI evals](https://github.com/openai/evals)
- [Vicuna benchmark](https://lmsys.org/blog/2023-03-30-vicuna)
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

The following models are currently supported:
- [Open-Assistant](https://open-assistant.io)
- [OpenAI chat models](https://platform.openai.com/docs/models)

## Evaluating the model on benchmarks

### Installation

1. Install `python3.10` and `git-lfs`, e.g.: `apt install python3.10 git-lfs`
2. Clone this repository: `git clone https://github.com/tju01/oasst-automatic-model-eval.git`
3. Make it the current working directory: `cd oasst-automatic-model-eval`
4. Create a virtual environment: `python3.10 -m venv .venv`
5. Activate the environment: `source .venv/bin/activate`
6. Install the python dependencies: `pip install -r requirements.txt`
7. The `lm-evaluation-harness` installation is currently buggy, you need to [install it manually by cloning the repository](https://github.com/EleutherAI/lm-evaluation-harness#install)

### Evaluation

Run `./evaluate.py -b <benchmark_name> -m <model_name>`.
The `<benchmark_name>` should be one of `openai-evals`, `vicuna` or `lm-evaluation-harness`.
The `<model_name>` can be either a path to a local folder a huggingface path.
For example, use the following command to evaluate the [`pythia-12b-sft-v8-7k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-7k-steps) model from Open-Assistant:
```
./evaluate.py -b openai-evals -m OpenAssistant/pythia-12b-sft-v8-7k-steps
```
This will generate an evaluation report in the `reports/openai-evals/OpenAssistant--pythia-12b-sft-v8-7k-steps` folder if it doesn't exist already (which it does for this model).

## Viewing the reports

Use `python3 -m http.server` in the root of this repository.
This will start a simple webserver for static files.
The webserver usually runs on port `8000`, so you can go to http://127.0.0.1:8000/ and view the results.
