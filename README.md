# Instruction following language model evaluation

This repository contains code to automatically evaluate instruction following language models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the [website to view those reports](https://tju01.github.io/ilm-eval/).

## Supported benchmarks & models

Right now, the following benchmarks are supported:
- [OpenAI evals](https://github.com/openai/evals): Contains various tasks to measure different capabilities of instruction-following language models. Uses both basic tasks that are just compared to the solution directly and model-graded tasks where another language model is used for evaluation.
- [Vicuna Elo Ranking](https://lmsys.org/blog/2023-03-30-vicuna): Uses another more capable model like `gpt-4` or `gpt-3.5-turbo` for comparing outputs of different models and computes win rates and Elo ratings based on these comparisons.
- [HumanEval+](https://github.com/evalplus/evalplus): Gives the model the start of a function as input with a docstring comment on what the function is supposed to do. The model should then complete the code. The model output code is evaluated for correctness by running it against a few tests.
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness): This is the only benchmark that does not take the prompt format into account. However, despite this fact, lm-evaluation-harness is very popular for evaluating instruction following language models. It is therefore part of the evaluation here together with other benchmarks that take the prompt format into account.

Since this repository is about instruction following models and different instruction models require different prompt formatting, a corresponding implementation of the prompt format is needed to evaluate a model. The following model types are currently supported:
- [Open-Assistant](https://open-assistant.io)
- [OpenAI](https://platform.openai.com/docs/models)
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Falcon Instruct](https://huggingface.co/tiiuae)

Note that some prompt formats are also used by other models. In this case, you can also use this tool. E.g. the alpaca prompt format is also used by other models like [`NousResearch/Nous-Hermes-13b`](https://huggingface.co/NousResearch/Nous-Hermes-13b).

## Evaluating models on benchmarks

### Installation

```bash
# Install `python3.10`, `python3.10-venv` and `git-lfs`.
# The following code assumes an ubuntu system.
apt install python3.10 python3.10-venv git-lfs

# Clone this repository, make it the current working directory
git clone --depth 1 https://github.com/tju01/ilm-eval.git
cd ilm-eval

# Set up the virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OpenAI API

Some benchmarks use `gpt-3.5-turbo` as a model to judge the output of another model. This is the case for `OpenAI evals` as well as for `Vicuna Elo Rank`.
For these benchmarks (and also for evaluating `gpt-3.5-turbo` itself), you need to configure an OpenAI API key.
There are multiple methods for doing that, but the simplest one is to set the `OPENAI_API_KEY` environment variable to the API key you can obtain [here](https://platform.openai.com/account/api-keys). 
The cost of evaluating `gpt-3.5-turbo` or using it for the vicuna benchmark is something like $2.

### Evaluation

Call the `evaluate.py` script in the following way:
```
./evaluate.py [-b <benchmark_name_1>...] -m model_type_1:model_name_1...
````
- A benchmark name can be `all` (default), `openai-evals`, `vicuna`, `human-eval-plus` or `lm-evaluation-harness`.
- A model type can be either `open-assistant`, `openai`, `alpaca` or `falcon-instruct`.
- A model name can be either a path to a local folder or a huggingface path.

For example, use the following command to evaluate the [`pythia-12b-sft-v8-2.5k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-2.5k-steps) model from Open-Assistant on the OpenAI evals benchmark:
```
./evaluate.py -b openai-evals -m open-assistant:OpenAssistant/pythia-12b-sft-v8-2.5k-steps
```
This will generate an evaluation report in the `reports/openai-evals/OpenAssistant--pythia-12b-sft-v8-7k-steps` folder.
If the report already exists, then the evaluation is skipped.

If you want to evaluate multiple models on the vicuna benchmark, you should specify them all in one command since the reviewing with `gpt-3.5-turbo` is done once in the end independently of the number of models evaluated in a single command.

## Viewing the reports

Use `python3 -m http.server` in the root of this repository.
This will start a simple webserver for static files.
The webserver usually runs on port `8000`, so you can go to http://127.0.0.1:8000/ and view the results.
