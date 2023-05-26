# Instruction following language model evaluation

This repository contains code to automatically evaluate instruction following language models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the [website to view those reports](https://tju01.github.io/oasst-automatic-model-eval/).

Right now, the following benchmarks are supported:
- [OpenAI evals](https://github.com/openai/evals)
- [Vicuna benchmark](https://lmsys.org/blog/2023-03-30-vicuna)
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

The following models are currently supported:
- [Open-Assistant](https://open-assistant.io)
- [OpenAI chat models](https://platform.openai.com/docs/models)

## Evaluating models on benchmarks

### Installation

```
# Install `python3.10` and `git-lfs`. The following code assumes an ubuntu system.
apt install python3.10 git-lfs

# Clone this repository, make it the current working directory
git clone https://github.com/tju01/oasst-automatic-model-eval.git
cd oasst-automatic-model-eval

# Set up the virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# The `lm-evaluation-harness` installation is currently buggy, it needs to be installed manually
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### OpenAI API

If you want to evaluate OpenAI models like `gpt-3.5-turbo` or if you want to use the vicuna benchmark which uses `gpt-3.5-turbo` as a reviewer model, then you need to set up an OpenAI API key.
There are multiple methods for doing that, but the simplest one is to set the `OPENAI_API_KEY` environment variable to the API key you can obtain [here](https://platform.openai.com/account/api-keys).
The cost of evaluating `gpt-3.5-turbo` or using it for the vicuna benchmark is something like $2.

### Evaluation

Call the `evaluate.py` script in the following way:
```
./evaluate.py [-b <benchmark_name_1>...] -m model_type_1:model_name_1...
````
- A benchmark name can be `all` (default), `openai-evals`, `vicuna` or `lm-evaluation-harness`.
- A model type can be either `open-assistant` or `openai`.
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
