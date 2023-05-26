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

### Evaluation

Run `./evaluate.py -b <benchmark_name_1> [<benchmark_name_2>...] -m <model_name_1> [<model_name_2>...]`.
The benchmark name(s) can be `all`, `openai-evals`, `vicuna` or `lm-evaluation-harness`.
The model names can be either a path to a local folder or a huggingface path.
For example, use the following command to evaluate the [`pythia-12b-sft-v8-2.5k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-2.5k-steps) model from Open-Assistant on the OpenAI evals benchmark:
```
./evaluate.py -b openai-evals -m OpenAssistant/pythia-12b-sft-v8-2.5k-steps
```
This will generate an evaluation report in the `reports/openai-evals/OpenAssistant--pythia-12b-sft-v8-7k-steps` folder.
If the report already exists, then the evaluation is skipped.

## Viewing the reports

Use `python3 -m http.server` in the root of this repository.
This will start a simple webserver for static files.
The webserver usually runs on port `8000`, so you can go to http://127.0.0.1:8000/ and view the results.
