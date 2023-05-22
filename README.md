# Open-Assistant Automatic Model Evaluation

This repository contains tools to automatically evaluate Open-Assistant models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the [website to view those reports](https://tju01.github.io/oasst-automatic-model-eval/).

Right now, only the [OpenAI evals](https://github.com/openai/evals) benchmark is supported.
A [vicuna-style evaluation](https://lmsys.org/blog/2023-03-30-vicuna/) is currently WIP and other benchmarks are also planned.

## Evaluating the model on benchmarks

### Installation

1. Make sure `python3.10` and `git-lfs` are installed
2. Create a virtual environment: `python3.10 -m venv .venv`
3. Activate the venv: `source .venv/bin/activate`
4. Install the python dependencies: `pip install -r requirements.txt`

### Evaluation

Run `./evaluate.py -b openai-evals -m <model_name>` where `<model_name>` is the path to an Open-Assistant model.
This path can either be a local folder or a huggingface model path.
For example, use the following command to evaluate the [`pythia-12b-sft-v8-7k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-7k-steps) model from Open-Assistant:
```
./evaluate.py -b openai-evals -m OpenAssistant/pythia-12b-sft-v8-7k-steps
```
This will generate an evaluation report in the `reports/openai-evals/OpenAssistant--pythia-12b-sft-v8-7k-steps` folder if it doesn't exist already (which it does for this model).

## Viewing the reports

Go to https://tju01.github.io/oasst-automatic-model-eval/ or open the `index.html` file in your browser.
Use the `Click here to show & edit report urls` button on the top to show the URLs of the reports that are shown in the table.
Enter the URL of the report you generated in the previous step (it needs to be accessible as an URL in some way).
Click somewhere outside the textarea and the table should update with your newly added report.
