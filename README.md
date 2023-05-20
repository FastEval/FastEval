# Open-Assistant Automatic Model Evaluation

This repository contains tools to automatically evaluate Open-Assistant models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the [website to view those reports](https://tju01.github.io/oasst-openai-evals/).

Right now, only the [OpenAI evals](https://github.com/openai/evals) benchmark is supported, but there are plans for other benchmarks.

## Evaluating the model on benchmarks

### Installation

1. Make sure `python 3.10` is installed
2. Create a virtual environment: `python3.10 -m venv .venv`
3. Activate the venv: `source .venv/bin/activate`
4. Install dependencies by executing `pip install -r requirements.txt` in the root directory of this repository

### Evaluation

Run `./evaluate.py <model_name>` where `<model_name>` is the path to an Open-Assistant model.
This path can either be a local folder or an huggingface model.
For example, use the following command to evaluate the `pythia-12b-sft-v8-7k-steps` model from Open-Assistant:
```
./evaluate.py OpenAssistant/pythia-12b-sft-v8-7k-steps
```
This will generate an evaluation report in the `reports/OpenAssistant--pythia-12b-sft-v8-7k-steps` folder if it doesn't exist already (which it does for this model).

## Viewing the reports

Go to https://tju01.github.io/oasst-openai-evals/ or open the `index.html` file in your browser.
Use the `Click here to show & edit report urls` button on the top to show the URLs of the reports that are shown in the table.
Enter the URL of the report you generated in the previous step (it needs to be accessible as an URL in some way).
Click somewhere outside the textarea and the table should update with your newly added report.
