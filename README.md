# Open-Assistant Automatic Model Evaluation

This repository contains tools to automatically evaluate Open-Assistant models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the website to view those reports.
The current relevant reports can be viewed [on github pages site for this repository](https://tju01.github.io/oasst-openai-evals/).

Right now only OpenAI evals is supported as a benchmark, but there are plans for other benchmarks.

## Evaluating the model on benchmarks

### Installation

Run `./setup.py` on an ubuntu system.

### Evaluation

Run `./main.py <model_name>` where `<model_name>` is the path to an Open-Assistant model, e.g. `./main.py OpenAssistant/pythia-12b-sft-v8-7k-steps`.
This will generate an evaluation report in the `reports/` folder.

## Viewing the reports

Go to https://tju01.github.io/oasst-openai-evals/.
Click on `Click here to show & edit report urls`.
Enter the URL of the report you generated in the previous step (it needs to be accessible) as an URL in some way.
