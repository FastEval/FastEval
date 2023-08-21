# FastEval

This project allows you to quickly evaluate language models on a number of benchmarks.
It focuses on instruction-following and chat language models. See the [comparison to lm-evaluation-harness](docs/comparison-to-lm-eval.md) for more information.
There is also a [leaderboard](https://fasteval.github.io/FastEval/).

## Features

- **Evaluation on various benchmarks with a single command.** Supported benchmarks are [MT‑Bench](https://arxiv.org/abs/2306.05685) for conversational capabilities, [HumanEval+](https://github.com/evalplus/evalplus) for Python coding performance, Chain of Thought (GSM8K + MATH + BBH + MMLU) for reasoning capabilities, [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness) for general capabilities as well as [custom test data](docs/custom-test-data.md).
- **High performance.** FastEval uses [vLLM](https://github.com/vllm-project/vllm) for fast inference by default and can also optionally make use of [text-generation-inference](https://github.com/huggingface/text-generation-inference). Both methods are ~20x faster than using huggingface transformers.
- **Detailed information about model performance.** FastEval saves the outputs of the language model and other intermediate results to disk. This makes it possible to get deeper insight into model performance. You can look at the [performance on different categories](https://fasteval.github.io/FastEval/#?benchmark=mt-bench) and even inspect [individual model outputs](https://fasteval.github.io/FastEval/#?benchmark=cot&task=bbh/date_understanding&id=eb74c9e1-8836-4c3a-8f50-a25808d20eee).
- **Use of model-specific prompt templates**: FastEval uses the right prompt template depending on the evaluated model. Many prompt templates are supported and the use of [Fastchat](https://github.com/lm-sys/FastChat) expands this even further.

## Installation

```bash
# Install `python3.10`, `python3.10-venv` and `python3.10-dev`.
# The following command assumes an ubuntu >= 22.04 system.
apt install python3.10 python3.10-venv python3.10-dev

# Clone this repository, make it the current working directory
git clone --depth 1 https://github.com/FastEval/FastEval.git
cd FastEval

# Set up the virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This already installs [vLLM](https://github.com/vllm-project/vllm) for fast inference which is usually enough [for most models](https://vllm.readthedocs.io/en/latest/models/supported_models.html). However, if you encounter any problems with vLLM or your model is not supported, FastEval also supports using [text-generation-inference](https://github.com/huggingface/text-generation-inference) as an alternative. Please see [here](docs/text-generation-inference.md) if you would like to use text-generation-inference.

### OpenAI API Key

[MT-Bench](https://arxiv.org/abs/2306.05685) uses GPT-4 as a judge for evaluating model outputs. For this benchmark, you need to configure an OpenAI API key by setting the `OPENAI_API_KEY` environment variable. Note that methods other than setting this environment variable won't work. The cost of evaluating a new model on MT-Bench is approximately $5.

## Evaluation

⚠️ Running `fasteval` currently executes untrusted code from models with remote code as well as LLM generated code when using [HumanEval+](https://github.com/evalplus/evalplus). Please note that there is currently no integrated sandbox.

To evaluate a new model, call `fasteval` in the following way:
```
./fasteval [-b <benchmark_name_1>...] -t model_type -m model_name
````

The `-b` flag specifies the benchmarks that you want to evaluate your model on. The default is `all`, but you can also specify one or multiple individual benchmarks. Possible values are [`mt-bench`](https://fasteval.github.io/FastEval/#?benchmark=mt-bench), [`human-eval-plus`](https://fasteval.github.io/FastEval/#?benchmark=human-eval-plus), [`cot`](https://fasteval.github.io/FastEval/#?benchmark=cot), [`lm-evaluation-harness`](https://fasteval.github.io/FastEval/#?benchmark=lm-evaluation-harness) and [`custom-test-data`](docs/custom-test-data.md). Note that lm-evaluation-harness doesn't use fast inference or prompt templates.

The `-t` flag specifies the type of the model which is either the prompt template or the API client that will be used. [Please see here](docs/model-type.md) for information on which model type to select for your model.

The `-m` flag specifies the name of the model which can be a path to a model on huggingface, a local folder or an OpenAI model name.

For example, this command will evaluate [`OpenAssistant/pythia-12b-sft-v8-2.5k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-2.5k-steps) on [HumanEval+](https://fasteval.github.io/FastEval/#?benchmark=human-eval-plus):
```bash
./fasteval -b human-eval-plus -t open-assistant -m OpenAssistant/pythia-12b-sft-v8-2.5k-steps
```

There are also flags available for enabling & configuring data parallel evaluation, setting model arguments and changing the inference backend. Please use `./fasteval -h` or see [here](docs/cmdline-flags.md) for more information.

## Viewing the results

A very short summary of the final scores will be written to stdout after the evaluation has finished.

More details are available through the web UI where you can view performance on different subtasks or inspect individual model inputs & outputs.
To access the web UI, use `python3 -m http.server` in the root folder of this repository.
This will start a simple webserver for static files.
The server usually runs on port `8000` in which case you can view the detailed results at [localhost:8000](http://localhost:8000).
