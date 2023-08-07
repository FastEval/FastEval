# FastEval

This project allows you to quickly evaluate language models on a number of benchmarks. It's currently focused on instruction-following language models, but we plan to extend this in the future. It also contains [a leaderboard](https://fasteval.github.io/FastEval/) for comparison between different models.

## Features

- **Evaluation on various benchmarks with a single command.** Supported benchmarks are [MT‑Bench](https://arxiv.org/abs/2306.05685) for conversational capabilities, [HumanEval+](https://github.com/evalplus/evalplus) for Python coding performance, Chain of Thought (GSM8K + BBH + MMLU) for reasoning capabilities and [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness) for general capabilities.
- **High performance.** FastEval uses [vLLM](https://github.com/vllm-project/vllm) for fast inference by default and can also optionally make use of [text-generation-inference](https://github.com/huggingface/text-generation-inference). Both methods are ~20x faster than using huggingface transformers.
- **Detailed information about model performance.** By saving not just the final scores but also all of the intermediate results, FastEval enables you to get both [a general overview of model performance](https://fasteval.github.io/FastEval/) but also go deeper and look at the [performance on different categories](https://fasteval.github.io/FastEval/#?benchmark=mt-bench) down to inspecting the [individual model outputs](https://fasteval.github.io/FastEval/#?benchmark=cot&task=bbh/date_understanding&id=eb74c9e1-8836-4c3a-8f50-a25808d20eee).
- **Use of model-specific prompt templates**: Different instruction following models are prompted in different ways, but many other benchmarks & leaderboards ignore this and prompt all of them the same way. FastEval uses the right prompt template depending on the model that is being evaluated. Support is added for various prompt templates and the integration with [Fastchat](https://github.com/lm-sys/FastChat) expands this even further.

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

This already installs [vLLM](https://github.com/vllm-project/vllm) for fast inference which is usually enough [for most models](https://vllm.readthedocs.io/en/latest/models/supported_models.html). However, if you encounter any problems with vLLM or your model is not supported, FastEval also supports using [text-generation-inference](https://github.com/huggingface/text-generation-inference) as an alternative. The performance is very similar to vLLM, but the installation process is more complex and therefore separate. If you would like to use text-generation-inference, you can install it as follows:

```bash
# Install various system packages.
# The following command assumes an ubuntu >= 22.04 system.
apt install protobuf-compiler libssl-dev gcc pkg-config g++ make

# Install rust.
# The rust-all package on ubuntu >= 22.04 might also work, but sometimes a newer version is required.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install text-generation-inference to the `FastEval/text-generation-inference` folder.
./install-text-generation-inference
```

### OpenAI API Key

[MT-Bench](https://arxiv.org/abs/2306.05685) uses GPT-4 as a judge for evaluating model outputs. For this benchmark, you need to configure an OpenAI API key by setting the `OPENAI_API_KEY` environment variable. Note that methods other than setting this environment variable won't work. The cost of evaluating a new model on MT-Bench is approximately $5.

## Evaluation

⚠️ Running `fasteval` currently executes untrusted code, both from models with remote code as well as LLM generated code when using [HumanEval+](https://github.com/evalplus/evalplus). There is currently no integrated sandbox, so make sure to only execute the code in an environment where this is not a problem.

To evaluate a new model, call `fasteval` in the following way:
```
./fasteval [-b <benchmark_name_1>...] -t model_type -m model_name
````

The `-b` flag specifies the benchmark that you want to evaluate your model on. The default is `all`, but you can also specify one or multiple individual benchmarks. Possible values are [`mt-bench`](https://fasteval.github.io/FastEval/#?benchmark=mt-bench), [`human-eval-plus`](https://fasteval.github.io/FastEval/#?benchmark=human-eval-plus), [`cot`](https://fasteval.github.io/FastEval/#?benchmark=cot) or [`lm-evaluation-harness`](https://fasteval.github.io/FastEval/#?benchmark=lm-evaluation-harness).

Type `-t` flag specifies the type of the model which is either the prompt template or the API client that will be used. Supported values are [`openai`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/open_ai.py), [`alpaca-without-prefix`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/alpaca_without_prefix.py), [`alpaca-with-prefix`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/alpaca_with_prefix.py), [`chatml`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/chatml.py), [`guanaco`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/guanaco.py), [`open-assistant`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/open_assistant.py), [`falcon-instruct`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/falcon_instruct.py), [`starchat`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/starchat.py) or [`fastchat`](https://github.com/FastEval/FastEval/blob/main/evaluation/models/fastchat.py).

The `-m` flag specifies the name of the model which can be a path to a model on huggingface, a local folder or an OpenAI model name.

For example, this command will evaluate [`OpenAssistant/pythia-12b-sft-v8-2.5k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-2.5k-steps) on [MT-Bench](https://fasteval.github.io/FastEval/#?benchmark=mt-bench):
```bash
./fasteval -b mt-bench -t open-assistant -m OpenAssistant/pythia-12b-sft-v8-2.5k-steps`
```

### Viewing the results

Use `python3 -m http.server` in the root folder of this repository.
This will start a simple webserver for static files.
This server usually runs on port `8000` in which case you can view the results at [127.0.0.1:8000](http://127.0.0.1:8000).
