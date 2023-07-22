# ILM-Eval

This project allows you to automatically evaluate instruction following language models on a number of benchmarks.
It also contains [a leaderboard](https://tju01.github.io/ilm-eval) for comparison between different models.

## Features

- **Evaluation on various benchmarks with a single command.** Supported benchmarks are [MT‑Bench](https://arxiv.org/abs/2306.05685) for conversational capabilities, [HumanEval+](https://github.com/evalplus/evalplus) for Python coding performance, Chain of Thought (GSM8K + BBH + MMLU) for reasoning capabilities and [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness) for general capabilities.
- **High performance.** ILM-Eval uses [vLLM](https://github.com/vllm-project/vllm) for fast inference by default and can also optionally make use of [text-generation-inference](https://github.com/huggingface/text-generation-inference). Both methods are ~20x faster than using huggingface transformers.
- **Detailed information about model performance.** By saving not just the final scores but also all of the intermediate results, ILM-Eval enables you to get both [a general overview of model performance](https://tju01.github.io/ilm-eval/) but also go deeper and look at the [performance on different categories](https://tju01.github.io/ilm-eval/#?benchmark=mt-bench) down to inspecting the [individual model outputs](https://tju01.github.io/ilm-eval/#?benchmark=cot&task=bbh/date_understanding&model=mosaicml/mpt-30b-chat).
- **Use of model-specific prompt templates**: Different instruction following models are prompted in different ways, but many other benchmarks & leaderboards ignore this and prompt all of them the same way. ILM-Eval uses the right prompt template depending on the model that is being evaluated. Support is added for various prompt templates and the integration with [Fastchat](https://github.com/lm-sys/FastChat) expands this even further.

## Installation

```bash
# Install `python3.10` and `python3.10-venv`.
# The following command assumes an ubuntu >= 22.04 system.
apt install python3.10 python3.10-venv

# Clone this repository, make it the current working directory
git clone --depth 1 https://github.com/tju01/ilm-eval.git
cd ilm-eval

# Set up the virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This already installs [vLLM](https://github.com/vllm-project/vllm) for fast inference. However, since vLLM lacks support for some models, ILM-Eval can also use [text-generation-inference](https://github.com/huggingface/text-generation-inference). For some models that are not supported by vLLM (Falcon, StarCoder) it is therefore strongly recommended to install text-generation-inference for ~20x faster inference for these models:

```bash
# Install various system packages.
# The following command assumes an ubuntu >= 22.04 system.
apt install rust-all protobuf-compiler libssl-dev gcc pkg-config g++ make python3.10-dev

# Install text-generation-inference to the `ilm-eval/text-generation-inference` folder.
./install-text-generation-inference
```

### OpenAI API Key

[MT-Bench](https://arxiv.org/abs/2306.05685) uses GPT-4 as a judge for evaluating model outputs. For this benchmark, you need to configure an OpenAI API key by setting the `OPENAI_API_KEY` environment variable. Note that methods other than setting this environment variable won't work. The cost of evaluating a new model on MT-Bench is approximately $5.

## Evaluation

⚠️ Running `evaluate.py` currently executes untrusted code, both from models with remote code as well as LLM generated code when using [HumanEval+](https://github.com/evalplus/evalplus). There is currently no integrated sandbox, so make sure to only execute the code in an environment where this is not a problem.

To evaluate a new model, call the `evaluate.py` script in the following way:
```
./evaluate.py [-b <benchmark_name_1>...] -m model_type_1:model_name_1...
````

The `-b` flag specifies the benchmark that you want to evaluate your model on. The default is `all`, but you can also specify one or multiple individual benchmarks. Possible values are [`mt-bench`](https://tju01.github.io/ilm-eval/#?benchmark=mt-bench), [`human-eval-plus`](https://tju01.github.io/ilm-eval/#?benchmark=human-eval-plus), [`cot`](https://tju01.github.io/ilm-eval/#?benchmark=cot) or [`lm-evaluation-harness`](https://tju01.github.io/ilm-eval/#?benchmark=lm-evaluation-harness).

The `-m` flag specifies the model type and the name of the model. The model type is either the prompt template or the API client that will be used. Supported values are [`openai`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/open_ai.py), [`alpaca-without-prefix`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/alpaca_without_prefix.py), [`alpaca-with-prefix`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/alpaca_with_prefix.py), [`chatml`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/chatml.py), [`guanaco`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/guanaco.py), [`open-assistant`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/open_assistant.py), [`falcon-instruct`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/falcon_instruct.py), [`starchat`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/starchat.py) or [`fastchat`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/fastchat.py).

For example, this command will evaluate [`OpenAssistant/pythia-12b-sft-v8-2.5k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-2.5k-steps) on [MT-Bench](https://tju01.github.io/ilm-eval/#?benchmark=mt-bench):
```bash
./evaluate.py -b mt-bench -m open-assistant:OpenAssistant/pythia-12b-sft-v8-2.5k-steps`
```

### Viewing the results

Use `python3 -m http.server` in the root folder of this repository.
This will start a simple webserver for static files.
This server usually runs on port `8000` in which case you can view the results at [127.0.0.1:8000](http://127.0.0.1:8000).

## Contact

If you encounter some problems with this code or if you have any questions or suggestions, you can either [create a github issue](https://github.com/tju01/ilm-eval/issues/new) or [ping me on discord](https://discord.com/users/1090923181910532167). You can find me as [`tju01`](https://discord.com/users/1090923181910532167) on many LLM-related servers.
