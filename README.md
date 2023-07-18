# ILM-Eval

This repository contains code to automatically evaluate instruction following language models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the [leaderboard to view those reports](https://tju01.github.io/ilm-eval/).

## Features

- **Evaluation on various benchmarks with a single command.** Supported benchmarks are [OpenAI Evals](https://github.com/openai/evals) for general performance, [MT‑Bench](https://arxiv.org/abs/2306.05685) for conversational capabilities, [HumanEval+](https://github.com/evalplus/evalplus) for Python coding performance, Chain of Thought (GSM8K + BBH + MMLU) for reasoning capabilities and [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness) for another method of evaluating general capabilities.
- **High performance.** ILM-Eval uses [vLLM](https://github.com/vllm-project/vllm) for inference by default and can also optionally make use of [text-generation-inference](https://github.com/huggingface/text-generation-inference). Both of them are 20x faster than using huggingface transformers.
- **Detailed information about model performance.** By saving not just the final scores but also all of the intermediate results, ILM-Eval enables you to get both [a general overview of model performance](https://tju01.github.io/ilm-eval/) but also go deeper and look at the [performance on different categories](https://tju01.github.io/ilm-eval/#?benchmark=mt-bench) down to inspecting the [individual model outputs on questions](https://tju01.github.io/ilm-eval/#?benchmark=cot&task=bbh/date_understanding&model=mosaicml/mpt-30b-chat).
- **Use of model-specific prompt templates**: Different instruction following models are prompted in different ways, but many other benchmarks & leaderboards ignore this and prompt all of them the same way. ILM-Eval uses the right prompt template depending on the model that is being evaluated. Support is added for various prompt templates and the integration with [Fastchat](https://github.com/lm-sys/FastChat) expands this even further.

## Supported benchmarks

Right now, the following benchmarks are supported:
- [OpenAI evals](https://github.com/openai/evals): Contains various tasks to measure different capabilities of instruction-following language models. Uses both basic tasks that are just compared to the solution directly and model-graded tasks where another language model is used for evaluation.
- [MT-Bench](https://arxiv.org/abs/2306.05685): Uses GPT-4 to score the model outputs on a set of 80 questions for two conversation turns each, i.e. 160 GPT-4 judgments in total.
- [HumanEval+](https://github.com/evalplus/evalplus): Evaluates python coding performance. The model is given the start of a function as input with a docstring comment on what the function is supposed to do. The model then completes the code which is then evaluated for correctness by running it against a few tests.
- CoT: Evaluates chain-of-thought reasoning capabilities of the model. It prompts the model to respond to a set of questions step-by-step. Currently combines [GSM8K](https://github.com/openai/grade-school-math), [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) and [MMLU](https://arxiv.org/abs/2009.03300).
- [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness): Different from all the other benchmarks, this more classical benchmark does not take the model-specific prompt format into account. However, despite this fact, LM-Eval is popular for evaluating instruction following language models. It is therefore part of the evaluation here together with other benchmarks that take the model-specific prompt format into account.

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

<details>
<summary>Install text-generation-inference for ~20x faster performance for some models like Falcon and StarCoder</summary>

By default, ilm-eval tries to use [vLLM](https://github.com/vllm-project/vllm) to do fast inference. When supported, this is a lot (~20x) faster than using huggingface transformers. However, vLLM does not support all models. An alternative to vLLM with similar performance is [text-generation-inference](https://github.com/huggingface/text-generation-inference). While it also doesn't support all models either, it can serve as a useful addition to vLLM and together they support many models.

While vLLM is part of the `requirements.txt` and therefore already installed if you followed the above installation instructions, installing text-generation-inference requires additional steps. If all the models you need are supported in vLLM, then you don't need to follow these instructions. If some model (e.g. Falcon, StarCoder) is not supported in vLLM, then it's probably worth setting up text-generation-inference:

```bash
# Install various system packages. The following code assumes an ubuntu system.
apt install rust-all protobuf-compiler libssl-dev gcc pkg-config g++ make python3.10-dev

# Install text-generation-inference to the `ilm-eval/text-generation-inference` folder.
./install-text-generation-inference
```
</details>

### OpenAI API

Some benchmarks use `gpt-3.5-turbo` as a model to judge the output of another model. This is the case for `OpenAI evals` as well as for `Vicuna Rank`.
For these benchmarks (and also for evaluating `gpt-3.5-turbo` itself), you need to configure an OpenAI API key by setting the `OPENAI_API_KEY` environment variable.
Make sure to set this environment variable, other methods of configuring the API key won't work.
The cost of evaluating `gpt-3.5-turbo` or using it for benchmarks on another model is something like $5.

### Prompt format

Since this repository is about instruction following models and different instruction models require different prompt formatting, a corresponding implementation of the prompt format or an API client is needed to evaluate a model. Currently, support is added for the following model types: [OpenAI](https://platform.openai.com/docs/models), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) (also used by many other models like [`NousResearch/Nous-Hermes-13b`](https://huggingface.co/NousResearch/Nous-Hermes-13b)), [ChatML](https://github.com/openai/openai-python/blob/main/chatml.md), [Guanaco](https://huggingface.co/timdettmers/guanaco-65b-merged), [Open-Assistant](https://open-assistant.io), [Falcon Instruct](https://huggingface.co/tiiuae) and [Starchat](https://huggingface.co/HuggingFaceH4/starchat-beta). In addition, models that are supported in [Fastchat](https://github.com/lm-sys/FastChat) can also be used.

### Evaluation

⚠️ Running `evaluate.py` currently executes untrusted code (from models with remote code, LLM generated code when using HumanEval+). There is currently no sandbox, so make sure to only execute the code in a reasonable isolated environment.

Call the `evaluate.py` script in the following way:
```
./evaluate.py [-b <benchmark_name_1>...] -m model_type_1:model_name_1...
````
- A benchmark name can be `all` (default), `openai-evals`, `vicuna`, `mt-bench`, `human-eval-plus`, `cot` or `lm-evaluation-harness`.
- A model type can be either [`openai`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/open_ai.py), [`alpaca-without-prefix`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/alpaca_without_prefix.py), [`alpaca-with-prefix`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/alpaca_with_prefix.py), [`chatml`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/chatml.py), [`guanaco`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/guanaco.py), [`open-assistant`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/open_assistant.py) or [`falcon-instruct`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/falcon_instruct.py), [`starchat`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/starchat.py) or [`fastchat`](https://github.com/tju01/ilm-eval/blob/main/evaluation/models/fastchat.py).
- A model name can be either a path to a local folder or a huggingface path.

For example, use the following command to evaluate the [`pythia-12b-sft-v8-2.5k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-2.5k-steps) model from Open-Assistant on the OpenAI evals benchmark:
```
./evaluate.py -b openai-evals -m open-assistant:OpenAssistant/pythia-12b-sft-v8-2.5k-steps
```
This will generate an evaluation report in the `reports/openai-evals/OpenAssistant--pythia-12b-sft-v8-7k-steps` folder.
If the report already exists, then the evaluation is skipped.

## Viewing the reports

Use `python3 -m http.server` in the root of this repository.
This will start a simple webserver for static files.
The webserver usually runs on port `8000`, so you can go to http://127.0.0.1:8000/ and view the results.

## Contact

If you encounter some problems with this code or if you have any questions or suggestions, you can either create a github issue or ping me on discord. My username is [tju01](https://discord.com/users/1090923181910532167) and you can find me on many of the open-source LLM discord servers.
