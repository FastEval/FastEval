# Instruction following language model evaluation

This repository contains code to automatically evaluate instruction following language models on benchmarks.
It also contains the evaluation reports for different models as well as the code for the [leaderboard to view those reports](https://tju01.github.io/ilm-eval/).

## Comparison to other leaderboards

There are a number of other leaderboards for LLMs. Here is a comparison of how they compare to this repository:
<details>
<summary>LMSys Leaderboard: https://chat.lmsys.org/?leaderboard</summary>

- Both leaderboards have a Vicuna ranking as part of the benchmarks, though they are computed differently. The LMSys leaderboard uses human ratings while this repository uses GPT-3.5. Good human ratings are generally better, but ilm-eval doesn't focus on them in order to make it easier to add a new model.
- The LMSys leaderboard also includes [MT-Bench](https://arxiv.org/abs/2306.05685), which uses GPT-4 for rating the two-turn conversational capabilities. This is currently not included in ilm-eval, though it is planned.
- This repository (ilm-eval) includes [HumanEval+](https://github.com/evalplus/evalplus) and CoT for evaluating more complex reasoning capabilities. The Vicuna rank does not reflect these and tends to overestimate less capable models. MT-Bench improves on that by using a reference-guided judge, but it's still very limited when it comes to complex reasoning.
- For evaluating more general capabilities, the LMSys leaderboard uses 5-shot MMLU. This repository (ilm-eval) also contains zero-shot MMLU as a part of CoT evaluation, but it additionally also uses a combination of tasks from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and OpenAI evals which also evaluate general capabilities.
- In ilm-eval, for all benchmarks except lm-evaluation-harness, the model outputs are also stored in addition to the scores. These model outputs can also be viewed [on the website](https://tju01.github.io/ilm-eval/) and can sometimes be useful as a quality indicator in addition to the resulting scores.
</details>

<details>
<summary>InstructEval Leaderboard: https://github.com/declare-lab/instruct-eval</summary>

- Both repositories focus on evaluating instruction following LLMs.
- However, InstructEval uses 3-shot for some of the benchmarks and 0-shot for some others. Even in the cases where 0-shot is used, the model-specific prompt format is never used. By comparison, ilm-eval focuses _only_ on 0-shot evaluation and uses the model-specific prompt format in most cases (except one) because this is how the models will be used in the end.
- This repository (ilm-eval) stores the model outputs for most of the benchmarks. They can be viewed [on the website](https://tju01.github.io/ilm-eval/) and can sometimes be useful in addition to just the benchmark scores.
- The benchmarks that are used are different.
- The InstructEval leaderboard currently contains way more models.
</details>

<details>
<summary>HuggingFace Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard</summary>

- HF Open LLM Leaderboard is not specifically focused on instruction following language models. The main thing that matters for instruction following LLMs is 0-shot performance and the only task that is evaluated with 0-shot there is TruthfulQA which is very limited.
- More recently, HF Open LLM Leaderboard added human & GPT-4 evaluations which _does_ evaluate the instruction following capabilities. The GPT-4 evaluation is esentially what [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) introduced. This repository (ilm-eval) also contains this vicuna benchmark, though currently only with GPT-3.5.
- However, ilm-eval also contains other benchmarks like [OpenAI evals](https://github.com/openai/evals) and [HumanEval+](https://github.com/evalplus/evalplus) which HF Open LLM Leaderboard doesn't contain. This repository also uses lm-evaluation-harness, but in a different way to focus only on 0-shot performance.
- While one part of their leaderboard uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the evaluation [seems to be straightforward](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/60), the GPT-4 evaluation code doesn't seem to be open source at the moment.
- The model outputs are not stored on the HF Open LLM Leaderboard. By comparison, ilm-eval also stores model outputs for most benchmarks (except lm-evaluation-harness) and they can be viewed [on the website](https://tju01.github.io/ilm-eval/) in addition to just the resulting scores.
- The HF Open LLM Leaderboard contains way more models, but less benchmarks.
</details>

<details>
<summary>AlpacaEval Leaderboard: https://tatsu-lab.github.io/alpaca_eval/</summary>

- This leaderboard is limited to automatic evaluation using GPT-4 and Anthropic Claude. This kind of evaluation has been shown to be subject to very simple biases (e.g. simply preferring longer answers) and it tends to overestimate the capabilities of smaller models.
- Nevertheless, this kind of benchmark can be _part_ of a useful evaluation. This is also why ilm-eval also contains this type of benchmark, but combines it with other benchmarks.
- On the [ilm-eval website](https://tju01.github.io/ilm-eval/), in addition to the resulting Vicuna Rankings, the model outputs can also be viewed and one can filter for things like only viewing the prompts where one specific model won against another specific model.
- The AlpacaEval Leaderboard currently contains way more models.
</details>

<details>
<summary>GPT4All Leaderboard: https://gpt4all.io/index.html (scroll down to section "Performance Benchmarks")</summary>

- This leaderboard is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- It therefore does not use the model-specific prompt format that the models have been finetuned with.
- It is also limited to benchmarks where the solution can be checked in a simple way (e.g. exact match or some simple post-processing). It does not use another LLM to evaluate the model and it does not use programmatic benchmarks.
- The results can still be useful, but they should not be everything. This is why ilm-eval also uses lm-evaluation-harness in the exact same way so that the results are comparable, but combines the results with other benchmarks that use the model-specific prompt format and either use another model for evaluation (like the [Vicuna Rank](https://lmsys.org/blog/2023-03-30-vicuna) or some parts of [OpenAI evals](https://github.com/openai/evals)) or are programmatic like [HumanEval+](https://github.com/evalplus/evalplus).
- In addition, for these other benchmarks, ilm-eval also stores the model outputs so that they can be viewed [on the website](https://tju01.github.io/ilm-eval/) which can sometimes be useful in addition to the resulting benchmark scores.
- The GPT4All Leaderboard currently contains way more models.
</details>

## Supported benchmarks

Right now, the following benchmarks are supported:
- [OpenAI evals](https://github.com/openai/evals): Contains various tasks to measure different capabilities of instruction-following language models. Uses both basic tasks that are just compared to the solution directly and model-graded tasks where another language model is used for evaluation.
- [Vicuna Ranking](https://lmsys.org/blog/2023-03-30-vicuna): Uses another more capable model like `gpt-4` or `gpt-3.5-turbo` (used here) for comparing outputs of different models and computes win rates and rankings based on these comparisons.
- [MT-Bench](https://arxiv.org/abs/2306.05685): Similar to the Vicuna ranking, this benchmark uses another more capable model like `gpt-4` or `gpt-3.5-turbo` (used here) to rate the models. However, instead of comparing the models against each other in pairwise battles, it directly assigns scores to model outputs. This makes it more scalable to more models compared to the Vicuna ranking which requires a lot of comparisons between model pairs.
- [HumanEval+](https://github.com/evalplus/evalplus): Gives the model the start of a function as input with a docstring comment on what the function is supposed to do. The model should then complete the code. The model output code is evaluated for correctness by running it against a few tests.
- CoT: This focuses specifically on the chain-of-thought capabilities of the model. It prompts the model to respond to a set of questions step-by-step. Currently it only combines [GSM8K](https://github.com/openai/grade-school-math), [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) and [MMLU](https://arxiv.org/abs/2009.03300) though that may be expanded in the future.
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness): This is the only benchmark that does not take the prompt format into account. However, despite this fact, lm-evaluation-harness is popular for evaluating instruction following language models. It is therefore part of the evaluation here together with other benchmarks that take the prompt format into account.

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
<summary>Install text-generation-inference for greatly improved performance for some models (e.g. Falcon, StarCoder)</summary>

By default, ilm-eval tries to use [vLLM](https://github.com/vllm-project/vllm) to do fast inference. When supported, this is a lot (~20x) faster than using huggingface transformers. However, vLLM does not support all models. An alternative to vLLM with similar performance is [text-generation-inference](https://github.com/huggingface/text-generation-inference). While it also doesn't support all models either, it can serve as a useful addition to vLLM and together they support many models.

While vLLM is part of the `requirements.txt` and therefore already installed if you followed the above installation instructions, installing text-generation-inference is more complex. If all the models you need are supported in vLLM, then you don't need to follow these instructions. If some model (e.g. Falcon, StarCoder) is not supported in vLLM, then it's probably worth setting up text-generation-inference:

```bash
# Install `rust`, `protobuf-compiler`, `libssl-dev`, `gcc` and `pkg-config`.
# The following code assumes an ubuntu system.
apt install rust-all protobuf-compiler libssl-dev gcc pkg-config # libprotobuf-dev

# Install `text-generation-inference`
# Run this code inside the `ilm-eval` folder
# I.e. the `git clone` should happen to `ilm-eval/text-generation-inference`.
git clone --depth 1 https://github.com/huggingface/text-generation-inference.git
cd text-generation-inference

# We will use *a second virtual environment* specifically for `text-generation-inference`.
# This is to avoid package conflicts.
python3.10 -m venv .venv
source .venv/bin/activate

BUILD_EXTENSIONS=True make install

cd server
# This will take a long time
make install-flash-attention
# Installs *a part* of vLLM used in text-generation-inference.
# This is not the full vLLM that we already installed.
make install-vllm
cd ..

# Return to the virtual environment for `ilm-eval`.
deactivate
cd ..
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
