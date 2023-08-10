The following is the output of running `./fasteval -h` to show information about the possible commandline flags.

```
usage: fasteval [-h] [-b [{all,mt-bench,cot,human-eval-plus,lm-evaluation-harness} ...]]
                [-t MODEL_TYPE] [-m MODEL_NAME] [--model-tokenizer MODEL_TOKENIZER]
                [--model-default-system-message MODEL_DEFAULT_SYSTEM_MESSAGE]
                [--force-backend {vllm,tgi,hf_transformers}]
                [--model-force-dtype {float16,bfloat16,float32}]
                [--num-gpus-per-model NUM_GPUS_PER_MODEL] [--run-correctness-check]

options:
  -h, --help            show this help message and exit
  -b [{all,mt-bench,cot,human-eval-plus,lm-evaluation-harness} ...], --benchmarks [{all,mt-bench,cot,human-eval-plus,lm-evaluation-harness} ...]
                        Benchmark(s) that the model will be evaluated on
  -t MODEL_TYPE, --model-type MODEL_TYPE
                        Type of the model that will be evaluated. Can be an API client name
                        (openai), a prompt template (e.g. chatml) or `fastchat` for the fastchat
                        backend.
  -m MODEL_NAME, --model-name MODEL_NAME
                        Name of the model that will be evaluated. Depending on the type, it can be
                        an OpenAI model name or a path to a huggingface model.
  --model-tokenizer MODEL_TOKENIZER
                        By default, the tokenizer will be the same as the model, but it can also be
                        overwritten with this argument.
  --model-default-system-message MODEL_DEFAULT_SYSTEM_MESSAGE
                        The default system message of the model. Only applicable for models that use
                        a system message and only if no other system message has been specified.
  --force-backend {vllm,tgi,hf_transformers}
                        Force a specific backend for model inference. By default, the backend will
                        be selected automatically depending on model support, but if you encounter
                        bugs with this you can overwrite the backend with this argument.
  --model-force-dtype {float16,bfloat16,float32}
                        By default, the dtype of the model will be taken from the model config.json.
                        However, you can overwrite it with this argument.
  --num-gpus-per-model NUM_GPUS_PER_MODEL
                        This argument controls data parallelism. By default, the model will only be
                        instantiated a single time distributed across all GPUs. This works fine if
                        you have one GPU or if you have a big model and two GPUs, but it is not a
                        fast approach if you e.g. have 8 GPUs. In these cases, it is recommended to
                        instantiate the model multiple times on different GPUs and do data parallel
                        evaluation. It is recommended to set --num-gpus-per-model to the number of
                        GPUs that your model will require. For example, if your model requires 2
                        GPUs and you have 8 GPUs, setting "--num-gpus-per-model 2" will create the
                        model 4 times on 2 GPUs each.
  --run-correctness-check
                        Runs a check to make sure that the outputs of the chosen fast inference
                        backend (vLLM or TGI) are equal to those that HF transformers outputs. This
                        is needed because vLLM & TGI sometimes have incorrect implementations or
                        haven't implemented a new feature yet but don't even warn about that.
```
