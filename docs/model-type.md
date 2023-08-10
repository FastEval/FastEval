FastEval uses model-specific prompt templates to prompt the model.
This is different from other tools that prompt all models the same way.
Using model-specific prompt templates makes the evaluation closer to how the models will actually be used in practice.
However, using model-specific prompt templates also means that the corresponding template must be implemented for the model that you want to evaluate.
Luckily, many models share the same prompt template and FastEval implements various of them.
The following is a list of supported templates & other model types in FastEval.
The model type can be specified with the `-t` flag, e.g. `-t alpaca-with-prefix`.

# Prompt templates

## [alpaca-with-prefix](https://github.com/FastEval/FastEval/blob/main/evaluation/models/alpaca_with_prefix.py)

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
[user_input]

### Response:
[assistant_output]
```

Example models that make use of this prompt template:
- [WizardLM/WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)

## [alpaca-without-prefix](https://github.com/FastEval/FastEval/blob/main/evaluation/models/alpaca_without_prefix.py)

```
### Instruction:
[user_input]

### Response:
[assistant_output]
```

Example models that make use of this prompt template:
- [NousResearch/Nous-Hermes-13b](https://huggingface.co/NousResearch/Nous-Hermes-13b)
- [NousResearch/Nous-Hermes-Llama2-13b](https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b)

## [chatml](https://github.com/FastEval/FastEval/blob/main/evaluation/models/chatml.py)

```
<|im_start>system
[system_message]
<|im_end|>
<|im_start|>user
[user_input]
<|im_end|>
<|im_start|>assistant
[assistant_output]
<|im_end|>
```

Example models that make use of this prompt template:
- [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)
- [mosaicml/mpt-30b-chat](https://huggingface.co/mosaicml/mpt-30b-chat)

## [dolphin](https://github.com/FastEval/FastEval/blob/main/evaluation/models/dolphin.py)

```
SYSTEM: [system_message]
USER: [user_input]
ASSISTANT: [assistant_output]
```

Example models that make use of this prompt template:
- [ehartford/dolphin-llama-13b](https://huggingface.co/ehartford/dolphin-llama-13b)

## [falcon-instruct](https://github.com/FastEval/FastEval/blob/main/evaluation/models/falcon_instruct.py)

```
User: [user_input]
Assistant: [assistant_output]
```

Example models that make use of this prompt template:
- [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
- [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)

## [guanaco](https://github.com/FastEval/FastEval/blob/main/evaluation/models/guanaco.py)

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: [user_input]
### Assistant: [assistant_output]
```

Example models that make use of this prompt template:
- [timdettmers/guanaco-65b-merged](https://huggingface.co/timdettmers/guanaco-65b-merged)

## [llama2-chat](https://github.com/FastEval/FastEval/blob/main/evaluation/models/llama2_chat.py)

```
[INST] <<SYS>>
[system_message]
<</SYS>>

[user_input] [/INST] [assistant_output]
```

Example models that make use of this prompt template:
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

## [open-assistant](https://github.com/FastEval/FastEval/blob/main/evaluation/models/open_assistant.py)

```
<|prompter|>[user_input]<|endoftext|><|assistant|>[assistant_output]<|endoftext|>
```

For llama models, the `<|endoftext|>` is replaced with `</s>` instead.

Example models that make use of this prompt template:
- [OpenAssistant/oasst-sft-1-pythia-12b](https://huggingface.co/OpenAssistant/oasst-sft-1-pythia-12b)
- [OpenAssistant/pythia-12b-sft-v8-7k-steps](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-7k-steps)
- [OpenAssistant/oasst-sft-7-llama-30b](https://huggingface.co/OpenAssistant/oasst-sft-7-llama-30b-xor)
- [OpenAssistant/falcon-40b-sft-top1-560](https://huggingface.co/OpenAssistant/falcon-40b-sft-top1-560)
- [OpenAssistant/falcon-40b-sft-mix-1226](https://huggingface.co/OpenAssistant/falcon-40b-sft-mix-1226)

## [openchat-llama2-v1](https://github.com/FastEval/FastEval/blob/main/evaluation/models/openchat_llama2_v1.py)

```
User: [user_input]<|end_of_turn|>Assistant: [assistant_output]<|end_of_turn|>
```

Example models that make use of this prompt template:
- [Open-Orca/OpenOrcaxOpenChat-Preview2-13B](https://huggingface.co/Open-Orca/OpenOrcaxOpenChat-Preview2-13B)

## [stable-beluga](https://github.com/FastEval/FastEval/blob/main/evaluation/models/stable_beluga.py)

```
### System:
[system_message]

### User:
[user_input]

### Assistant:
[assistant_output]
```

Example models that make use of this prompt template:
- [stabilityai/StableBeluga-13B](https://huggingface.co/stabilityai/StableBeluga-13B)
- [stabilityai/StableBeluga2](https://huggingface.co/stabilityai/StableBeluga2)

## [starchat](https://github.com/FastEval/FastEval/blob/main/evaluation/models/starchat.py)

```
<|system|>
[system_message]<|end|>
<user>
[user_input]<|end|>
<|assistant|>
[assistant_output]<|end|>
```

# Other possible values

## [fastchat](https://github.com/FastEval/FastEval/blob/main/evaluation/models/fastchat.py)

FastEval can also use fastchat as a backend for inference.
In this case, the conversation is directly sent to fastchat which will then actually convert it to a prompt according to its own logic and run it through the model.

Note that while using this model type is easy, it will have cons:
- Reproducability will be worse because fastchat changes some prompt templates from time to time.
- No data parallel evaluation is currently implemented for this backend.
- Only vLLM can be used for fast evaluation. Using text-generation-inference is not supported.

Example models that make use of this model type:
- [lmsys/vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)
- [lmsys/vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3)

## [openai](https://github.com/FastEval/FastEval/blob/main/evaluation/models/open_ai.py)

If you choose this model type, then the model name will refer to an OpenAI model instead of a huggingface model or a local path.

Example models that make use of this model type:
- gpt-3.5-turbo-0301
- gpt-3.5-turbo-0613
- gpt-4-0613
