FastEval is not a fast version of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
The main difference apart from the performance is that FastEval focuses on evaluating instruction-following language models while lm-evaluation-harness focuses on evaluation of base models.
This difference is reflected in multiple design decisions.

**Use of model-specific prompt templates:**
FastEval uses [model-specific prompt templates](docs/model-type.md) to prompt the model since this is also how instruction-following language models are trained.
LM-Eval does not do this, which also makes sense since base models have not been trained with any sort of prompt template.

**Log probabilities vs. long text generation:**
For the majority of tasks, LM-Eval measures the (log) probabilities of a set of considered output sequences and judges the model answer to be correct if the highest probability sequence corresponds to the correct one.
This mostly works fine for short output sequences like multiple-choice knowledge.
However, it is not a good method for evaluating things like conversational, programming, multi-step reasoning and tool use abilities.
FastEval uses long-text generation and various forms of feedback like execution-feedback for programming, GPT-4 feedback for conversations, checking the final answer for multi-step reasoning.

**Few-shot vs. zero-shot prompting:**
LM-Eval is "A framework for few-shot evaluation of autoregressive language models". 
By comparison, FastEval uses zero-shot evaluation almost exclusively since this is mostly also how instruction-following models will be used in practice.
Note that LM-Eval _can_ also use zero-shot prompting and this is also why it is used as a part of FastEval.
However, it's not the main focus and some tasks are written in a way that few-shot prompting is basically required.
