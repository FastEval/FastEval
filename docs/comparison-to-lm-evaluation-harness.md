# Comparison to LM-Evaluation-Harness

FastEval is not a fast version of [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness).
The main difference apart from the performance is that FastEval is primarily for instruction-following and chat language models while LM-Eval focuses on base models.
This difference is reflected in multiple design decisions.

## Prompt templates
FastEval uses [model-specific prompt templates](docs/model-type.md) to prompt the model since this is also how chat language models are trained.
LM-Eval does not do this, which also makes sense since base models have not been trained with any sort of prompt template.

## Various types of evaluation
For most tasks, LM-Eval evaluates models using simple text matching like multiple choice.
This approach works fine for testing whether the model generally contains certain types of knowledge and it is therefore a very good choice for evaluating base models.

However, while it is possible to test for various knowledge using these types of tests, it does not tell us much about _in what ways the model is able to use this knowledge_.
Yet this part is what we are usually interested in when considering instruction-following and chat capabilities.

To solve this problem, FastEval uses various forms of evaluations in addition to simple text-based matching.

**Programming capabilities:**
To test for the ability to write code, it is insufficient to do simple text matching.
Instead, execution based evaluation is required.
FastEval currently includes [HumanEval+](https://github.com/evalplus/evalplus) for evaluating simple python coding abilities, though we plan to expand it with additional benchmarks that measure the ability to edit existing code and also contain more languages and more complex problems.

**Conversational abilities:**
It is very hard to measure automatically how well a language model is generally able to handle conversations with a user which is certainly something that should not be missing from a chat language model.
The best approach right now seems to be to use a more powerful language model like GPT-4 to judge the model that should be evaluated.
While this dependency on GPT-4 is quite annoying and also expensive ($5 per model), it is still easier & less expensive than human evaluation, so we include it.
But if there are new methods to evaluate similar capabilities in a better way, we will also add them.

**Multi-step reasoning:**
Most tasks in LM-Eval measure the ability of a model to immediately output the answer in a few tokens.
We believe that this is not a realistic setting for chat language models where the user is often willing to give the language model time to think in order to get back a more accurate answer.
In order to evaluate this ability, we focus our evaluations more on CoT (chain-of-thought) reasoning.
Since giving the answer directly can still be useful in some cases, we also additional include evaluations for this by using LM-Eval itself.
However, it is not the main focus.

**Tool use:**
TODO

## Zero-shot prompting:
LM-Eval is "A framework for few-shot evaluation of autoregressive language models". 
By comparison, FastEval uses zero-shot evaluation almost exclusively since this is mostly also how instruction-following models will be used in practice.
Note that LM-Eval _can_ also use zero-shot prompting and this is also why it is used as a part of FastEval.
However, it's not the main focus and some tasks are written in a way that few-shot prompting is basically required.
