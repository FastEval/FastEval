# Comparison to LM-Evaluation-Harness

FastEval is not a fast version of [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness).
The main difference apart from the performance is that FastEval is primarily designed for instruction-following and chat language models while LM-Eval focuses on base models.
This difference is reflected in multiple design decisions.

## Prompt templates
FastEval uses [model-specific prompt templates](/docs/model-type.md) to prompt the model since this is also how chat language models are trained.
LM-Eval does not do this, which also makes sense since base models have not been trained with any sort of prompt template.

While it is _possible_ to evaluate chat language models without their corresponding prompt template, it is not a _realistic_ setting.
To provide evaluation that is as close as possible to how the chat models will be used in practice, using prompt templates is a requirement.

## Various evaluation methods
For most tasks, LM-Eval evaluates models using a simple comparison with a ground truth, e.g. multiple choice.
This approach works fine for testing whether the model generally contains certain types of knowledge and it is therefore a very good method for evaluating base models.

However, while these tests tell us about the existence of certain knowledge, they do not tell us much about _in what ways the model is able to use this knowledge_.
Yet this part certainly matters for instruction-following and chat language models.

FastEval uses multiple methods to make sure that these abilities are also measured.
The evaluation is therefore more realistic and closer to how the models are used in the end.
Every method tells us whether the model is able to use its knowledge in specific ways.

### Programming capabilities
To measure the programming abilities of a language model, simple matching against a ground truth is insufficient.
Instead, the common approach is to make the model work with code and then _execute_ the resulting code against a number of tests.

FastEval currently includes [HumanEval+](https://fasteval.github.io/FastEval/#?benchmark=human-eval-plus) for evaluating simple python coding abilities.
However, we plan to expand it with additional benchmarks with a focus on code editing, multiple programming languages and more complex problems.

### Conversational abilities
Simple matching against a ground truth solution does not measure well how a language model is able to handle conversations with a user.
However, this ability is certainly something that should not be missing from a chat language model.

The current approach to measure this is to use GPT-4 to judge the model outputs.
This dependency on GPT-4 is quite annoying and also costs $5 per model for [MT-Bench](https://fasteval.github.io/FastEval/#?benchmark=mt-bench).
Still, it is easier than human evaluation and better than other methods.

### Multi-step reasoning
Most tasks in LM-Eval measure the ability of a model to immediately output the answer in a few tokens.
This is not a realistic setting for chat language models where the user is often willing to give the language model time to think in order to obtain a more accurate answer.

To make evaluations closer to this setting used in practice, FastEval focuses more on [CoT (chain-of-thought) reasoning](https://fasteval.github.io/FastEval/#?benchmark=cot).
However, sometimes giving the answer immediately can still be useful, so we also include [some tasks from LM-Eval itself](https://fasteval.github.io/FastEval/#?benchmark=lm-evaluation-harness) for this.

### Tool use & acting as agent
The ability to use tools as well as act as an agent is becoming more and more useful as the general capabilities of the models increase.
Again, to provide a realistic evaluation setting, simple matching against a ground-truth solution is insufficient and actual execution is required.

FastEval currently does not include benchmarks for measuring these capabilities.
However, it is work in progress and it is another example of the very different focus compared to LM-Eval which mostly does simple matching against a ground-truth solution.

## Zero-shot prompting
LM-Eval is very often used with few-shot prompting.
While this is a good approach for base models, it is an unrealistic setting for chat models.
Most users want to use the model directly in a zero-shot setting without formulating a few-shot prompt.

FastEval focuses almost exclusively on the zero-shot setting.
Note that this is also possible with LM-Eval.
However, it won't work well for tasks that require the model to respond in a specific format.
FastEval deals with this problem using additional answer extraction code.
