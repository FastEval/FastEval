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

However, while these tests tell us about the existence of certain knowledge, they does not tell us much about _in what ways the model is able to use this knowledge_.
Yet this part certainly matters for instruction-following and chat language models.

To solve this problem, FastEval uses various forms of evaluations.
Every method tells us whether the model is able to _use_ the knowledge it contains in specific ways.

### Programming capabilities
To measure the programming abilities of a language model, it is insuffient to do some simple text matching against a ground truth solution.
Instead, the usual method is to let the model write new or modify existing code and then execute the resulting code against a number of tests to evaluate the correctness.

FastEval currently includes [HumanEval+](https://github.com/evalplus/evalplus) for evaluating simple python coding abilities.
However, we plan to expand it with additional benchmarks that measure the ability to edit existing code and also contain more languages and more complex problems.

### Conversational abilities
It is very hard to measure automatically how well a language model is generally able to handle conversations with a user.
Yet, this is certainly something that should not be missing from a chat language model.

The best approach right now seems to be to use a more powerful language model like GPT-4 to judge the model that should be evaluated.
This dependency on GPT-4 is of course quite annoying and also expensive ($5 per model).
However, since it is still easier and less expensive than human evaluation, we include it as a benchmark.

### Multi-step reasoning
Most tasks in LM-Eval measure the ability of a model to immediately output the answer in a few tokens.
This is not a realistic setting for chat language models where the user is often willing to give the language model time to think in order to get back a more accurate answer.

To make evaluations closer to this setting used in practice, FastEval focuses on CoT (chain-of-thought) reasoning.
However, sometimes giving the answer immediately can still be useful, so we also additional parts of LM-Eval itself for this.

### Tool use & acting as agent
FastEval currently does not include benchmarks for measuring the ability to use tools and act as an agent.
However, it will contain them in the future and it shows that the focus is quite different from LM-Eval.

## Zero-shot prompting:
LM-Eval is very often used with few-shot prompting.
While this is again no problem for base models, it is an unrealistic setting for chat models since most users will use the model directly in a zero-shot setting without formulating a few-shot prompt.

For this reason, FastEval focuses almost exclusively on the zero-shot setting.
It also includes some tasks from LM-Eval, but in the zero-shot CoT setting which is more realistic for chat language models.

Note that while LM-Eval can also be used with zero-shot, few-shot is almost a requirement for tasks where a specific output format is expected.
This is because otherwise the model will often answer in an unexpected format and fail the tests due to this reason instead of actually getting the answer wrong.

Some tasks from LM-Eval can still be useful for chat language models despite all of the reasons above which is also why FastEval includes some tasks from LM-Eval as part.
