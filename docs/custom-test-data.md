# Custom Test Data

**Note:** Some details about this benchmark are still subject to change which may impact the final scores.

FastEval supports evaluating models on custom test data. To explain how it works and how to use it, let's consider an example.

We will begin with the following file.
It contains a list of conversations between a user and an assistant.
The final conversation item is always a user prompt.
In addition, for every conversation there is also a reference answer.
This data corresponds to the test data that you would like to evaluate the model on.
For every input conversation, there is an approximately expected output answer.

```json
{
    "0": {
        "conversation": [
            ["user", "What's 2 + 2?"]
        ],
        "reference": "2 + 2 = 4"
    },
    "1": {
        "conversation": [
            ["user", "What's the population of antarctica?"],
            ["assistant", "There are about 4,000 people through the summer months and about 1,000 overwinter each year."],
            ["user", "And what about the artic?"]
        ],
        "reference": "The Arctic is home to almost four million people today - Indigenous Peoples, more recent arrivals, hunters and herders living on the land and city dwellers. Roughly 10 percent of the inhabitants are Indigenous and many of their peoples distinct to the Arctic."
    }
}
```

To evaluate a model with this test data, use `./fasteval -t MODEL_TYPE -m MODEL_NAME -b custom-test-data --custom-test-data-file THE_FILE_THAT_CONTAINS_THE_PREVIOUS_DATA.json`.

Now let's see what FastEval does internally.
First, the evaluated model will be used to compute the replies for every conversation.
This will ignore the reference answer in your test data file for now.
The output could be something like the following:

```json
{
    "0": "2 + 2 is 4.",
    "1": "The Arctic, which consists of the Arctic Ocean and the areas surrounding it, does not have a specific population figure. However, the region is home to various indigenous people such as Inuits, S\u00e1mi, and others, and there are also small settlements in countries bordering the Arctic like Russia, Norway, Greenland, and Canada. The total population of the Arctic regions is relatively small in comparison to the rest of the world."
}
```

Next up, a judge model will be used to evaluate how well the model outputs correspond to the reference answers that you specified.
Right now, this judge model is [`Open-Orca/OpenOrcaxOpenChat-Preview2-13B`](https://huggingface.co/Open-Orca/OpenOrcaxOpenChat-Preview2-13B).
While a more powerful model would generally give better judgments, the model doesn't need to be _that_ powerful since it has a reference output to work with.
The judge will be given the conversation as context, the reference output as well as the output of the evaluated model.
It will then produce judgments like the following:

```json
{
    "0": "The assistant's answer is correct and matches the reference answer. The assistant provided the correct answer to the user question, which is 4. Therefore, the final rating is:\n\nRating: [[10]]",
    "1": "The assistant's answer provides a general idea about the population of the Arctic, mentioning that it does not have a specific population figure and that it is home to various indigenous people and small settlements in countries bordering the Arctic. However, the reference answer provides a more specific and accurate number, stating that there are almost four million people living in the Arctic.\n\nThe assistant's answer is not entirely incorrect, but it lacks the specificity and accuracy of the reference answer. Therefore, I would rate the response as:\n\nRating: [[6]]"
}
```

From these judgments, the ratings will be extracted automatically. In this case, the ratings would be `[10, 6]`. The final score then corresponds to the average rating, i.e. `8`. A higher score would indicate more favorable judgments corresponding to a model that answers your questions more like in your test data.

Different from the other benchmarks, custom test data results are currently not shown on the leaderboard. They will only be saved to the `reports/custom-test-data` folder.
