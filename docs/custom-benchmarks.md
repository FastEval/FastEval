FastEval supports evaluating a new model on custom test data.
This can be used both if you are looking for a good model on your own test data as well as if you are a model developer looking to improve your model on some custom data.

The way that custom evaluations work in FastEval is by using a judge model to judge how close the model output corresponds to a reference solution.
To explain how this works, let's consider an example and see how FastEval uses it.
We will begin with the following file that contains some conversations that end with a user question.
In addition, every entry also contains a reference output which is the kind of output that you would like the model to respond with.

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

To evaluate a model with this test data, you can then use the following command:

```bash
./fasteval -b custom --custom-benchmark-data-file <THE_FILE_THAT_CONTAINS_THE_PREVIOUS_DATA.json> -t MODEL_TYPE -m MODEL_NAME 
```

Now let's see what FastEval does internally.
First, the model answers for every conversation will be computed.
This will ignore the reference answer in your file for now.
The output could be something like the following:

```json
{
    "0": "2 + 2 is 4.",
    "1": "The Arctic, which consists of the Arctic Ocean and the areas surrounding it, does not have a specific population figure. However, the region is home to various indigenous people such as Inuits, S\u00e1mi, and others, and there are also small settlements in countries bordering the Arctic like Russia, Norway, Greenland, and Canada. The total population of the Arctic regions is relatively small in comparison to the rest of the world."
}
```

Next up, a judge model will be used to evaluate how well the model outputs correspond to the reference answers that you specified.
Right now, this judge model is `Open-Orca/OpenOrcaxOpenChat-Preview2-13B`.
While a more powerful model would generally give better judgments, the model doesn't need to be _that_ powerful since since it has a reference solution to work with.
The judge will be given the conversation as context, the model outputs as well as the reference outputs.
It will then produce judgments like the following:

```json
{
    "0": "The assistant's answer is correct and matches the reference answer. The assistant provided the correct answer to the user question, which is 4. Therefore, the final rating is:\n\nRating: [[10]]",
    "1": "The assistant's answer provides a general idea about the population of the Arctic, mentioning that it does not have a specific population figure and that it is home to various indigenous people and small settlements in countries bordering the Arctic. However, the reference answer provides a more specific and accurate number, stating that there are almost four million people living in the Arctic.\n\nThe assistant's answer is not entirely incorrect, but it lacks the specificity and accuracy of the reference answer. Therefore, I would rate the response as:\n\nRating: [[6]]"
}
```

From these, the rating will be extracted automatically and a final average rating will be computed. The final average rating would in this case be 8.
