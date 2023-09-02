import copy

from evaluation.models.models import compute_model_replies, create_model, unload_model
from evaluation.utils import join_threads


async def run_inference_backend_correctness_check(model_type, model_name, model_args):
    assert model_type is not None and model_name is not None

    # Idk. Just some conversations. Can be absolutely changed to better reflect what would be good tests
    # to confirm that different inference backends output the same text.
    conversations = [
        [("user", "Please tell me a joke.")],
        [("user", "What's 46912984610 + 5610927469123 * 64127401823 / 123?")],
        [
            ("user", "Why did the chicken cross the road?"),
            (
                "assistant",
                'The classic joke answer to "Why did the chicken cross the road?" is simply, "To get to the other side." '
                "The humor lies in the fact that the listener expects a more elaborate or clever answer, "
                "but the punchline is straightforward and unexpected, resulting in a lighthearted and silly joke. "
                "The joke has become a well-known and often-used example of anti-humor, "
                "where the humor comes from the lack of a complex or unexpected punchline.",
            ),
            ("user", "Can you write a long essay about this topic?"),
        ],
        [
            (
                "user",
                "^_^ </s> <|endoftext|> [PAD] [PAD][PAD] What do you think? Please answer only in emojis. ^_^",
            )
        ],
    ]

    conversations = [
        {
            "conversation": conversation,
            "temperature": 0,
        }
        for conversation in conversations
    ]

    async def get_outputs(model_args, conversation, n):
        return compute_model_replies(
            await create_model(model_type, model_name, model_args, max_new_tokens=1024),
            [conversation] * n,
            progress_bar_description=model_name
            + " :: Computing replies with "
            + model_args["inference_backend"]
            + " backend",
        )

    model_args_with_hf_transformers_backend = copy.deepcopy(model_args)
    model_args_with_hf_transformers_backend["inference_backend"] = "hf_transformers"
    hf_transformers_model_outputs = [
        await get_outputs(model_args_with_hf_transformers_backend, conversation, 1)[0]
        for conversation in conversations
    ]

    # Both vLLM as well as TGI may actually not give deterministic outputs when processing multiple outputs
    # in parallel even if temperature = 0 is used. The reason for this seems floating point accuracy.
    # Increasing the floating point accuracy to float32 can fix this problem and give deterministic outputs,
    # possibly equivalent to what HF transformers outputs.
    #
    # Also processing only a single prompt at once, i.e. disabling any batching also makes the output deterministic,
    # though the result differs from the HF transformers result quite often, possibly because the exact calculation
    # order is slightly different.
    #
    # So increasing the floating point accuracy to float32 is one option. But it is kind of a pain to do for
    # larger models that already require or would require multiple GPUs. And sometimes even though increasing
    # to float32 does make the results more similar (if the backends are basically equivalent), it is sometimes
    # not enough.
    #
    # So to work around this, one can also try to do multiple attempts at inference with different batch sizes
    # since that's one aspect that can slightly influence the result.
    #
    # We are going to do this here. If we immediately get the correct results on the first try, then we are going
    # to stop immediately, so the user can also just increase the floating point accuracy if they have enough GPUs
    # and if that is enough.
    #
    # Note that if this code fails, i.e. it always gives a different output than HF transformers, then that does
    # not mean that the inference is not basically equivalent to HF transformers. It can still be the case that
    # increasing the floating point accuracy fixes the problem and makes the output equivalent.
    default_backend_model_outputs = [[] for _ in conversations]
    ns = list(range(1, 18)) + [19, 20, 21, 25, 31, 32]
    for i, conversation in enumerate(conversations):
        for n in ns:
            default_backend_model_outputs[i] += get_outputs(model_args, conversation, n)
            if hf_transformers_model_outputs[i] in default_backend_model_outputs[i]:
                break

    print("@@@@@@@@@@@@@@ START CORRECTNESS CHECK RESULTS @@@@@@@@@@@@@@")

    got_error = False
    for i in range(len(conversations)):
        hf_transformers_model_output = (
            hf_transformers_model_outputs[i].replace("\n", "\\n").replace("\r", "\\r")
        )
        default_backend_model_outputs_on_conversation = [
            output.replace("\n", "\\n").replace("\r", "\\r")
            for output in default_backend_model_outputs[i]
        ]

        if (
            hf_transformers_model_output
            in default_backend_model_outputs_on_conversation
        ):
            print("OK")
        else:
            print("ERROR")
            got_error = True

        print("- HF TRANSFORMERS: " + hf_transformers_model_output)
        default_backend_model_output_counts = {}
        for (
            default_backend_model_output
        ) in default_backend_model_outputs_on_conversation:
            if default_backend_model_output not in default_backend_model_output_counts:
                default_backend_model_output_counts[default_backend_model_output] = 0
            default_backend_model_output_counts[default_backend_model_output] += 1

        for (
            default_backend_model_output,
            count,
        ) in default_backend_model_output_counts.items():
            print(
                "- DEFAULT BACKEND ["
                + str(count)
                + "x]: "
                + default_backend_model_output
            )

        print()

    if got_error:
        print("ERROR: Correctness check failed!")
    else:
        print("Correctness check successful!")

    print("@@@@@@@@@@@@@@ END CORRECTNESS CHECK RESULTS @@@@@@@@@@@@@@")

    await unload_model()
    join_threads()
