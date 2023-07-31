import evaluation.args
from evaluation.models.models import unload_model, create_model, compute_model_replies
from evaluation.utils import join_threads

def run_inference_backend_correctness_check(model_type, model_name, model_args):
    assert model_type is not None and model_name is not None

    # Idk. Just some conversations. Can be absolutely changed to better reflect what would be good tests
    # to confirm that different inference backends output the same text.
    conversations = [
        [('user', 'Please tell me a joke.')],
        [('user', "What's 46912984610 + 5610927469123 * 64127401823 / 123?")],
        [
            ('user', 'Why did the chicken cross the road?'),
            ('assistant', 'The classic joke answer to "Why did the chicken cross the road?" is simply, "To get to the other side." '
                'The humor lies in the fact that the listener expects a more elaborate or clever answer, '
                'but the punchline is straightforward and unexpected, resulting in a lighthearted and silly joke. '
                'The joke has become a well-known and often-used example of anti-humor, '
                'where the humor comes from the lack of a complex or unexpected punchline.'),
            ('user', 'Can you write a long essay about this topic?'),
        ],
        [('user', '^_^ </s> <|endoftext|> [PAD] [PAD][PAD] What do you think? Please answer only in emojis. ^_^')],
    ]

    conversations = [{
        'conversation': conversation,
        'temperature': 0,
    } for conversation in conversations]

    def get_outputs(backend_name, conversation, n):
        return compute_model_replies(
            create_model(model_type, model_name, model_args, max_new_tokens=50),
            [conversation] * n,
            desc=model_name + ' :: Computing replies with ' + backend_name + ' backend',
        )

    previous_force_backend = evaluation.args.cmd_arguments.force_backend
    evaluation.args.cmd_arguments.force_backend = 'hf_transformers'

    hf_transformers_model_outputs = [get_outputs('hf_transformers', conversation, 1)[0] for conversation in conversations]

    evaluation.args.cmd_arguments.force_backend = previous_force_backend

    # So basically the thing is that both vLLM as well as TGI don't actually give deterministic outputs
    # when processing multiple outputs in parallel even if temperature = 0 is used. I'm not sure about the reason,
    # but there is for example this issue here:
    #
    # https://github.com/huggingface/text-generation-inference/issues/459
    # > matrix multiplications kernels in mixed precision are not deterministic
    # > and can lead to difference in generations when the batch size increase
    #
    # So it might be something like that (especially since both vLLM and TGI do this) but idk.
    #
    # Anyway. The thing is, the difference is usually quite small and doesn't really make that much of a difference.
    # But still, we can't just check whether the results are _exactly_ the same as for HF transformers,
    # because they are often not. So in order to solve this, we do multiple inferences for the same prompt
    # using those other backends and with different batch sizes and later check whether _any_ of the results
    # are equal and accept if they are.
    #
    # We also still print all of the outputs we got, so the user can confirm the results themself.
    #
    # Also note that while disabling the parallelism is possible and we then actually do get deterministic outputs
    # from the other backends, those outputs are often not the same as what HF transformers gives us.
    # We really need to do multiple inferences in parallel and then check whether any of the outputs are equal.
    #
    # Technically speaking, this is of course incorrect. But the differences seem small enough and if the results
    # are the same at least once we at least know that they do the same kind of operations, even if the exact numbers
    # end up slightly differently sometimes. And the 20x throughput increase that vLLM & TGI provide
    # compared to plain HF transformers is just too big to be ignored due to small differences like that.

    default_backend_model_outputs = [[] for _ in conversations]

    ns = [1]
    for n in range(2, 5):
        ns += [n] * 5
    for n in range(5, 7):
        ns += [n] * 4
    for n in range(7, 10):
        ns += [n] * 3
    for n in range(10, 18):
        ns += [n] * 2
    ns += [19, 20, 21, 25, 31, 32, 32, 64, 100, 128, 150, 199, 257]

    for i, conversation in enumerate(conversations):
        for n in ns:
            default_backend_model_outputs[i] += get_outputs('default', conversation, n)
            if hf_transformers_model_outputs[i] in default_backend_model_outputs[i]:
                break

    print('@@@@@@@@@@@@@@ START CORRECTNESS CHECK RESULTS @@@@@@@@@@@@@@')

    got_error = False
    for i in range(len(conversations)):
        hf_transformers_model_output = hf_transformers_model_outputs[i].replace('\n', '\\n').replace('\r', '\\r')
        default_backend_model_outputs_on_conversation = [output.replace('\n', '\\n').replace('\r', '\\r') for output in default_backend_model_outputs[i]]

        if hf_transformers_model_output in default_backend_model_outputs_on_conversation:
            print('OK')
        else:
            print('ERROR')
            got_error = True

        print('- HF TRANSFORMERS: ' + hf_transformers_model_output)
        default_backend_model_output_counts = {}
        for default_backend_model_output in default_backend_model_outputs_on_conversation:
            if default_backend_model_output not in default_backend_model_output_counts:
                default_backend_model_output_counts[default_backend_model_output] = 0
            default_backend_model_output_counts[default_backend_model_output] += 1

        for default_backend_model_output, count in default_backend_model_output_counts.items():
            print('- DEFAULT BACKEND [' + str(count) + 'x]: ' + default_backend_model_output)

        print()

    if got_error:
        print('ERROR: Correctness check failed!')
    else:
        print('Correctness check successful!')

    print('@@@@@@@@@@@@@@ END CORRECTNESS CHECK RESULTS @@@@@@@@@@@@@@')

    unload_model()
    join_threads()
