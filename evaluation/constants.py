NUM_THREADS_LOCAL_MODEL = 256
NUM_THREADS_OPENAI_GPT3_5 = 10
NUM_THREADS_OPENAI_GPT4 = 2

DEFAULT_MAX_NEW_TOKENS = 1024

COT_MAX_NEW_TOKENS = 2048

MT_BENCH_JUDGE_MAX_NEW_TOKENS = 2048

MT_BENCH_JUDGE = ('openai', 'gpt-4-0613')

HUMAN_EVAL_PLUS_TEMPERATURE = 0.2
COT_TEMPERATURE = 0

WEIGHTS = {
    # (weight, maximum possible value)
    'mt-bench': (2, 10),
    'cot': (18, 1),
    'human-eval-plus': (3, 1),
    'lm-evaluation-harness': (0, 100),
    'ds1000': (5, 1),
}
