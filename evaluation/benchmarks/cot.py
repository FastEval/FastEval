import json
import os
import random
import re
import statistics

from evaluation.benchmarks.cot_math_equivalence import is_math_correct
from evaluation.benchmarks.utils import model_name_to_filename
from evaluation.constants import COT_MAX_NEW_TOKENS, COT_TEMPERATURE
from evaluation.models.models import compute_model_replies, create_model

RECOMPUTE_SCORES = False

GSM8K_LIMIT = 500
MATH_LIMIT = 1000
BBH_LIMIT_PER_TASK = 30
MMLU_LIMIT_PER_TASK = 10
AGIEVAL_LIMIT_PER_TASK = 20


def create_conversation(answer_format, question):
    return [
        (
            "user",
            "Please answer the following question step-by-step. "
            "Do not output the answer immediately. "
            "Instead first explain your reasoning step-by-step. "
            "Only afterwards output the answer. "
            "The final line should contain the answer "
            + answer_format
            + "without anything else."
            "\n\n" + question,
        ),
    ]


def evaluate_model_on_dataset(
    *,
    name,
    data,
    question_column,
    answer_column,
    answer_format,
    is_correct,
    output_path,
    limit=float("inf"),
    create_question=None
):
    output_file_path = os.path.join(output_path, name + ".json")
    if os.path.exists(output_file_path):
        _ = yield []
        _ = yield []

        with open(output_file_path) as f:
            output = json.load(f)

        if not RECOMPUTE_SCORES:
            yield output["score"]
            raise Exception("Nothing more to do")

        num_correct = 0
        num_unmatched = 0
        for model_output in output["model_outputs"]:
            model_answer_is_correct = is_correct(
                model_answer=model_output["model_answer"],
                correct_answer=model_output["correct_answer"],
                question=model_output["question"],
            )
            model_output["correct"] = model_answer_is_correct
            if model_answer_is_correct is not None:
                num_correct += float(model_answer_is_correct)
            if (
                model_answer_is_correct is not True
                and model_answer_is_correct is not False
            ):
                num_unmatched += 1
        output["score"] = num_correct / len(output["model_outputs"])
        output["num_unmatched"] = num_unmatched

        with open(output_file_path, "w") as f:
            json.dump(output, f, indent=4)

        yield output["score"]
        raise Exception("Nothing more to do")

    [data] = yield [data]

    randomness = random.Random()
    randomness.seed(1, version=2)
    selected_samples = randomness.sample(range(len(data)), limit)

    requests = []
    for i, item in enumerate(data.select(selected_samples)):
        if isinstance(question_column, str):
            question = item[question_column]
        elif isinstance(question_column, list):
            question = create_question(
                {column: item[column] for column in question_column}
            )
        correct_answer = item[answer_column]
        conversation = create_conversation(answer_format, question)
        requests.append(
            {
                "id": selected_samples[i],
                "question": question,
                "correct_answer": correct_answer,
                "conversation": conversation,
            }
        )

    model_requests = [
        {"conversation": request["conversation"], "temperature": COT_TEMPERATURE}
        for request in requests
    ]
    model_answers = yield model_requests

    model_outputs = []
    num_correct = 0
    num_unmatched = 0
    for i, request in enumerate(requests):
        model_answer = model_answers[i]
        model_answer_is_correct = is_correct(
            model_answer=model_answer,
            correct_answer=request["correct_answer"],
            question=request["question"],
        )
        model_outputs.append(
            {
                "id": request["id"],
                "question": request["question"],
                "correct_answer": request["correct_answer"],
                "model_answer": model_answer,
                "correct": model_answer_is_correct,
            }
        )
        if model_answer_is_correct is not None:
            num_correct += float(model_answer_is_correct)
        if model_answer_is_correct is not True and model_answer_is_correct is not False:
            num_unmatched += 1

    score = num_correct / len(selected_samples)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(
            {
                "score": score,
                "num_unmatched": num_unmatched,
                "model_outputs": model_outputs,
            },
            f,
            indent=4,
        )

    yield score


def evaluate_model_on_gsm8k(output_path):
    def is_correct(model_answer, correct_answer, question):
        model_answer_lines = reversed(model_answer.split("\n"))
        for line in model_answer_lines:
            model_answer_matches = re.findall(r"(\d+(,\d+)*(\.\d+)?)", line)
            if len(model_answer_matches) == 0:
                continue
            model_answer_processed = float(model_answer_matches[-1][0].replace(",", ""))
            correct_answer_processed = float(
                correct_answer.split("\n")[-1].split("####")[1].replace(",", "").strip()
            )
            return abs(model_answer_processed - correct_answer_processed) < 1e-8
        return None

    evaluator = evaluate_model_on_dataset(
        name="gsm8k",
        data=("gsm8k", "main", "test"),
        question_column="question",
        answer_column="answer",
        answer_format="as a single number ",
        is_correct=is_correct,
        output_path=output_path,
        limit=GSM8K_LIMIT,
    )

    datasets = yield next(evaluator)
    model_responses = yield evaluator.send(datasets)
    yield evaluator.send(model_responses)


def evaluate_model_on_math(output_path):
    evaluator = evaluate_model_on_dataset(
        name="math",
        data=("competition_math", None, "test"),
        question_column="problem",
        answer_column="solution",
        answer_format="",
        is_correct=is_math_correct,
        output_path=output_path,
        limit=MATH_LIMIT,
    )

    datasets = yield next(evaluator)
    model_responses = yield evaluator.send(datasets)
    yield evaluator.send(model_responses)


def combine_evaluators(evaluators):
    dataset_requests = []
    for evaluator in evaluators:
        evaluator_dataset_requests = next(evaluator)
        dataset_requests.append(evaluator_dataset_requests)

    dataset_requests_flat = [
        request for requests in dataset_requests for request in requests
    ]
    dataset_responses_flat = yield dataset_requests_flat

    current_index = 0
    model_requests = []
    for i, evaluator in enumerate(evaluators):
        end_index = current_index + len(dataset_requests[i])
        evaluator_dataset_responses = dataset_responses_flat[current_index:end_index]
        evaluator_model_requests = evaluator.send(evaluator_dataset_responses)
        model_requests.append(evaluator_model_requests)
        current_index = end_index

    model_requests_flat = [
        request for requests in model_requests for request in requests
    ]
    model_responses_flat = yield model_requests_flat

    current_index = 0
    scores = []
    for i in range(len(evaluators)):
        end_index = current_index + len(model_requests[i])
        evaluator_model_responses = model_responses_flat[current_index:end_index]
        score = evaluators[i].send(evaluator_model_responses)
        scores.append(score)
        current_index = end_index

    yield scores


def find_multiple_choice_answer(*, model_answer, options):
    letter_options = "".join([option[0] for option in options])

    regex_to_try = [
        r"\([" + letter_options + r"]\)",  # "The answer is (A)."
        r" [" + letter_options + r"]\)",  # "The answer is A)."
        r" [" + letter_options + r"]$",  # "The answer is A"
        r" [" + letter_options + r"][^a-zA-Z]",  # "The answer is A."
        r"[^a-zA-Z][" + letter_options + r"]$",  # "The answer is:A"
        r"[^a-zA-Z][" + letter_options + r"][^a-zA-Z]",  # "The answer is:A."
    ]

    model_answer_lines = list(reversed(model_answer.split("\n")))

    for line in model_answer_lines:
        for regex in regex_to_try:
            model_answer_matches = re.findall(regex, line)
            if len(model_answer_matches) != 0:
                return re.sub("[^" + letter_options + "]", "", model_answer_matches[-1])

    for line in model_answer_lines:
        positions = {}
        for letter_option, text_option in options:
            positions[letter_option] = line.rfind(text_option)
        positions = list(positions.items())
        last_item = max(positions, key=lambda e: e[1])
        last_item_letter, last_item_position = last_item
        if last_item_position == -1:
            continue
        return last_item_letter

    return None


def multiple_choice_is_correct(model_answer, correct_answer, question):
    possible_answers = []
    for line in question.split("\n"):
        if (
            len(line) >= 5
            and line[0] == "("
            and line[2] == ")"
            and line[1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ):
            possible_answers.append((line[1], line[4:]))
    model_answer = find_multiple_choice_answer(
        model_answer=model_answer, options=possible_answers
    )
    if model_answer is None:
        return 1 / len(possible_answers)
    correct_answer = correct_answer.replace("(", "").replace(")", "")
    return model_answer == correct_answer


def evaluate_model_on_bbh(output_path):
    tasks = [
        "date_understanding",
        "disambiguation_qa",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
    ]

    evaluators = combine_evaluators(
        [
            evaluate_model_on_dataset(
                name="bbh/" + task,
                data=("lukaemon/bbh", task, "test"),
                question_column="input",
                answer_column="target",
                answer_format="as a single letter with parenthesis ",
                is_correct=multiple_choice_is_correct,
                output_path=output_path,
                limit=BBH_LIMIT_PER_TASK,
            )
            for task in tasks
        ]
    )

    datasets = yield next(evaluators)
    model_responses = yield evaluators.send(datasets)
    scores = evaluators.send(model_responses)

    yield {
        "tasks": {task: scores[i] for i, task in enumerate(tasks)},
        "average": statistics.mean(scores),
    }


def evaluate_model_on_mmlu(output_path):
    tasks = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def create_question(columns):
        return (
            columns["question"]
            + "\n\n"
            + "\n".join(
                [
                    "(" + name + ") " + columns["choices"][index]
                    for index, name in enumerate(["A", "B", "C", "D"])
                ]
            )
        )

    def is_correct(model_answer, correct_answer, question):
        return multiple_choice_is_correct(
            model_answer=model_answer,
            correct_answer=["A", "B", "C", "D"][correct_answer],
            question=question,
        )

    evaluators = combine_evaluators(
        [
            evaluate_model_on_dataset(
                name="mmlu/" + task,
                data=("cais/mmlu", task, "test"),
                question_column=["question", "choices"],
                create_question=create_question,
                answer_column="answer",
                answer_format="as a single letter with parenthesis ",
                is_correct=is_correct,
                output_path=output_path,
                limit=MMLU_LIMIT_PER_TASK,
            )
            for task in tasks
        ]
    )

    datasets = yield next(evaluators)
    model_responses = yield evaluators.send(datasets)
    scores = evaluators.send(model_responses)

    yield {
        "tasks": {task: scores[i] for i, task in enumerate(tasks)},
        "average": statistics.mean(scores),
    }


def evaluate_model_on_agieval(output_path):
    tasks = [
        "sat_en",
        "sat_en_without_passage",
        "sat_math",
        "lsat_ar",
        "lsat_lr",
        "lsat_rc",
        "logiqa_en",
        "logiqa_zh",
        "aqua_rat",
        "gaokao_biology",
        "gaokao_chemistry",
        "gaokao_chinese",
        "gaokao_chemistry",
        "gaokao_english",
        "gaokao_geography",
        "gaokao_history",
        "gaokao_mathqa",
    ]

    def create_question(columns):
        if columns["passage"] == None or columns["passage"] == "":
            return (
                columns["question"]
                + "\n\n"
                + "\n".join(
                    [
                        columns["options"][index]
                        if columns["options"][index][3] == " "
                        else " ".join(
                            [
                                columns["options"][index][:3],
                                columns["options"][index][3:],
                            ]
                        )
                        for index in range(len(columns["options"]))
                    ]
                )
            )
        return (
            columns["passage"]
            + "\n\n"
            + columns["question"]
            + "\n\n"
            + "\n".join(
                [
                    columns["options"][index]
                    if columns["options"][index][3] == " "
                    else " ".join(
                        [columns["options"][index][:3], columns["options"][index][3:]]
                    )
                    for index in range(len(columns["options"]))
                ]
            )
        )

    def is_correct(model_answer, correct_answer, question):
        return multiple_choice_is_correct(
            model_answer=model_answer,
            correct_answer=["A", "B", "C", "D", "E"][correct_answer],
            question=question,
        )

    evaluators = combine_evaluators(
        [
            evaluate_model_on_dataset(
                name="agieval/" + task,
                data=("kimvu/agieval", task, "test"),
                question_column=["passage", "question", "options"],
                create_question=create_question,
                answer_column="label",
                answer_format="as a single letter with parenthesis ",
                is_correct=is_correct,
                output_path=output_path,
                limit=AGIEVAL_LIMIT_PER_TASK,
            )
            for task in tasks
        ]
    )

    datasets = yield next(evaluators)
    model_responses = yield evaluators.send(datasets)
    scores = evaluators.send(model_responses)

    yield {
        "tasks": {task: scores[i] for i, task in enumerate(tasks)},
        "average": statistics.mean(scores),
    }


def load_datasets(model_name, dataset_requests):
    if len(dataset_requests) == 0:
        return []

    import datasets
    import tqdm

    def load_dataset(dataset_request):
        dataset_name, subset, split = dataset_request
        return datasets.load_dataset(dataset_name, subset)[split]

    return [
        load_dataset(dataset_request)
        for dataset_request in tqdm.tqdm(
            dataset_requests, desc=model_name + " :: CoT :: Loading datasets"
        )
    ]


async def evaluate_model(
    model_type, model_name, model_args, evaluation_id, lower_level_benchmarks
):
    if RECOMPUTE_SCORES:
        print(model_name)

    output_folder = os.path.join(
        "reports", "cot", model_name_to_filename(model_name), evaluation_id
    )
    final_scores_file = os.path.join(output_folder, "scores.json")

    tasks_path = os.path.join(output_folder, "tasks")

    evaluation_functions = [
        ("gsm8k", evaluate_model_on_gsm8k),
        ("math", evaluate_model_on_math),
        ("bbh", evaluate_model_on_bbh),
        ("mmlu", evaluate_model_on_mmlu),
        ("agieval", evaluate_model_on_agieval),
    ]

    tasks = [e.split("/")[1] for e in lower_level_benchmarks]
    evaluation_functions_to_use = [e for e in evaluation_functions if e[0] in tasks]

    evaluators = combine_evaluators(
        [
            evaluation_function(tasks_path)
            for task_name, evaluation_function in evaluation_functions_to_use
        ]
    )

    dataset_requests = next(evaluators)
    datasets = load_datasets(model_name, dataset_requests)

    model_requests = evaluators.send(datasets)

    if len(model_requests) == 0:
        return

    model = await create_model(
        model_type, model_name, model_args, max_new_tokens=COT_MAX_NEW_TOKENS
    )

    model_responses = await compute_model_replies(
        model,
        model_requests,
        progress_bar_description=model_name + " :: CoT :: Computing model replies",
    )

    scores_list = evaluators.send(model_responses)

    scores = {
        task_name: scores_list[i]
        for i, (task_name, _) in enumerate(evaluation_functions_to_use)
    }

    if "gsm8k" in scores and "math" in scores and "bbh" in scores and "mmlu" in scores:
        scores["total"] = (
            0.2 * scores["gsm8k"]
            + 0.4 * scores["math"]
            + 0.2 * scores["bbh"]["average"]
            + 0.2 * scores["mmlu"]["average"]
            + 0.2 * scores["agieval"]["average"]
        )

    os.makedirs(os.path.dirname(final_scores_file), exist_ok=True)
    with open(final_scores_file, "w") as f:
        json.dump(scores, f, indent=4)
