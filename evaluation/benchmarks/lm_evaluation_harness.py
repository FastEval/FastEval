import json
import os
import shutil
import statistics
import subprocess

import evaluation.args
from evaluation.benchmarks.utils import model_name_to_filename
from evaluation.models.models import get_dtype


async def run_evaluation(*, model_name, model_args, output_path):
    tasks = [
        "openbookqa",
        "arc_easy",
        "winogrande",
        "hellaswag",
        "arc_challenge",
        "piqa",
        "boolq",
    ]

    kwargs = {}
    if "tokenizer" in model_args:
        kwargs["tokenizer"] = model_args["tokenizer"]

    async def build_lm_eval_command(*, parallelize):
        lm_eval_model_args = ",".join(
            [
                k + "=" + str(v)
                for k, v in (
                    {
                        "pretrained": model_name,
                        "dtype": str(await get_dtype(model_name)).replace("torch.", ""),
                        "trust_remote_code": True,
                        "parallelize": parallelize,
                        **kwargs,
                    }
                ).items()
            ]
        )

        cmd = [
            shutil.which("lm_eval"),
            "--batch_size",
            "auto:5",
            "--tasks",
            ",".join(tasks),
            "--model",
            "hf",
            "--model_args",
            lm_eval_model_args,
            "--output_path",
            output_path,
        ]

        return cmd

    if evaluation.args.cmd_arguments.num_gpus_per_model == 1:
        cmd = ["accelerate", "launch"] + await build_lm_eval_command(parallelize=False)
    else:
        cmd = await build_lm_eval_command(parallelize=True)

    print(model_name + " :: LM-Eval :: Evaluating")

    subprocess.run(cmd)


async def evaluate_model(model_type, model_name, model_args, evaluation_id):
    if model_type == "openai":
        return

    output_folder = os.path.join(
        "./reports/lm-evaluation-harness",
        model_name_to_filename(model_name),
        evaluation_id,
    )
    os.makedirs(output_folder, exist_ok=True)

    gpt4all_output_filepath = os.path.join(output_folder, "gpt4all.json")
    if not os.path.exists(gpt4all_output_filepath):
        await run_evaluation(
            model_name=model_name,
            model_args=model_args,
            output_path=gpt4all_output_filepath,
        )

    with open(gpt4all_output_filepath) as f:
        results = json.load(f)

    total_scores_filepath = os.path.join(output_folder, "total.json")
    if os.path.exists(total_scores_filepath):
        return

    scores = {"tasks": {}}
    for task_name in results["results"].keys():
        task_results = results["results"][task_name]
        score = (
            task_results.get("acc_norm")
            or task_results.get("acc")
            or task_results.get("acc_norm,none")
            or task_results.get("acc,none")
        )
        scores["tasks"][task_name] = 100 * score

    scores["average"] = statistics.mean(scores["tasks"].values())

    with open(total_scores_filepath, "w") as f:
        json.dump(scores, f, indent=4)
