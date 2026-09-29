#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RANKING_ROOT = ROOT / "ranking" / "pointwise"
COMMON_MAIN = RANKING_ROOT / "common" / "main.py"
DATASETS_TSV = ROOT / "data" / "datasets.tsv"

MODEL_CONFIG = {
    "llama": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "role_dir": ROOT / "roles" / "llama",
        "role_prefix": "llama_first_stage_5each_token_fullcross_fixed10",
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "role_dir": ROOT / "roles" / "qwen",
        "role_prefix": "qwen_first_stage_5each_token_fullcross_fixed10",
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "role_dir": ROOT / "roles" / "mistral",
        "role_prefix": "mistral_adjadv_balanced_15x15_tokenmix_fixed10",
    },
}


def load_datasets():
    datasets = {}

    with DATASETS_TSV.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            datasets[row["dataset"]] = {
                "ir_dataset_name": row["ir_dataset_name"],
                "run_file": ROOT / row["run_file"],
            }

    return datasets


def role_ids_for_chunk(path, chunk_id):
    with path.open(encoding="utf-8") as f:
        roles = json.load(f)
    expected = 5 if chunk_id == 23 else 10
    keys = [f"role_{i}" for i in range(1, expected + 1)]
    for polarity in ("pos_roles", "neg_roles"):
        group = roles.get(polarity)
        if not isinstance(group, dict) or list(group) != keys:
            raise ValueError(
                f"{path}: expected exactly {keys} in {polarity}"
            )
    return [f"role_{i}" for i in range(1, expected + 1)] + [
        f"role_{10 + i}" for i in range(1, expected + 1)
    ]


def run_one(
    model,
    dataset,
    model_path,
    save_path,
    role="role_0",
    role_json=None,
    baseline=False,
):
    model_cfg = MODEL_CONFIG[model]
    dataset_cfg = load_datasets()[dataset]

    code_dir = RANKING_ROOT / model / "code"

    env = os.environ.copy()

    # Make the model-specific pointwise_ranker.py visible to common/main.py.
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(code_dir)
        if not old_pythonpath
        else str(code_dir) + os.pathsep + old_pythonpath
    )

    if role_json is not None:
        env["ROLE_JSON_PATH"] = str(role_json)

        # Mistral's adapter also supports this variable.
        env["ROLE_DICT_PATH"] = str(role_json)

    experiment = "baseline_without_role" if baseline else "zero_shot_ranking"

    cmd = [
        sys.executable,
        str(COMMON_MAIN),
        experiment,
        "--run_path",
        str(dataset_cfg["run_file"]),
        "--save_path",
        str(save_path),
        "--model_path",
        str(model_path),
        "--model_name",
        model_cfg["model_name"],
        "--role",
        role,
        "--ir_dataset_name",
        dataset_cfg["ir_dataset_name"],
        "--hits",
        "100",
        "--query_length",
        "20",
        "--passage_length",
        "80",
        "--batch_size",
        "128",
        "--data_format",
        "pointwise",
        "--method",
        "yes_no",
        "--order",
        "query_first",
        "--position",
        "beginning",
        "--role_at_beginning",
        "True",
        "--query_before_instruction",
        "False",
        "--prompt_type",
        "adjusted",
        "--instruction",
        "instruction_1",
        "--output",
        "output_1",
        "--tone",
        "tone_1",
        "--device",
        "cuda",
    ]

    # Llama and Mistral historical launchers explicitly used the local
    # model directory as tokenizer path. Using it here for all three models
    # keeps model/tokenizer loading self-contained.
    cmd.extend([
        "--tokenizer_name_or_path",
        str(model_path),
    ])

    save_path.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print(f"model      : {model}")
    print(f"dataset    : {dataset}")
    print(f"experiment : {experiment}")
    print(f"role       : {role}")
    print(f"save_path  : {save_path}")

    if role_json is not None:
        print(f"role_json  : {role_json}")

    print("============================================================\n")

    subprocess.run(cmd, env=env, cwd=code_dir, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Unified pointwise role-ranking launcher."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["llama", "qwen", "mistral"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "dl19",
            "dl20",
            "scifact",
            "covid",
            "fiqa",
            "climatefever",
        ],
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to the Hugging Face model snapshot/directory.",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        choices=range(1, 24),
        metavar="1-23",
        help="Run one role JSON chunk.",
    )

    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Run 225 positive and 225 negative roles across 23 chunks.",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the empty-role baseline.",
    )

    args = parser.parse_args()

    if sum([
        bool(args.chunk),
        args.all_chunks,
        args.baseline,
    ]) != 1:
        parser.error(
            "Choose exactly one of --chunk, --all-chunks, or --baseline."
        )

    model_cfg = MODEL_CONFIG[args.model]

    result_root = (
        ROOT
        / "results"
        / "pointwise_ranking"
        / args.model
        / args.dataset
    )

    if args.baseline:
        run_one(
            model=args.model,
            dataset=args.dataset,
            model_path=Path(args.model_path).expanduser().resolve(),
            save_path=result_root / "baseline",
            baseline=True,
        )
        return

    if args.chunk:
        chunk_ids = [args.chunk]
    else:
        chunk_ids = range(1, 24)

    for chunk_id in chunk_ids:
        chunk_name = (
            f"{model_cfg['role_prefix']}_chunk_{chunk_id:03d}"
        )

        role_json = model_cfg["role_dir"] / f"{chunk_name}.json"

        if not role_json.is_file():
            raise FileNotFoundError(
                f"Missing role JSON: {role_json}"
            )

        for role in role_ids_for_chunk(role_json, chunk_id):
            run_one(
                model=args.model,
                dataset=args.dataset,
                model_path=Path(args.model_path).expanduser().resolve(),
                save_path=result_root / chunk_name,
                role=role,
                role_json=role_json,
                baseline=False,
            )


if __name__ == "__main__":
    main()
