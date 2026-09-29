#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAIRWISE_ROOT = ROOT / "ranking" / "pairwise"
CODE_DIR = PAIRWISE_ROOT / "common"
MAIN = CODE_DIR / "main.py"


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


DATASET_CONFIG = {
    "dl19": {
        "run_file": ROOT / "data" / "runs" / "dl19.txt",
        "ir_dataset_name": "msmarco-passage/trec-dl-2019",
    },
    "dl20": {
        "run_file": ROOT / "data" / "runs" / "dl20.txt",
        "ir_dataset_name": "msmarco-passage/trec-dl-2020/judged",
    },
    "covid": {
        "run_file": ROOT / "data" / "runs" / "covid.txt",
        "ir_dataset_name": "beir/trec-covid",
    },
}


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
    return [i for i in range(1, expected + 1)] + [
        10 + i for i in range(1, expected + 1)
    ]


def build_common_args(model, dataset, model_path):
    model_cfg = MODEL_CONFIG[model]
    dataset_cfg = DATASET_CONFIG[dataset]

    return [
        "--model_name",
        model_cfg["model_name"],
        "--model_path",
        str(model_path),
        "--tokenizer_name_or_path",
        str(model_path),

        "--run_path",
        str(dataset_cfg["run_file"]),
        "--ir_dataset_name",
        dataset_cfg["ir_dataset_name"],

        "--hits",
        "100",
        "--query_length",
        "20",
        "--passage_length",
        "80",

        "--scoring",
        "generation",
        "--device",
        "cuda",

        "--data_format",
        "pairwise",
        "--method",
        "heapsort",
        "--batch_size",
        "1",

        "--prompt_type",
        "adjusted",
        "--instruction",
        "instruction_2",
        "--output",
        "output_2",
        "--tone",
        "tone_0",

        "--order",
        "passage_first",
        "--position",
        "beginning",
        "--query_before_instruction",
        "True",
        "--role_at_beginning",
        "True",
    ]


def run_one(
    model,
    dataset,
    model_path,
    save_path,
    role,
    role_json=None,
    baseline=False,
):
    env = os.environ.copy()

    if role_json is not None:
        env["ROLE_JSON_PATH"] = str(role_json)
    else:
        env.pop("ROLE_JSON_PATH", None)

    experiment = (
        "baseline_without_role"
        if baseline
        else "zero_shot_ranking"
    )

    cmd = [
        sys.executable,
        str(MAIN),
        experiment,
        *build_common_args(model, dataset, model_path),
        "--save_path",
        str(save_path),
        "--role",
        role,
    ]

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

    subprocess.run(
        cmd,
        cwd=CODE_DIR,
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified pairwise Yes/No ranking launcher."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["llama", "qwen", "mistral"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dl19", "dl20", "covid"],
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face model directory/snapshot.",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the role-free baseline.",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        choices=range(1, 24),
        metavar="1-23",
        help="Role JSON chunk number.",
    )

    parser.add_argument(
        "--role",
        type=int,
        choices=range(1, 21),
        metavar="1-20",
        help="Role number within the selected chunk.",
    )

    parser.add_argument(
        "--all-roles",
        action="store_true",
        help="Run all available roles in one selected chunk.",
    )

    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Run 225 positive and 225 negative roles across 23 chunks.",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model path: {model_path}"
        )

    dataset_cfg = DATASET_CONFIG[args.dataset]

    if not dataset_cfg["run_file"].is_file():
        raise FileNotFoundError(
            f"Missing run file: {dataset_cfg['run_file']}"
        )

    result_root = (
        ROOT
        / "results"
        / "pairwise_ranking"
        / args.model
        / args.dataset
    )

    if args.baseline:
        if any([
            args.chunk is not None,
            args.role is not None,
            args.all_roles,
            args.all_chunks,
        ]):
            parser.error(
                "--baseline cannot be combined with role/chunk options."
            )

        run_one(
            model=args.model,
            dataset=args.dataset,
            model_path=model_path,
            save_path=result_root / "baseline",
            role="role_0",
            baseline=True,
        )
        return

    if args.all_chunks:
        if any([
            args.chunk is not None,
            args.role is not None,
            args.all_roles,
        ]):
            parser.error(
                "--all-chunks cannot be combined with other role/chunk options."
            )

        chunk_ids = range(1, 24)

    else:
        if args.chunk is None:
            parser.error(
                "Specify --chunk, --all-chunks, or --baseline."
            )

        chunk_ids = [args.chunk]

        if args.all_roles:
            if args.role is not None:
                parser.error(
                    "Use either --role or --all-roles."
                )
        elif args.role is None:
            parser.error(
                "With --chunk, specify --role or --all-roles."
            )

    model_cfg = MODEL_CONFIG[args.model]

    for chunk_id in chunk_ids:
        chunk_name = (
            f"{model_cfg['role_prefix']}_chunk_{chunk_id:03d}"
        )

        role_json = (
            model_cfg["role_dir"]
            / f"{chunk_name}.json"
        )

        if not role_json.is_file():
            raise FileNotFoundError(
                f"Missing role JSON: {role_json}"
            )

        available = role_ids_for_chunk(role_json, chunk_id)
        if args.role is not None and args.role not in available:
            parser.error(
                f"role_{args.role} is not in chunk_{chunk_id:03d}"
            )
        role_ids = [args.role] if args.role is not None else available

        for role_id in role_ids:
            role = f"role_{role_id}"

            save_path = (
                result_root
                / "chunks"
                / f"chunk_{chunk_id:03d}"
                / role
            )

            run_one(
                model=args.model,
                dataset=args.dataset,
                model_path=model_path,
                save_path=save_path,
                role=role,
                role_json=role_json,
                baseline=False,
            )


if __name__ == "__main__":
    main()
