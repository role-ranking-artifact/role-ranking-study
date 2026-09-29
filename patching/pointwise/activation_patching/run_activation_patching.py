#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "patching" / "pointwise" / "activation_patching" / "code"

MODEL_CONFIG = {
    "llama": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "role_json": (
            REPO_ROOT
            / "roles"
            / "llama"
            / "selected20"
            / "llama_best_token_combo_selected20_nested_for_patching.json"
        ),
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "role_json": (
            REPO_ROOT
            / "roles"
            / "qwen"
            / "selected20"
            / "qwen_best_token_combo_selected20_nested_for_patching.json"
        ),
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "role_json": (
            REPO_ROOT
            / "roles"
            / "mistral"
            / "selected20"
            / "mistral_best_token_combo_selected20_nested_for_patching.json"
        ),
    },
}


DATASET_CONFIG = {
    "dl19": {
        "ir_dataset_name": "msmarco-passage/trec-dl-2019",
        "pairs": REPO_ROOT / "data" / "pairs" / "pointwise" / "dl19.tsv",
    },
    "dl20": {
        "ir_dataset_name": "msmarco-passage/trec-dl-2020",
        "pairs": REPO_ROOT / "data" / "pairs" / "pointwise" / "dl20.tsv",
    },
    "covid": {
        "ir_dataset_name": "beir/trec-covid",
        "pairs": REPO_ROOT / "data" / "pairs" / "pointwise" / "covid.tsv",
    },
}


COMPONENT_ACTIVATIONS = [
    "resid_pre",
    "resid_post",
    "attn_out",
    "mlp_out",
]

COMPONENT_TARGETS = [
    "role_adj",
    "role_adv",
    "role_adj_adv",
    "role_all",
    "query_all",
    "doc_all",
    "inst_all",
    "last",
]

HEAD_ACTIVATIONS = [
    "z",
]

HEAD_TARGETS = [
    "role_all",
    "role_adj",
    "role_adv",
    "role_adj_adv",
    "inst_all",
    "last",
]

ROLE_PAIRS = [
    (f"role_{i}", f"role_{i + 10}")
    for i in range(1, 11)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run unified pointwise activation patching for "
            "Llama, Qwen, or Mistral."
        )
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
        "--condition",
        default="all",
        choices=["relevance", "irrelevance", "all"],
    )

    parser.add_argument(
        "--kind",
        default="all",
        choices=["component", "head", "all"],
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model path or Hugging Face model identifier.",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Logical model name used by the ranker. "
            "Defaults to the canonical name for --model."
        ),
    )

    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path. Defaults to --model-path.",
    )

    parser.add_argument(
        "--role-json",
        default=None,
        help="Override the default selected20 role JSON.",
    )

    parser.add_argument(
        "--pairs-path",
        default=None,
        help="Override the default controlled-pairs TSV.",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output root. Defaults to "
            "results/pointwise_activation_patching."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    return parser.parse_args()


def validate_file(path, label):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )

    return path.resolve()


def build_units(kind):
    units = []

    if kind in ("component", "all"):
        for activation in COMPONENT_ACTIVATIONS:
            for target in COMPONENT_TARGETS:
                for clean_role, corrupt_role in ROLE_PAIRS:
                    units.append(
                        {
                            "kind": "component",
                            "activation": activation,
                            "target": target,
                            "axis": "layer,pos",
                            "clean_role": clean_role,
                            "corrupt_role": corrupt_role,
                        }
                    )

    if kind in ("head", "all"):
        for activation in HEAD_ACTIVATIONS:
            for target in HEAD_TARGETS:
                for clean_role, corrupt_role in ROLE_PAIRS:
                    units.append(
                        {
                            "kind": "head",
                            "activation": activation,
                            "target": target,
                            "axis": "layer,pos,head",
                            "clean_role": clean_role,
                            "corrupt_role": corrupt_role,
                        }
                    )

    return units


def main():
    args = parse_args()

    model_cfg = MODEL_CONFIG[args.model]
    dataset_cfg = DATASET_CONFIG[args.dataset]

    model_path = args.model_path
    model_name = (
        args.model_name
        if args.model_name
        else model_cfg["model_name"]
    )
    tokenizer_path = (
        args.tokenizer_path
        if args.tokenizer_path
        else model_path
    )

    role_json = (
        Path(args.role_json)
        if args.role_json
        else model_cfg["role_json"]
    )

    pairs_path = (
        Path(args.pairs_path)
        if args.pairs_path
        else dataset_cfg["pairs"]
    )

    role_json = validate_file(
        role_json,
        "Role JSON",
    )

    pairs_path = validate_file(
        pairs_path,
        "Pairs TSV",
    )

    main_py = validate_file(
        CODE_DIR / "main.py",
        "Activation-patching main.py",
    )

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = (
            REPO_ROOT
            / "results"
            / "pointwise_activation_patching"
        )

    conditions = (
        ["relevance", "irrelevance"]
        if args.condition == "all"
        else [args.condition]
    )

    units = build_units(args.kind)

    env = os.environ.copy()
    env["ROLE_JSON_PATH"] = str(role_json)

    total = len(units) * len(conditions)

    print("===== POINTWISE ACTIVATION PATCHING =====")
    print(f"model          = {args.model}")
    print(f"model_name     = {model_name}")
    print(f"model_path     = {model_path}")
    print(f"tokenizer      = {tokenizer_path}")
    print(f"dataset        = {args.dataset}")
    print(
        f"ir_dataset     = "
        f"{dataset_cfg['ir_dataset_name']}"
    )
    print(f"pairs          = {pairs_path}")
    print(f"role_json      = {role_json}")
    print(f"condition      = {args.condition}")
    print(f"kind           = {args.kind}")
    print("batch_size     = 128")
    print("query_length   = 20")
    print("passage_length = 80")
    print(f"total units    = {total}")
    print()

    unit_number = 0

    for condition in conditions:
        for unit in units:
            unit_number += 1

            save_path = (
                output_root
                / args.model
                / args.dataset
                / condition
                / unit["kind"]
            )

            save_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            cmd = [
                sys.executable,
                str(main_py),
                "activation_patching",

                "--pairs_path",
                str(pairs_path),

                "--doc_source",
                condition,

                "--save_path",
                str(save_path),

                "--model_path",
                model_path,

                "--model_name",
                model_name,

                "--tokenizer_name_or_path",
                tokenizer_path,

                "--ir_dataset_name",
                dataset_cfg["ir_dataset_name"],

                "--data_format",
                "pointwise",

                "--method",
                "yes_no",

                "--batch_size",
                "128",

                "--query_length",
                "20",

                "--passage_length",
                "80",

                "--device",
                "cuda",

                "--prompt_type",
                "adjusted",

                "--order",
                "query_first",

                "--position",
                "beginning",

                "--query_before_instruction",
                "True",

                "--role_at_beginning",
                "True",

                "--clean_role",
                unit["clean_role"],

                "--corrupt_role",
                unit["corrupt_role"],

                "--patch_target",
                unit["target"],

                "--patch_activation",
                unit["activation"],

                "--patch_index_axis_names",
                unit["axis"],

                "--patch_direction",
                "clean_to_corrupt",

                "--query_doc_limit",
                "1",
            ]

            label = (
                f"[{unit_number}/{total}] "
                f"{args.model} | "
                f"{args.dataset} | "
                f"{condition} | "
                f"{unit['kind']} | "
                f"{unit['activation']} | "
                f"{unit['target']} | "
                f"{unit['clean_role']} -> "
                f"{unit['corrupt_role']}"
            )

            print(label)

            if args.dry_run:
                print(
                    " ".join(
                        subprocess.list2cmdline([x])
                        for x in cmd
                    )
                )
                continue

            subprocess.run(
                cmd,
                cwd=CODE_DIR,
                env=env,
                check=True,
            )


if __name__ == "__main__":
    main()
