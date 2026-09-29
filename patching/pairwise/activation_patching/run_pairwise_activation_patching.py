#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CODE_DIR = SCRIPT_DIR / "code"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Llama pairwise activation-patching "
            "experiment on the controlled DL20-54 pairs."
        )
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to Llama-3.1-8B-Instruct.",
    )

    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )

    parser.add_argument(
        "--role-json",
        default=str(
            REPO_ROOT
            / "roles"
            / "llama"
            / "selected20"
            / "pairwise"
            / "llama_dl20_pairwise_selected20_1x1_balanced.json"
        ),
    )

    parser.add_argument(
        "--pairs-tsv",
        default=str(
            REPO_ROOT
            / "data"
            / "pairs"
            / "pointwise"
            / "dl20.tsv"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            REPO_ROOT
            / "results"
            / "pairwise_activation_patching"
            / "llama_dl20"
        ),
    )

    return parser.parse_args()


def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )

    try:
        df.to_csv(
            tmp,
            index=False,
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        if tmp.exists():
            tmp.unlink()


def build_units(
    conditions,
    component_targets,
    head_targets,
    role_pairs,
):
    units = []

    for condition in conditions:
        for target in component_targets:
            for role_pair in range(
                1,
                role_pairs + 1,
            ):
                units.append({
                    "condition": condition,
                    "kind": "component",
                    "activation": "resid_pre",
                    "target": target,
                    "role_pair": role_pair,
                })

    for condition in conditions:
        for target in head_targets:
            for role_pair in range(
                1,
                role_pairs + 1,
            ):
                units.append({
                    "condition": condition,
                    "kind": "head",
                    "activation": "z",
                    "target": target,
                    "role_pair": role_pair,
                })

    return units


def unit_paths(
    output_root,
    condition,
    kind,
    activation,
    target,
    role_pair,
):
    clean_role = f"role_{role_pair}"
    corrupt_role = f"role_{role_pair + 10}"

    out_dir = (
        output_root
        / condition
        / kind
    )

    stem = (
        f"pairs_{condition}"
        f"__clean_{clean_role}"
        f"__corrupt_{corrupt_role}"
        f"__{target}"
        f"__{activation}"
    )

    return (
        out_dir
        / f"patch_metrics_{stem}.csv",
        out_dir
        / f"patch_summary_{stem}.csv",
    )


def expected_rows(
    activation,
    expected_qids,
    n_layers,
    n_heads,
):
    rows = (
        expected_qids
        * n_layers
    )

    if activation == "z":
        rows *= n_heads

    return rows


def make_summary(
    df,
    n_layers,
    n_heads,
):
    group_cols = ["layer"]

    if "head" in df.columns:
        group_cols.append("head")

    metric_cols = [
        "clean_correct_logit",
        "clean_wrong_logit",
        "clean_correct_prob",
        "clean_wrong_prob",
        "clean_ld",
        "clean_prob_diff",
        "corrupted_correct_logit",
        "corrupted_wrong_logit",
        "corrupted_correct_prob",
        "corrupted_wrong_prob",
        "corrupted_ld",
        "corrupted_prob_diff",
        "patched_correct_logit",
        "patched_wrong_logit",
        "patched_correct_prob",
        "patched_wrong_prob",
        "patched_ld",
        "patched_prob_diff",
        "normalized_ld",
        "ld_recovery",
        "prob_recovery",
    ]

    metric_cols = [
        c
        for c in metric_cols
        if c in df.columns
    ]

    summary = (
        df.groupby(
            group_cols,
            as_index=False,
        )
        .agg({
            c: "mean"
            for c in metric_cols
        })
    )

    counts = (
        df.groupby(
            group_cols,
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "n_rows"}
        )
    )

    summary = summary.merge(
        counts,
        on=group_cols,
        how="left",
        validate="one_to_one",
    )

    summary.insert(
        0,
        "condition",
        df["condition"].iloc[0],
    )
    summary.insert(
        1,
        "source_pair_id",
        df["source_pair_id"].iloc[0],
    )
    summary.insert(
        2,
        "clean_role",
        df["clean_role"].iloc[0],
    )
    summary.insert(
        3,
        "corrupt_role",
        df["corrupt_role"].iloc[0],
    )
    summary.insert(
        4,
        "patch_target",
        df["patch_target"].iloc[0],
    )
    summary.insert(
        5,
        "patch_activation",
        df["patch_activation"].iloc[0],
    )

    expected = n_layers

    if "head" in df.columns:
        expected *= n_heads

    if len(summary) != expected:
        raise RuntimeError(
            "Summary row-count mismatch: "
            f"{len(summary)} != {expected}"
        )

    return summary


def main():
    args = parse_args()

    # Heavy runtime dependencies are imported only after
    # argument parsing, so --help works in a lightweight environment.
    import numpy as np
    import pandas as pd

    model_path = Path(
        args.model_path
    ).expanduser().resolve()

    role_json = Path(
        args.role_json
    ).expanduser().resolve()

    pairs_tsv = Path(
        args.pairs_tsv
    ).expanduser().resolve()

    output_root = Path(
        args.output_root
    ).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not role_json.is_file():
        raise FileNotFoundError(role_json)

    if not pairs_tsv.is_file():
        raise FileNotFoundError(pairs_tsv)

    # pairwise_core reads these at import time.
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["ROLE_JSON_PATH"] = str(role_json)
    os.environ["PAIRS_TSV"] = str(pairs_tsv)

    if str(CODE_DIR) not in sys.path:
        sys.path.insert(
            0,
            str(CODE_DIR),
        )

    from pairwise_core import (
        COMPONENT_TARGETS,
        CONDITIONS,
        EXPECTED_NO_ID,
        EXPECTED_QIDS,
        EXPECTED_ROLE_PAIRS,
        EXPECTED_YES_ID,
        HEAD_TARGETS,
        install_role_json,
        load_examples,
    )

    from pairwise_activation_ranker import (
        PairwiseActivationPatchingRanker,
    )

    install_role_json()

    units = build_units(
        conditions=CONDITIONS,
        component_targets=COMPONENT_TARGETS,
        head_targets=HEAD_TARGETS,
        role_pairs=EXPECTED_ROLE_PAIRS,
    )

    expected_units = (
        len(CONDITIONS)
        * (
            len(COMPONENT_TARGETS)
            + len(HEAD_TARGETS)
        )
        * EXPECTED_ROLE_PAIRS
    )

    if len(units) != expected_units:
        raise RuntimeError(
            f"Unit-count mismatch: "
            f"{len(units)} != {expected_units}"
        )

    if expected_units != 280:
        raise RuntimeError(
            f"Expected formal 280-unit suite, got "
            f"{expected_units}"
        )

    print(
        "=============================================="
    )
    print(
        "PAIRWISE ACTIVATION PATCHING"
    )
    print(
        "Llama-3.1-8B-Instruct × DL20-54"
    )
    print(
        f"component_targets={COMPONENT_TARGETS}"
    )
    print(
        f"head_targets={HEAD_TARGETS}"
    )
    print(
        f"conditions={CONDITIONS}"
    )
    print(
        f"units={len(units)}"
    )
    print(
        "=============================================="
    )

    # One model load for the whole suite.
    ranker = (
        PairwiseActivationPatchingRanker()
    )

    if ranker.yes_id != EXPECTED_YES_ID:
        raise RuntimeError(
            "Runtime Yes ID mismatch: "
            f"{ranker.yes_id} != "
            f"{EXPECTED_YES_ID}"
        )

    if ranker.no_id != EXPECTED_NO_ID:
        raise RuntimeError(
            "Runtime No ID mismatch: "
            f"{ranker.no_id} != "
            f"{EXPECTED_NO_ID}"
        )

    examples = load_examples(
        ranker.tokenizer
    )

    if len(examples) != EXPECTED_QIDS:
        raise RuntimeError(
            "Example-count mismatch: "
            f"{len(examples)} != "
            f"{EXPECTED_QIDS}"
        )

    n_layers = ranker.model.cfg.n_layers
    n_heads = ranker.model.cfg.n_heads

    required_numeric = [
        "clean_correct_logit",
        "clean_wrong_logit",
        "corrupted_correct_logit",
        "corrupted_wrong_logit",
        "patched_correct_logit",
        "patched_wrong_logit",
        "clean_ld",
        "corrupted_ld",
        "patched_ld",
        "ld_recovery",
    ]

    for index, unit in enumerate(
        units,
        start=1,
    ):
        raw_path, summary_path = (
            unit_paths(
                output_root,
                **unit,
            )
        )

        key = (
            f"{unit['kind']}|"
            f"{unit['condition']}|"
            f"{unit['activation']}|"
            f"{unit['target']}|"
            f"role_{unit['role_pair']}"
            f"->role_{unit['role_pair'] + 10}"
        )

        print()
        print(
            f"[{index:03d}/{len(units)}] "
            f"{key}",
            flush=True,
        )

        # Intentionally always compute the unit.
        # Each invocation executes the requested experiment units directly.
        df = ranker.run_unit(
            examples=examples,
            role_pair=unit["role_pair"],
            condition=unit["condition"],
            activation=unit["activation"],
            target=unit["target"],
        )

        expected = expected_rows(
            activation=unit["activation"],
            expected_qids=EXPECTED_QIDS,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        if len(df) != expected:
            raise RuntimeError(
                f"Row-count mismatch for {key}: "
                f"{len(df)} != {expected}"
            )

        for col in required_numeric:
            if col not in df.columns:
                raise RuntimeError(
                    f"Missing numeric column: {col}"
                )

            values = (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                .to_numpy()
            )

            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Non-finite values: "
                    f"{key}, column={col}"
                )

        summary = make_summary(
            df,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        atomic_csv(
            df,
            raw_path,
        )

        atomic_csv(
            summary,
            summary_path,
        )

        print(
            f"[SAVED] {raw_path}",
            flush=True,
        )
        print(
            f"[SAVED] {summary_path}",
            flush=True,
        )

    print()
    print(
        "PAIRWISE ACTIVATION PATCHING COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
