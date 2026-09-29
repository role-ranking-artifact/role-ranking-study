#!/usr/bin/env python3
"""Llama DL20 Pairwise Q-only three-stage causal path patching.

The intervention logic is taken from the completed Pairwise Q-only run.
The `inst_all` mediator is the question line only, as returned by the
formal Pairwise prompt generator. Results and run state are written under
the output directory and can be resumed by rerunning the same command.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CODE_DIR = SCRIPT_DIR / "code"
SHARED_CODE_DIR = REPO_ROOT / "patching" / "pairwise" / "activation_patching" / "code"

for module_dir in (SHARED_CODE_DIR, CODE_DIR):
    if not module_dir.is_dir():
        raise FileNotFoundError(f"Required code directory: {module_dir}")
    if str(module_dir) in sys.path:
        sys.path.remove(str(module_dir))
    sys.path.insert(0, str(module_dir))

EXPECTED_INST = "Is document A more relevant to the query?\n"
EXPECTED_PROMPT_TAIL = (
    "Is document A more relevant to the query?\n"
    "Answer 'Yes' or 'No'.\n"
    "Answer: "
)


# Same C0-C7 interventions and per-example metrics as the completed run.
def ensure_list_pos(x):
    if isinstance(x, list):
        return [int(v) for v in x]

    if isinstance(x, tuple):
        return [int(v) for v in x]

    if isinstance(x, slice):
        if x.start is None or x.stop is None:
            raise RuntimeError(
                f"Open slice cannot be converted: {x}"
            )

        step = (
            1
            if x.step is None
            else x.step
        )

        return list(
            range(
                x.start,
                x.stop,
                step,
            )
        )

    return [int(x)]


def get_pos_from_ranges(
    seqlen,
    pos_type,
    role_range=None,
    query_range=None,
    doc_range=None,
    inst_range=None,
):
    return get_pos(
        seqlen=seqlen,
        pos_type=pos_type,
        role_range=role_range,
        query_range=query_range,
        doc_range=doc_range,
        inst_range=inst_range,
    )


base = None  # Initialized after CLI validation and imports in main().


def span_positions(rng):
    """[(start,end)] -> [start, ..., end-1]."""
    start, end = rng[0]
    return list(range(int(start), int(end)))

def safe_div(num, den):
    if abs(den) < 1e-12:
        return math.nan
    return num / den

def run_one_example(
    model,
    clean_tokens,
    corrupt_tokens,
    clean_ranges,
    corrupt_ranges,
    source_layer,
    source_head,
    mediator_layer,
    mediator_head,
    receiver_layer,
    receiver_head,
    correct_token_id,
    wrong_token_id,
):
    # ---------------------------------------------------------
    # Positions
    # ---------------------------------------------------------
    clean_seqlen = clean_tokens.shape[-1]
    corrupt_seqlen = corrupt_tokens.shape[-1]

    clean_role_range = clean_ranges[0]
    corrupt_role_range = corrupt_ranges[0]

    clean_inst_range = clean_ranges[3]
    corrupt_inst_range = corrupt_ranges[3]

    clean_source_pos = base.ensure_list_pos(
        base.get_pos_from_ranges(
            clean_seqlen,
            "role_adj_adv",
            clean_role_range,
        )
    )

    corrupt_source_pos = base.ensure_list_pos(
        base.get_pos_from_ranges(
            corrupt_seqlen,
            "role_adj_adv",
            corrupt_role_range,
        )
    )

    clean_inst_pos = span_positions(clean_inst_range)
    corrupt_inst_pos = span_positions(corrupt_inst_range)

    clean_last_pos = clean_seqlen - 1
    corrupt_last_pos = corrupt_seqlen - 1

    if len(clean_source_pos) != len(corrupt_source_pos):
        raise RuntimeError(
            f"source position mismatch: "
            f"clean={clean_source_pos} corrupt={corrupt_source_pos}"
        )

    if len(clean_inst_pos) != len(corrupt_inst_pos):
        raise RuntimeError(
            f"instruction length mismatch: "
            f"clean={len(clean_inst_pos)} corrupt={len(corrupt_inst_pos)}"
        )

    # Strict feed-forward chain.
    if not (source_layer < mediator_layer < receiver_layer):
        raise RuntimeError(
            f"Invalid causal layer order: "
            f"L{source_layer} -> L{mediator_layer} -> L{receiver_layer}"
        )

    source_hook_name = base.utils.get_act_name("z", source_layer)
    mediator_hook_name = base.utils.get_act_name("z", mediator_layer)
    receiver_hook_name = base.utils.get_act_name("z", receiver_layer)

    cache_names = [
        source_hook_name,
        mediator_hook_name,
        receiver_hook_name,
    ]

    # ---------------------------------------------------------
    # Baseline clean / corrupted runs + caches
    # ---------------------------------------------------------
    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens,
        names_filter=cache_names,
    )

    corrupt_logits, corrupt_cache = model.run_with_cache(
        corrupt_tokens,
        names_filter=cache_names,
    )

    clean_ld = float(
        base.logit_diff(
            clean_logits,
            correct_token_id,
            wrong_token_id,
        )
    )

    corrupted_ld = float(
        base.logit_diff(
            corrupt_logits,
            correct_token_id,
            wrong_token_id,
        )
    )

    clean_prob_diff = float(
        base.prob_diff(
            clean_logits,
            correct_token_id,
            wrong_token_id,
        )
    )

    corrupted_prob_diff = float(
        base.prob_diff(
            corrupt_logits,
            correct_token_id,
            wrong_token_id,
        )
    )

    # ---------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------
    def source_patch_hook(z, hook):
        for cp, bp in zip(clean_source_pos, corrupt_source_pos):
            z[:, bp, source_head, :] = (
                clean_cache[source_hook_name][:, cp, source_head, :]
            )
        return z

    def mediator_block_hook(z, hook):
        # Reset mediator head at ALL instruction tokens
        # back to original corrupted activation.
        z[:, corrupt_inst_pos, mediator_head, :] = (
            corrupt_cache[mediator_hook_name][
                :, corrupt_inst_pos, mediator_head, :
            ]
        )
        return z

    def clean_mediator_patch_hook(z, hook):
        # Patch clean instruction-head activations into corrupted run,
        # position-by-position.
        z[:, corrupt_inst_pos, mediator_head, :] = (
            clean_cache[mediator_hook_name][
                :, clean_inst_pos, mediator_head, :
            ]
        )
        return z

    def receiver_block_hook(z, hook):
        # Last = final assistant-generation prompt position.
        z[:, corrupt_last_pos, receiver_head, :] = (
            corrupt_cache[receiver_hook_name][
                :, corrupt_last_pos, receiver_head, :
            ]
        )
        return z

    # ---------------------------------------------------------
    # C1:
    # Source patched.
    #
    # IMPORTANT:
    # capture the Instruction mediator activation ACTUALLY
    # produced downstream of the Role source patch.
    # ---------------------------------------------------------
    captured = {}

    def capture_source_induced_mediator_hook(z, hook):
        captured["mediator"] = (
            z[:, corrupt_inst_pos, mediator_head, :]
            .detach()
            .clone()
        )
        return z

    source_patched_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[
            (source_hook_name, source_patch_hook),
            (
                mediator_hook_name,
                capture_source_induced_mediator_hook,
            ),
        ],
    )

    if "mediator" not in captured:
        raise RuntimeError(
            "Failed to capture source-induced mediator activation"
        )

    source_induced_mediator = captured["mediator"]

    # ---------------------------------------------------------
    # C2:
    # Source patched + Instruction mediator blocked.
    # Tests Role -> Instruction mediation.
    # ---------------------------------------------------------
    source_mediator_blocked_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[
            (source_hook_name, source_patch_hook),
            (mediator_hook_name, mediator_block_hook),
        ],
    )

    # ---------------------------------------------------------
    # C3:
    # Clean Instruction mediator patched into corrupted run.
    # ---------------------------------------------------------
    mediator_patched_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[
            (mediator_hook_name, clean_mediator_patch_hook),
        ],
    )

    # ---------------------------------------------------------
    # C4:
    # Clean mediator patched + Last receiver blocked.
    # Tests Instruction -> Last mediation.
    # ---------------------------------------------------------
    mediator_receiver_blocked_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[
            (mediator_hook_name, clean_mediator_patch_hook),
            (receiver_hook_name, receiver_block_hook),
        ],
    )

    # ---------------------------------------------------------
    # C5/C6:
    # FULL CHAIN TEST.
    #
    # No source patch is applied in these fresh runs.
    # Instead, transplant ONLY the Instruction-head state that
    # was produced by C1 after Role source patching.
    # ---------------------------------------------------------
    def source_induced_mediator_patch_hook(z, hook):
        z[:, corrupt_inst_pos, mediator_head, :] = (
            source_induced_mediator
        )
        return z

    source_induced_mediator_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[
            (
                mediator_hook_name,
                source_induced_mediator_patch_hook,
            ),
        ],
    )

    source_induced_mediator_receiver_blocked_logits = (
        model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[
                (
                    mediator_hook_name,
                    source_induced_mediator_patch_hook,
                ),
                (receiver_hook_name, receiver_block_hook),
            ],
        )
    )

    # ---------------------------------------------------------
    # C7:
    # Source patch + Last receiver blocked.
    # This reproduces the old two-node Role -> Last test
    # for direct comparison with the completed DL20 full36.
    # ---------------------------------------------------------
    source_receiver_blocked_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[
            (source_hook_name, source_patch_hook),
            (receiver_hook_name, receiver_block_hook),
        ],
    )

    # ---------------------------------------------------------
    # Convert all runs to label-aware LD / probability difference
    # ---------------------------------------------------------
    def get_ld(logits):
        return float(
            base.logit_diff(
                logits,
                correct_token_id,
                wrong_token_id,
            )
        )

    def get_pd(logits):
        return float(
            base.prob_diff(
                logits,
                correct_token_id,
                wrong_token_id,
            )
        )

    source_patched_ld = get_ld(source_patched_logits)
    source_mediator_blocked_ld = get_ld(
        source_mediator_blocked_logits
    )

    mediator_patched_ld = get_ld(mediator_patched_logits)
    mediator_receiver_blocked_ld = get_ld(
        mediator_receiver_blocked_logits
    )

    source_induced_mediator_ld = get_ld(
        source_induced_mediator_logits
    )

    source_induced_mediator_receiver_blocked_ld = get_ld(
        source_induced_mediator_receiver_blocked_logits
    )

    source_receiver_blocked_ld = get_ld(
        source_receiver_blocked_logits
    )

    # ---------------------------------------------------------
    # Core effects
    # ---------------------------------------------------------
    clean_corrupt_gap = clean_ld - corrupted_ld

    # Overall recovery caused by Role source patch
    source_recovery = source_patched_ld - corrupted_ld

    # Stage A: Role -> Instruction
    role_to_inst_drop = (
        source_patched_ld
        - source_mediator_blocked_ld
    )

    # Independent Instruction clean-patch recovery
    mediator_clean_recovery = (
        mediator_patched_ld
        - corrupted_ld
    )

    # Stage B: Instruction -> Last
    inst_to_last_drop = (
        mediator_patched_ld
        - mediator_receiver_blocked_ld
    )

    # Effect carried by the Instruction state specifically induced
    # by the Role source patch.
    source_induced_mediator_recovery = (
        source_induced_mediator_ld
        - corrupted_ld
    )

    # Stage C: full Role -> Instruction -> Last chain.
    chain_to_last_drop = (
        source_induced_mediator_ld
        - source_induced_mediator_receiver_blocked_ld
    )

    # Old two-node comparison:
    # Role source -> Last receiver
    source_to_last_drop = (
        source_patched_ld
        - source_receiver_blocked_ld
    )

    # ---------------------------------------------------------
    # Return detailed sample-level record
    # ---------------------------------------------------------
    return {
        "clean_ld": clean_ld,
        "corrupted_ld": corrupted_ld,
        "clean_corrupt_gap": clean_corrupt_gap,

        "source_patched_ld": source_patched_ld,
        "source_mediator_blocked_ld":
            source_mediator_blocked_ld,

        "mediator_patched_ld": mediator_patched_ld,
        "mediator_receiver_blocked_ld":
            mediator_receiver_blocked_ld,

        "source_induced_mediator_ld":
            source_induced_mediator_ld,
        "source_induced_mediator_receiver_blocked_ld":
            source_induced_mediator_receiver_blocked_ld,

        "source_receiver_blocked_ld":
            source_receiver_blocked_ld,

        "source_recovery": source_recovery,

        "role_to_inst_drop": role_to_inst_drop,
        "role_to_inst_fraction":
            safe_div(role_to_inst_drop, source_recovery),

        "mediator_clean_recovery":
            mediator_clean_recovery,

        "inst_to_last_drop": inst_to_last_drop,
        "inst_to_last_fraction":
            safe_div(
                inst_to_last_drop,
                mediator_clean_recovery,
            ),

        "source_induced_mediator_recovery":
            source_induced_mediator_recovery,

        "chain_to_last_drop": chain_to_last_drop,
        "chain_to_last_fraction":
            safe_div(
                chain_to_last_drop,
                source_induced_mediator_recovery,
            ),

        "source_to_last_drop":
            source_to_last_drop,
        "source_to_last_fraction":
            safe_div(
                source_to_last_drop,
                source_recovery,
            ),

        "normalized_source_recovery":
            safe_div(source_recovery, clean_corrupt_gap),

        "normalized_role_to_inst_drop":
            safe_div(role_to_inst_drop, clean_corrupt_gap),

        "normalized_mediator_clean_recovery":
            safe_div(
                mediator_clean_recovery,
                clean_corrupt_gap,
            ),

        "normalized_inst_to_last_drop":
            safe_div(inst_to_last_drop, clean_corrupt_gap),

        "normalized_source_induced_mediator_recovery":
            safe_div(
                source_induced_mediator_recovery,
                clean_corrupt_gap,
            ),

        "normalized_chain_to_last_drop":
            safe_div(chain_to_last_drop, clean_corrupt_gap),

        "normalized_source_to_last_drop":
            safe_div(source_to_last_drop, clean_corrupt_gap),

        "clean_prob_diff": clean_prob_diff,
        "corrupted_prob_diff": corrupted_prob_diff,

        "source_patched_prob_diff":
            get_pd(source_patched_logits),

        "source_mediator_blocked_prob_diff":
            get_pd(source_mediator_blocked_logits),

        "mediator_patched_prob_diff":
            get_pd(mediator_patched_logits),

        "mediator_receiver_blocked_prob_diff":
            get_pd(mediator_receiver_blocked_logits),

        "source_induced_mediator_prob_diff":
            get_pd(source_induced_mediator_logits),

        "source_induced_mediator_receiver_blocked_prob_diff":
            get_pd(
                source_induced_mediator_receiver_blocked_logits
            ),

        "source_receiver_blocked_prob_diff":
            get_pd(source_receiver_blocked_logits),

        "clean_source_pos":
            json.dumps(clean_source_pos),

        "corrupt_source_pos":
            json.dumps(corrupt_source_pos),

        "clean_inst_pos":
            json.dumps(clean_inst_pos),

        "corrupt_inst_pos":
            json.dumps(corrupt_inst_pos),

        "clean_last_pos":
            clean_last_pos,

        "corrupt_last_pos":
            corrupt_last_pos,
    }


# Paths are resolved from the checked-out repository, or set via CLI.
CANDIDATE = SCRIPT_DIR / "candidates" / "llama" / "dl20.csv"
RESULT_ROOT = REPO_ROOT / "results" / "pairwise_path_patching" / "llama" / "dl20"
MANIFEST_DIR = RESULT_ROOT / "00_manifest"
RAW_DIR = RESULT_ROOT / "01_raw"
CHAIN_DIR = RESULT_ROOT / "02_chains"
SUMMARY_DIR = RESULT_ROOT / "03_summary"
RUN_STATE = MANIFEST_DIR / "run_state.json"
FINAL_MANIFEST = MANIFEST_DIR / "full_run_manifest.json"

EXPECTED_CANDIDATE_SHA = (
    "c950d30451da91ef92a497e785c9c5d53c171ef73ade340809b6756de5148cea"
)
EXPECTED_QIDS = 54
EXPECTED_ROLE_PAIRS = 10
EXPECTED_PATHS = 68
EXPECTED_UNITS = 680
EXPECTED_SAMPLES = 36720
EXPECTED_FORWARDS = 330480


def configure_paths(args):
    """Keep output paths and shared imports relative to this checkout."""
    global CANDIDATE, RESULT_ROOT, MANIFEST_DIR, RAW_DIR, CHAIN_DIR
    global SUMMARY_DIR, RUN_STATE, FINAL_MANIFEST

    def choose(value, default):
        return Path(value).expanduser().resolve() if value else default

    CANDIDATE = choose(args.candidates_path, CANDIDATE)
    RESULT_ROOT = choose(args.output_dir, RESULT_ROOT)
    MANIFEST_DIR = RESULT_ROOT / "00_manifest"
    RAW_DIR = RESULT_ROOT / "01_raw"
    CHAIN_DIR = RESULT_ROOT / "02_chains"
    SUMMARY_DIR = RESULT_ROOT / "03_summary"
    RUN_STATE = MANIFEST_DIR / "run_state.json"
    FINAL_MANIFEST = MANIFEST_DIR / "full_run_manifest.json"

    role_json = choose(
        args.role_json_path,
        REPO_ROOT / "roles" / "llama" / "selected20" / "pairwise"
        / "llama_dl20_pairwise_selected20_1x1_balanced.json",
    )
    pairs_tsv = choose(
        args.pairs_path,
        REPO_ROOT / "data" / "pairs" / "pointwise" / "dl20.tsv",
    )

    for name, path in (("role JSON", role_json), ("pairs TSV", pairs_tsv)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Missing local model directory: {model_path}")
        os.environ["MODEL_PATH"] = str(model_path)

    os.environ["MODEL_NAME"] = "meta-llama/Llama-3.1-8B-Instruct"
    os.environ["ROLE_JSON_PATH"] = str(role_json)
    os.environ["PAIRS_TSV"] = str(pairs_tsv)


METRIC_COLS = [
    "clean_ld",
    "corrupted_ld",
    "clean_corrupt_gap",

    "source_patched_ld",
    "source_mediator_blocked_ld",

    "mediator_patched_ld",
    "mediator_receiver_blocked_ld",

    "source_induced_mediator_ld",
    "source_induced_mediator_receiver_blocked_ld",

    "source_receiver_blocked_ld",

    "source_recovery",
    "role_to_inst_drop",
    "role_to_inst_fraction",

    "mediator_clean_recovery",
    "inst_to_last_drop",
    "inst_to_last_fraction",

    "source_induced_mediator_recovery",
    "chain_to_last_drop",
    "chain_to_last_fraction",

    "source_to_last_drop",
    "source_to_last_fraction",

    "normalized_source_recovery",
    "normalized_role_to_inst_drop",

    "normalized_mediator_clean_recovery",
    "normalized_inst_to_last_drop",

    "normalized_source_induced_mediator_recovery",
    "normalized_chain_to_last_drop",

    "normalized_source_to_last_drop",
]


# ============================================================
# Generic helpers
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def audit_candidates_file():
    """Validate the fixed candidate set without GPU or third-party packages."""
    if not CANDIDATE.is_file():
        raise FileNotFoundError(CANDIDATE)
    if sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("CAUSAL68 SHA mismatch")

    with CANDIDATE.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "test_doc_source", "source_pos_type", "source_layer", "source_head",
            "mediator_pos_type", "mediator_layer", "mediator_head",
            "receiver_pos_type", "receiver_layer", "receiver_head",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError("Missing Pairwise path columns")
        rows = list(reader)

    if len(rows) != EXPECTED_PATHS:
        raise RuntimeError(f"Expected 68 paths, got {len(rows)}")

    counts = {"relevance": 0, "irrelevance": 0}
    for row in rows:
        condition = row["test_doc_source"]
        if condition not in counts:
            raise RuntimeError(f"Unexpected document condition: {condition}")
        counts[condition] += 1
        if (row["source_pos_type"], row["mediator_pos_type"], row["receiver_pos_type"]) != (
            "role_adj_adv", "inst_all", "last"
        ):
            raise RuntimeError("Position types differ from Pairwise Q-only")
        layers = [int(row[f"{stage}_layer"]) for stage in ("source", "mediator", "receiver")]
        heads = [int(row[f"{stage}_head"]) for stage in ("source", "mediator", "receiver")]
        if not all(0 <= value < 32 for value in layers + heads):
            raise RuntimeError("Candidate layer or head outside Llama range")
        if not layers[0] < layers[1] < layers[2]:
            raise RuntimeError("Candidate violates forward-only layer order")

    if counts != {"relevance": 32, "irrelevance": 36}:
        raise RuntimeError(f"Unexpected candidate split: {counts}")


def atomic_csv(
    df: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".csv",
        dir=str(path.parent),
    )

    os.close(fd)

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
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(
    obj,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".json",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                obj,
                f,
                indent=2,
                ensure_ascii=False,
            )

            f.write("\n")

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp,
            path,
        )

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ============================================================
# Candidate audit
# ============================================================

def load_candidates():

    if not CANDIDATE.is_file():
        raise FileNotFoundError(CANDIDATE)

    actual_sha = sha256(CANDIDATE)

    if actual_sha != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError(
            "CAUSAL68 SHA mismatch"
        )

    df = pd.read_csv(CANDIDATE)

    if len(df) != EXPECTED_PATHS:
        raise RuntimeError(
            f"Expected 68 paths, got {len(df)}"
        )

    counts = (
        df["test_doc_source"]
        .value_counts()
        .to_dict()
    )

    if counts.get("relevance") != 32:
        raise RuntimeError(
            f"Relevance path count mismatch: {counts}"
        )

    if counts.get("irrelevance") != 36:
        raise RuntimeError(
            f"Irrelevance path count mismatch: {counts}"
        )

    if not (
        df["source_pos_type"]
        .eq("role_adj_adv")
        .all()
    ):
        raise RuntimeError(
            "Source position mismatch"
        )

    if not (
        df["mediator_pos_type"]
        .eq("inst_all")
        .all()
    ):
        raise RuntimeError(
            "Mediator position mismatch"
        )

    if not (
        df["receiver_pos_type"]
        .eq("last")
        .all()
    ):
        raise RuntimeError(
            "Receiver position mismatch"
        )

    good_order = (
        (
            df["source_layer"]
            < df["mediator_layer"]
        )
        &
        (
            df["mediator_layer"]
            < df["receiver_layer"]
        )
    )

    if not good_order.all():
        raise RuntimeError(
            "Invalid causal layer order"
        )

    return df


# ============================================================
# Deterministic paths
# ============================================================

def make_chain_name(cand):

    sl = int(cand["source_layer"])
    sh = int(cand["source_head"])

    ml = int(cand["mediator_layer"])
    mh = int(cand["mediator_head"])

    rl = int(cand["receiver_layer"])
    rh = int(cand["receiver_head"])

    return (
        f"L{sl}H{sh}_role"
        f"_to_L{ml}H{mh}_inst"
        f"_to_L{rl}H{rh}_last"
    )


def unit_path(
    chain_id: str,
    condition: str,
    role_pair: int,
):

    return (
        RAW_DIR
        / condition
        / chain_id
        / (
            f"rolepair_{role_pair:02d}"
            f"__role_{role_pair}"
            f"_to_role_{role_pair + 10}.csv"
        )
    )


def chain_samples_path(
    chain_id: str,
    condition: str,
):

    return (
        CHAIN_DIR
        / condition
        / (
            f"{chain_id}"
            "__ALL540.csv"
        )
    )


def chain_summary_path(
    chain_id: str,
    condition: str,
):

    return (
        CHAIN_DIR
        / condition
        / (
            f"{chain_id}"
            "__SUMMARY.csv"
        )
    )


# ============================================================
# Resume validation
# ============================================================

def valid_unit(
    path: Path,
    expected_qids: set[str],
) -> bool:

    if not path.is_file():
        return False

    try:
        df = pd.read_csv(
            path,
            usecols=[
                "qid",
                "role_pair",
                "condition",
            ],
            dtype={
                "qid": str,
            },
        )

    except Exception:
        return False

    if len(df) != EXPECTED_QIDS:
        return False

    if df["qid"].nunique() != EXPECTED_QIDS:
        return False

    if set(df["qid"]) != expected_qids:
        return False

    if df["role_pair"].nunique() != 1:
        return False

    if df["condition"].nunique() != 1:
        return False

    return True


def count_completed_units(
    candidates,
    expected_qids,
):

    n = 0

    for idx, cand in (
        candidates
        .reset_index(drop=True)
        .iterrows()
    ):

        chain_id = (
            f"chain_{idx + 1:03d}"
        )

        condition = str(
            cand["test_doc_source"]
        )

        for role_pair in range(
            1,
            EXPECTED_ROLE_PAIRS + 1,
        ):

            path = unit_path(
                chain_id,
                condition,
                role_pair,
            )

            if valid_unit(
                path,
                expected_qids,
            ):
                n += 1

    return n


# ============================================================
# Summary, preserving old three-stage aggregation semantics
# ============================================================

def build_chain_outputs(
    chain_id,
    condition,
    candidate_group,
    chain_name,
    cand,
    expected_qids,
):

    frames = []

    for role_pair in range(
        1,
        EXPECTED_ROLE_PAIRS + 1,
    ):

        path = unit_path(
            chain_id,
            condition,
            role_pair,
        )

        if not valid_unit(
            path,
            expected_qids,
        ):
            return False

        frames.append(
            pd.read_csv(
                path,
                dtype={
                    "qid": str,
                },
            )
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(df) != (
        EXPECTED_QIDS
        * EXPECTED_ROLE_PAIRS
    ):
        raise RuntimeError(
            f"{chain_id}: expected 540 rows, "
            f"got {len(df)}"
        )

    if df["role_pair"].nunique() != 10:
        raise RuntimeError(
            f"{chain_id}: role-pair count != 10"
        )

    for col in METRIC_COLS:
        if col not in df.columns:
            raise RuntimeError(
                f"{chain_id}: missing metric {col}"
            )

    atomic_csv(
        df,
        chain_samples_path(
            chain_id,
            condition,
        ),
    )

    row = {
        "chain_id": chain_id,
        "chain_name": chain_name,
        "candidate_group": candidate_group,
        "condition": condition,

        "source_pos_type":
            "role_adj_adv",

        "source_layer":
            int(cand["source_layer"]),

        "source_head":
            int(cand["source_head"]),

        "mediator_pos_type":
            "inst_all",

        "mediator_layer":
            int(cand["mediator_layer"]),

        "mediator_head":
            int(cand["mediator_head"]),

        "receiver_pos_type":
            "last",

        "receiver_layer":
            int(cand["receiver_layer"]),

        "receiver_head":
            int(cand["receiver_head"]),

        "n": len(df),
    }

    for col in METRIC_COLS:
        row[col] = float(
            df[col].mean()
        )

    # --------------------------------------------------------
    # Same summary convention as the validated old runner:
    # ratio of aggregate means, NOT mean of sample ratios.
    # --------------------------------------------------------

    row["role_to_inst_fraction"] = (
        safe_div(
            row["role_to_inst_drop"],
            row["source_recovery"],
        )
    )

    row["inst_to_last_fraction"] = (
        safe_div(
            row["inst_to_last_drop"],
            row["mediator_clean_recovery"],
        )
    )

    row["chain_to_last_fraction"] = (
        safe_div(
            row["chain_to_last_drop"],
            row[
                "source_induced_mediator_recovery"
            ],
        )
    )

    row["source_to_last_fraction"] = (
        safe_div(
            row["source_to_last_drop"],
            row["source_recovery"],
        )
    )

    row["normalized_source_recovery"] = (
        safe_div(
            row["source_recovery"],
            row["clean_corrupt_gap"],
        )
    )

    row["normalized_role_to_inst_drop"] = (
        safe_div(
            row["role_to_inst_drop"],
            row["clean_corrupt_gap"],
        )
    )

    row[
        "normalized_mediator_clean_recovery"
    ] = (
        safe_div(
            row["mediator_clean_recovery"],
            row["clean_corrupt_gap"],
        )
    )

    row["normalized_inst_to_last_drop"] = (
        safe_div(
            row["inst_to_last_drop"],
            row["clean_corrupt_gap"],
        )
    )

    row[
        "normalized_source_induced_mediator_recovery"
    ] = (
        safe_div(
            row[
                "source_induced_mediator_recovery"
            ],
            row["clean_corrupt_gap"],
        )
    )

    row["normalized_chain_to_last_drop"] = (
        safe_div(
            row["chain_to_last_drop"],
            row["clean_corrupt_gap"],
        )
    )

    row["normalized_source_to_last_drop"] = (
        safe_div(
            row["source_to_last_drop"],
            row["clean_corrupt_gap"],
        )
    )

    atomic_csv(
        pd.DataFrame([row]),
        chain_summary_path(
            chain_id,
            condition,
        ),
    )

    return True


def build_global_summary(
    candidates,
):

    rows = []

    for idx, cand in (
        candidates
        .reset_index(drop=True)
        .iterrows()
    ):

        chain_id = (
            f"chain_{idx + 1:03d}"
        )

        condition = str(
            cand["test_doc_source"]
        )

        path = chain_summary_path(
            chain_id,
            condition,
        )

        if not path.is_file():
            continue

        df = pd.read_csv(path)

        if len(df) != 1:
            raise RuntimeError(
                f"Invalid chain summary: {path}"
            )

        rows.append(df.iloc[0].to_dict())

    if not rows:
        return 0

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "chain_id"
    ).reset_index(
        drop=True
    )

    atomic_csv(
        summary,
        SUMMARY_DIR
        / "SUMMARY_chains__three_stage_role_inst_last.csv",
    )

    return len(summary)


# ============================================================
# Main
# ============================================================

def main():
    global base, get_pos, logit_diff, prob_diff, torch, pd

    parser = argparse.ArgumentParser(
        description=(
            "Pairwise Llama/DL20 Q-only path patching: "
            "role_adj_adv -> inst_all -> last (68 paths, 10 role pairs)"
        )
    )
    parser.add_argument(
        "--model-path",
        help="Local Llama-3.1-8B-Instruct model directory (required for run)",
    )
    parser.add_argument("--pairs-path", help="Override DL20 pair TSV")
    parser.add_argument("--role-json-path", help="Override selected20 pairwise roles")
    parser.add_argument("--candidates-path", help="Override fixed CAUSAL68 CSV")
    parser.add_argument("--output-dir", help="Output directory outside tracked code")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Check candidate hash and input files without loading a model",
    )

    args = parser.parse_args()
    configure_paths(args)

    if args.audit_only:
        audit_candidates_file()
        print("STATIC INPUT AUDIT: PASS (68 fixed Pairwise Q-only paths)")
        return

    if not args.model_path:
        parser.error("--model-path is required unless --audit-only is used")

    import pandas as pd
    import torch
    from transformer_lens import utils
    from methods import get_pos, logit_diff, prob_diff

    candidates = load_candidates()

    base = SimpleNamespace(
        ensure_list_pos=ensure_list_pos,
        get_pos_from_ranges=get_pos_from_ranges,
        utils=utils,
        logit_diff=logit_diff,
        prob_diff=prob_diff,
    )

    # The shared Pairwise core reads these environment variables at import time.
    # Our local `methods.py` remains first on sys.path so its source positions
    # match the Q-only experiment that generated the candidate set.
    import pairwise_core as core
    from pairwise_activation_ranker import PairwiseActivationPatchingRanker

    torch.set_grad_enabled(False)

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHAIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "LLAMA DL20 PAIRWISE THREE-STAGE PATH PATCHING"
    )
    print("=" * 100)

    print("[PASS] candidate SHA")
    print(
        "[SCOPE] Llama / DL20 / Pairwise only"
    )
    print(
        "[PATHS] 68 = 32 relevance + 36 irrelevance"
    )
    print(
        "[POSITIONS] role_adj_adv -> inst_all -> last"
    )
    print(
        "[EXPECTED] units=680 samples=36720 forwards=330480"
    )

    # --------------------------------------------------------
    # Load the exact completed Pairwise model implementation.
    # --------------------------------------------------------

    pkg = core.build_prompt_package("role_1")
    if pkg[0] != "output_2" or pkg[3] != EXPECTED_INST:
        raise RuntimeError("Pairwise Q-only prompt or instruction mismatch")
    if not pkg[1].endswith(EXPECTED_PROMPT_TAIL):
        raise RuntimeError("Unexpected Pairwise Yes/No prompt tail")
    if not all(marker in pkg[1] for marker in ("Passage A:", "Passage B:", "Query:")):
        raise RuntimeError("Pairwise prompt is missing a passage or query")

    print()
    print("[LOAD] Pairwise Llama model")

    ranker = (
        PairwiseActivationPatchingRanker()
    )

    if ranker.yes_id != 9642:
        raise RuntimeError(
            f"Runtime Yes mismatch: "
            f"{ranker.yes_id}"
        )

    if ranker.no_id != 2822:
        raise RuntimeError(
            f"Runtime No mismatch: "
            f"{ranker.no_id}"
        )

    if ranker.model.cfg.n_layers != 32:
        raise RuntimeError(
            "Expected 32 layers"
        )

    if ranker.model.cfg.n_heads != 32:
        raise RuntimeError(
            "Expected 32 heads"
        )

    print(
        "[PASS] runtime model: "
        "32 layers / 32 heads / "
        "Yes=9642 / No=2822"
    )

    # --------------------------------------------------------
    # Exact controlled 54 DL20 examples.
    # --------------------------------------------------------

    examples = core.load_examples(
        ranker.tokenizer
    )

    if len(examples) != 54:
        raise RuntimeError(
            f"Expected 54 examples, got "
            f"{len(examples)}"
        )

    expected_qids = {
        str(x["qid"])
        for x in examples
    }

    if len(expected_qids) != 54:
        raise RuntimeError(
            "Expected 54 unique qids"
        )

    if any(int(x["neg_label"]) != 0 or int(x["pos_label"]) <= 0 for x in examples):
        raise RuntimeError("DL20 pair labels do not match the completed run")

    # --------------------------------------------------------
    # Build Pairwise prompt/token/range loaders ONCE.
    #
    # 2 conditions × 10 role pairs.
    # Verify positions in all 1080 cases instead of requiring a stored
    # preflight certificate from the university server.
    # --------------------------------------------------------

    print(
        "[CACHE] building 20 validated "
        "condition/role-pair loaders"
    )

    loader_cache = {}
    cases_checked = 0

    for condition in (
        "relevance",
        "irrelevance",
    ):

        correct_id, wrong_id = ranker._correct_wrong_ids(condition)
        if (correct_id, wrong_id) != (
            (ranker.yes_id, ranker.no_id)
            if condition == "relevance"
            else (ranker.no_id, ranker.yes_id)
        ):
            raise RuntimeError(f"Incorrect Yes/No direction for {condition}")

        for role_pair in range(
            1,
            11,
        ):

            package = (
                ranker._build_dataloaders(
                    examples,
                    role_pair,
                    condition,
                )
            )

            (
                clean_loader,
                corrupt_loader,
                sample_meta,
                clean_role,
                corrupt_role,
            ) = package

            if not (
                len(clean_loader)
                == len(corrupt_loader)
                == len(sample_meta)
                == 54
            ):
                raise RuntimeError(
                    "Runtime loader length mismatch"
                )

            if clean_role != (
                f"role_{role_pair}"
            ):
                raise RuntimeError(
                    "Runtime clean-role mismatch"
                )

            if corrupt_role != (
                f"role_{role_pair + 10}"
            ):
                raise RuntimeError(
                    "Runtime corrupt-role mismatch"
                )

            for clean, corrupt, meta in zip(clean_loader, corrupt_loader, sample_meta):
                clean_tokens, corrupt_tokens = clean[0], corrupt[0]
                if clean_tokens.shape != corrupt_tokens.shape:
                    raise RuntimeError(f"Unequal clean/corrupt tokens: {meta['qid']}")
                clean_ranges, corrupt_ranges = clean[1:], corrupt[1:]
                seq_len = int(clean_tokens.shape[-1])
                if seq_len >= core.MAX_LENGTH:
                    raise RuntimeError(f"Unexpected token truncation: {meta['qid']}")

                for ranges in (clean_ranges, corrupt_ranges):
                    role_range, inst_range = ranges[0], ranges[3]
                    source_pos = ensure_list_pos(
                        get_pos_from_ranges(seq_len, "role_adj_adv", role_range)
                    )
                    inst_pos = span_positions(inst_range)
                    if not source_pos or not inst_pos:
                        raise RuntimeError(f"Empty intervention span: {meta['qid']}")
                    if not (0 <= min(source_pos) and max(source_pos) < min(inst_pos)
                            and max(inst_pos) < seq_len - 1):
                        raise RuntimeError(f"Invalid Q-only position order: {meta['qid']}")
                    # Selected Llama adjectives and adverbs are single tokens;
                    # this is when terminal positions equal their full spans.
                    for key in ("adj", "adv"):
                        start, end = core.role_span(role_range, key)
                        if end - start != 1:
                            raise RuntimeError(
                                f"Expected one-token {key} in selected role: {meta['qid']}"
                            )

                clean_source = ensure_list_pos(
                    get_pos_from_ranges(seq_len, "role_adj_adv", clean_ranges[0])
                )
                corrupt_source = ensure_list_pos(
                    get_pos_from_ranges(seq_len, "role_adj_adv", corrupt_ranges[0])
                )
                if len(clean_source) != len(corrupt_source):
                    raise RuntimeError(f"Source position count mismatch: {meta['qid']}")
                if len(span_positions(clean_ranges[3])) != len(
                    span_positions(corrupt_ranges[3])
                ):
                    raise RuntimeError(f"Instruction span length mismatch: {meta['qid']}")

                if condition == "relevance":
                    expected_a, expected_b = meta["pos_docid"], meta["neg_docid"]
                    valid_labels = int(meta["passage_a_label"]) > 0 and int(meta["passage_b_label"]) == 0
                else:
                    expected_a, expected_b = meta["neg_docid"], meta["pos_docid"]
                    valid_labels = int(meta["passage_a_label"]) == 0 and int(meta["passage_b_label"]) > 0
                if (str(meta["passage_a_docid"]) != str(expected_a)
                        or str(meta["passage_b_docid"]) != str(expected_b)
                        or not valid_labels):
                    raise RuntimeError(f"Pairwise passage order/label mismatch: {meta['qid']}")
                cases_checked += 1

            loader_cache[
                (
                    condition,
                    role_pair,
                )
            ] = package

    if cases_checked != 1080:
        raise RuntimeError(f"Expected 1080 token/range cases; got {cases_checked}")
    print("[PASS] 1080 Pairwise Q-only token/range cases")

    # --------------------------------------------------------
    # Resume audit.
    # --------------------------------------------------------

    completed_units = (
        count_completed_units(
            candidates,
            expected_qids,
        )
    )

    print(
        f"[RESUME] complete units "
        f"{completed_units}/{EXPECTED_UNITS}"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # 68 paths × 10 role pairs.
    # --------------------------------------------------------

    for candidate_index, cand in (
        candidates
        .reset_index(drop=True)
        .iterrows()
    ):

        chain_id = (
            f"chain_{candidate_index + 1:03d}"
        )

        candidate_group = str(
            cand["candidate_group"]
        )

        condition = str(
            cand["test_doc_source"]
        )

        source_layer = int(
            cand["source_layer"]
        )

        source_head = int(
            cand["source_head"]
        )

        mediator_layer = int(
            cand["mediator_layer"]
        )

        mediator_head = int(
            cand["mediator_head"]
        )

        receiver_layer = int(
            cand["receiver_layer"]
        )

        receiver_head = int(
            cand["receiver_head"]
        )

        if not (
            source_layer
            < mediator_layer
            < receiver_layer
        ):
            raise RuntimeError(
                f"{chain_id}: invalid layer order"
            )

        chain_name = make_chain_name(
            cand
        )

        print()
        print("=" * 100)

        print(
            f"[{candidate_index + 1}/68] "
            f"{chain_id} | "
            f"{condition} | "
            f"{chain_name}"
        )

        print("=" * 100)

        correct_id, wrong_id = (
            ranker._correct_wrong_ids(
                condition
            )
        )

        for role_pair in range(
            1,
            EXPECTED_ROLE_PAIRS + 1,
        ):

            out = unit_path(
                chain_id,
                condition,
                role_pair,
            )

            if valid_unit(
                out,
                expected_qids,
            ):

                print(
                    f"[SKIP] {chain_id} "
                    f"rolepair={role_pair:02d} "
                    f"already complete"
                )

                continue

            (
                clean_loader,
                corrupt_loader,
                sample_meta,
                clean_role,
                corrupt_role,
            ) = loader_cache[
                (
                    condition,
                    role_pair,
                )
            ]

            source_pair_id = int(
                core.SOURCE_PAIR_IDS[
                    role_pair - 1
                ]
            )

            rows = []

            print(
                f"[RUN] {chain_id} "
                f"{condition} "
                f"{clean_role}->{corrupt_role} "
                f"source_pair_id={source_pair_id}"
            )

            for i in range(
                EXPECTED_QIDS
            ):

                clean = clean_loader[i]
                corrupt = corrupt_loader[i]
                meta = sample_meta[i]

                clean_tokens = (
                    clean[0]
                    .to(
                        "cuda",
                        non_blocking=True,
                    )
                )

                corrupt_tokens = (
                    corrupt[0]
                    .to(
                        "cuda",
                        non_blocking=True,
                    )
                )

                clean_ranges = clean[1:]
                corrupt_ranges = corrupt[1:]

                metrics = run_one_example(
                    model=ranker.model,

                    clean_tokens=
                        clean_tokens,

                    corrupt_tokens=
                        corrupt_tokens,

                    clean_ranges=
                        clean_ranges,

                    corrupt_ranges=
                        corrupt_ranges,

                    source_layer=
                        source_layer,

                    source_head=
                        source_head,

                    mediator_layer=
                        mediator_layer,

                    mediator_head=
                        mediator_head,

                    receiver_layer=
                        receiver_layer,

                    receiver_head=
                        receiver_head,

                    correct_token_id=
                        correct_id,

                    wrong_token_id=
                        wrong_id,
                )

                row = {
                    "chain_id":
                        chain_id,

                    "chain_name":
                        chain_name,

                    "candidate_group":
                        candidate_group,

                    "condition":
                        condition,

                    "source_pos_type":
                        "role_adj_adv",

                    "source_layer":
                        source_layer,

                    "source_head":
                        source_head,

                    "mediator_pos_type":
                        "inst_all",

                    "mediator_layer":
                        mediator_layer,

                    "mediator_head":
                        mediator_head,

                    "receiver_pos_type":
                        "last",

                    "receiver_layer":
                        receiver_layer,

                    "receiver_head":
                        receiver_head,

                    "role_pair":
                        role_pair,

                    "source_pair_id":
                        source_pair_id,

                    "clean_role":
                        clean_role,

                    "corrupt_role":
                        corrupt_role,

                    "sample":
                        int(meta["sample"]),

                    "qid":
                        str(meta["qid"]),

                    "pos_docid":
                        str(meta["pos_docid"]),

                    "neg_docid":
                        str(meta["neg_docid"]),

                    "pos_label":
                        int(meta["pos_label"]),

                    "neg_label":
                        int(meta["neg_label"]),

                    "passage_a_docid":
                        str(
                            meta[
                                "passage_a_docid"
                            ]
                        ),

                    "passage_b_docid":
                        str(
                            meta[
                                "passage_b_docid"
                            ]
                        ),

                    "passage_a_label":
                        int(
                            meta[
                                "passage_a_label"
                            ]
                        ),

                    "passage_b_label":
                        int(
                            meta[
                                "passage_b_label"
                            ]
                        ),

                    "correct_token_id":
                        int(correct_id),

                    "wrong_token_id":
                        int(wrong_id),

                    **metrics,
                }

                rows.append(row)

                if (
                    (i + 1) % 10 == 0
                    or i == 53
                ):

                    print(
                        f"    qids "
                        f"{i + 1}/54",
                        flush=True,
                    )

                del clean_tokens
                del corrupt_tokens

            unit_df = pd.DataFrame(
                rows
            )

            if len(unit_df) != 54:
                raise RuntimeError(
                    f"{chain_id}: "
                    f"unit produced "
                    f"{len(unit_df)} rows"
                )

            if (
                unit_df["qid"]
                .astype(str)
                .nunique()
                != 54
            ):
                raise RuntimeError(
                    f"{chain_id}: duplicate qids"
                )

            if set(
                unit_df["qid"]
                .astype(str)
            ) != expected_qids:

                raise RuntimeError(
                    f"{chain_id}: qid set mismatch"
                )

            atomic_csv(
                unit_df,
                out,
            )

            if not valid_unit(
                out,
                expected_qids,
            ):
                raise RuntimeError(
                    f"Saved unit failed "
                    f"validation: {out}"
                )

            completed_units += 1

            elapsed = (
                time.time()
                - start_time
            )

            atomic_json(
                {
                    "status":
                        "RUNNING",

                    "completed_units":
                        completed_units,

                    "expected_units":
                        EXPECTED_UNITS,

                    "current_chain":
                        chain_id,

                    "current_condition":
                        condition,

                    "current_role_pair":
                        role_pair,

                    "elapsed_seconds_this_job":
                        elapsed,
                },
                RUN_STATE,
            )

            print(
                f"[SAVED] {out}"
            )

            print(
                f"[PROGRESS] units "
                f"{completed_units}/"
                f"{EXPECTED_UNITS}",
                flush=True,
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Once all 10 role-pair units exist for this path,
        # create its 540-row combined file and one-row summary.
        # ----------------------------------------------------

        chain_complete = (
            build_chain_outputs(
                chain_id=chain_id,
                condition=condition,
                candidate_group=
                    candidate_group,
                chain_name=chain_name,
                cand=cand,
                expected_qids=
                    expected_qids,
            )
        )

        if chain_complete:

            print(
                f"[CHAIN COMPLETE] "
                f"{chain_id} "
                f"540 samples"
            )

            n_summaries = (
                build_global_summary(
                    candidates
                )
            )

            print(
                f"[SUMMARY] "
                f"{n_summaries}/68 "
                f"chain summaries"
            )

    # --------------------------------------------------------
    # Final global audit
    # --------------------------------------------------------

    completed_units = (
        count_completed_units(
            candidates,
            expected_qids,
        )
    )

    summary_count = (
        build_global_summary(
            candidates
        )
    )

    if completed_units != 680:
        raise RuntimeError(
            "FULL run ended but only "
            f"{completed_units}/680 units "
            "are complete"
        )

    if summary_count != 68:
        raise RuntimeError(
            "FULL run ended but only "
            f"{summary_count}/68 chain "
            "summaries exist"
        )

    final_summary = (
        SUMMARY_DIR
        / "SUMMARY_chains__three_stage_role_inst_last.csv"
    )

    summary_df = pd.read_csv(
        final_summary
    )

    if len(summary_df) != 68:
        raise RuntimeError(
            "Final summary row count != 68"
        )

    manifest = {
        "status":
            "COMPLETE",

        "scope":
            "Llama / DL20 / Pairwise",

        "paths":
            68,

        "relevance_paths":
            32,

        "irrelevance_paths":
            36,

        "role_pairs":
            10,

        "qids":
            54,

        "units":
            680,

        "samples":
            36720,

        "expected_forward_passes":
            330480,

        "yes_id":
            int(ranker.yes_id),

        "no_id":
            int(ranker.no_id),

        "candidate_sha256":
            sha256(CANDIDATE),

        "summary":
            str(final_summary),

        "elapsed_seconds_this_job":
            time.time()
            - start_time,
    }

    atomic_json(
        manifest,
        FINAL_MANIFEST,
    )

    atomic_json(
        {
            "status":
                "COMPLETE",

            "completed_units":
                680,

            "expected_units":
                680,

            "completed_chains":
                68,

            "expected_chains":
                68,
        },
        RUN_STATE,
    )

    print()
    print("=" * 100)
    print(
        "PAIRWISE THREE-STAGE FULL AUDIT: PASS"
    )
    print("=" * 100)

    print(
        "scope             = "
        "Llama / DL20 / Pairwise"
    )

    print(
        "paths             = 68/68"
    )

    print(
        "units             = 680/680"
    )

    print(
        "samples           = 36720"
    )

    print(
        "expected forwards = 330480"
    )

    print(
        "Yes / No          = "
        f"{ranker.yes_id} / "
        f"{ranker.no_id}"
    )

    print(
        "summary           =",
        final_summary,
    )

    print(
        "manifest          =",
        FINAL_MANIFEST,
    )

    print(
        "STATUS            = COMPLETE"
    )


if __name__ == "__main__":
    main()
