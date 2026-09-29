#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared helpers for pointwise three-stage path patching.

The released implementation evaluates candidate causal routes of the form

    Role (adjective + adverb) -> Instruction -> Last

across Llama, Qwen, and Mistral.

Model-specific selected-role files are supplied through ROLE_JSON_PATH.
Role adjective/adverb positions use their complete token spans.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
from tqdm import tqdm


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer, utils

from ranker import SearchResult
from pointwise_ranker import prompt_generator, get_pointwise_ranges
from methods import logit_diff, prob_diff


ROLE_PAIRS = [
    ("role_1", "role_11"),
    ("role_2", "role_12"),
    ("role_3", "role_13"),
    ("role_4", "role_14"),
    ("role_5", "role_15"),
    ("role_6", "role_16"),
    ("role_7", "role_17"),
    ("role_8", "role_18"),
    ("role_9", "role_19"),
    ("role_10", "role_20"),
]


def load_pairs_tsv(path: Path, doc_source: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"qid", "pos_doc", "pos_label", "neg_doc", "neg_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    rows = []
    for _, r in df.iterrows():
        qid = str(r["qid"])
        if doc_source == "relevance":
            rows.append({
                "qid": qid,
                "docid": str(r["pos_doc"]),
                "label": int(r["pos_label"]),
            })
        elif doc_source == "irrelevance":
            rows.append({
                "qid": qid,
                "docid": str(r["neg_doc"]),
                "label": int(r["neg_label"]),
            })
        else:
            raise ValueError(f"Unsupported doc_source={doc_source}")
    return pd.DataFrame(rows)


def load_role_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        roles = json.load(f)
    return roles


def get_role_entry(roles, role_name):
    """
    Robust role lookup.

    Supports:
    1) flat JSON:
        {"role_1": "...", "role_11": "..."}

    2) nested JSON with global role IDs:
        {"pos_roles": {"role_1": "..."}, "neg_roles": {"role_11": "..."}}

    3) nested JSON with local role IDs:
        {"pos_roles": {"role_1": "..."}, "neg_roles": {"role_1": "..."}}

    In case (3), global negative role_11 is mapped to neg_roles role_1,
    role_12 -> neg_roles role_2, ..., role_20 -> neg_roles role_10.
    """
    role_name = str(role_name)
    role_num = int(role_name.replace("role_", ""))

    # 1) Flat dictionary
    if isinstance(roles, dict) and role_name in roles:
        return roles[role_name]

    if not isinstance(roles, dict):
        raise KeyError(f"Cannot find {role_name}; role json is not dict: {type(roles)}")

    # Decide likely polarity and local role name
    if 1 <= role_num <= 10:
        preferred_groups = ["pos_roles", "positive_roles"]
        local_role_name = f"role_{role_num}"
    elif 11 <= role_num <= 20:
        preferred_groups = ["neg_roles", "negative_roles"]
        local_role_name = f"role_{role_num - 10}"
    else:
        preferred_groups = ["roles", "pos_roles", "neg_roles", "positive_roles", "negative_roles"]
        local_role_name = role_name

    # Search preferred groups first
    for group_name in preferred_groups + ["roles", "pos_roles", "neg_roles", "positive_roles", "negative_roles"]:
        group = roles.get(group_name)
        if group is None:
            continue

        if isinstance(group, dict):
            candidate_keys = [
                role_name,                 # global key, e.g. role_11
                local_role_name,           # local key, e.g. role_1 inside neg_roles
                role_name.replace("role_", ""),
                local_role_name.replace("role_", ""),
            ]

            for k in candidate_keys:
                if k in group:
                    return group[k]

            # robust string comparison
            for k, v in group.items():
                ks = str(k)
                if ks in candidate_keys:
                    return v
                if ks.replace("role_", "") in [str(x).replace("role_", "") for x in candidate_keys]:
                    return v

        elif isinstance(group, list):
            local_num = local_role_name.replace("role_", "")
            global_num = role_name.replace("role_", "")

            for entry in group:
                if not isinstance(entry, dict):
                    continue
                candidates = [
                    entry.get("role"),
                    entry.get("role_id"),
                    entry.get("role_name"),
                    entry.get("name"),
                    entry.get("id"),
                ]
                cand_strs = [str(x) for x in candidates if x is not None]
                cand_nums = [s.replace("role_", "") for s in cand_strs]

                if role_name in cand_strs or local_role_name in cand_strs:
                    return entry
                if global_num in cand_nums or local_num in cand_nums:
                    return entry

    raise KeyError(
        f"Cannot find {role_name}. "
        f"Top-level keys={list(roles.keys())[:20]}. "
        f"Tried local_role_name={local_role_name}."
    )
def role_to_prompt_pkg(roles, role_name):
    # Use the adjusted pointwise prompt shared by the released
    # path-patching implementation.
    #
    # ROLE_JSON_PATH points to the model-specific selected20 role JSON.
    return prompt_generator(
        prompt_type="adjusted",
        original_prompt_number=1,
        instruction="instruction_1",
        output="output_5",
        tone="tone_1",
        order="query_first",
        position="beginning",
        role=role_name,
        query_before_instruction="True",
        role_at_beginning="True",
        experiment_type="activation_patching",
    )


def load_queries_docs_from_ir_datasets(dataset_name: str, needed: pd.DataFrame):
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    qids = set(needed["qid"].astype(str))
    docids = set(needed["docid"].astype(str))

    queries = {}
    for q in dataset.queries_iter():
        qid = str(q.query_id)
        if qid in qids:
            queries[qid] = q.text

    docs = {}
    for d in dataset.docs_iter():
        docid = str(d.doc_id)
        if docid in docids:
            text = getattr(d, "text", None)
            if text is None:
                text = getattr(d, "body", "")
            docs[docid] = text

    missing_q = qids - set(queries)
    missing_d = docids - set(docs)
    if missing_q:
        raise RuntimeError(f"Missing queries: {sorted(list(missing_q))[:10]} ... total={len(missing_q)}")
    if missing_d:
        raise RuntimeError(f"Missing docs: {sorted(list(missing_d))[:10]} ... total={len(missing_d)}")

    return queries, docs


def get_pos_from_ranges(seqlen, pos_type, role_range):
    def role_span(rr, key="all"):
        if isinstance(rr, dict):
            return rr[key][0]
        return rr[0]

    if pos_type == "last":
        return seqlen - 1

    if pos_type == "role_adj":
        a, b = role_span(role_range, "adj")
        return list(range(a, b))

    if pos_type == "role_adv":
        a, b = role_span(role_range, "adv")
        return list(range(a, b))

    if pos_type == "role_adj_adv":
        aa, ab = role_span(role_range, "adj")
        va, vb = role_span(role_range, "adv")

        positions = (
            list(range(aa, ab))
            + list(range(va, vb))
        )

        return sorted(set(positions))

    raise ValueError(
        f"Unsupported source_pos_type={pos_type}"
    )


def ensure_list_pos(pos):
    if isinstance(pos, list):
        return pos
    return [pos]
