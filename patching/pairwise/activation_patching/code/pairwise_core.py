from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

ROLE_JSON = Path(
    os.environ.get(
        "ROLE_JSON_PATH",
        str(
            REPO_ROOT
            / "roles"
            / "llama"
            / "selected20"
            / "pairwise"
            / "llama_dl20_pairwise_selected20_1x1_balanced.json"
        ),
    )
)

PAIRS_TSV = Path(
    os.environ.get(
        "PAIRS_TSV",
        str(
            REPO_ROOT
            / "data"
            / "pairs"
            / "pointwise"
            / "dl20.tsv"
        ),
    )
)

_MODEL_PATH_ENV = os.environ.get("MODEL_PATH")
if not _MODEL_PATH_ENV:
    raise RuntimeError(
        "MODEL_PATH is not set. "
        "Set it through the pairwise activation-patching launcher."
    )

MODEL_PATH = Path(_MODEL_PATH_ENV).expanduser()

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct",
)

IR_DATASET = "msmarco-passage/trec-dl-2020/judged"

QUERY_LENGTH = 20
PASSAGE_LENGTH = 80
MAX_LENGTH = 2048

EXPECTED_QIDS = 54
EXPECTED_ROLE_PAIRS = 10

# Must match the completed formal Llama Pairwise Yes/No run.
EXPECTED_YES_ID = 9642
EXPECTED_NO_ID = 2822

# Original FULLCROSS225 pair_ids selected for the new patching role set.
SOURCE_PAIR_IDS = [
    19,
    33,
    1,
    49,
    63,
    61,
    17,
    47,
    5,
    35,
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

HEAD_TARGETS = [
    "role_all",
    "role_adj",
    "role_adv",
    "role_adj_adv",
    "inst_all",
    "last",
]

CONDITIONS = [
    "relevance",
    "irrelevance",
]


# Reuse the exact formal pairwise ranking prompt implementation
# shipped with this artifact.
from pairwise_ranker import prompt_generator as formal_prompt_generator


# Frozen activation-patching support copied from patching/code/.
from data_structure import (
    _make_len_getter,
    _build_role_range_dict,
    _char_span_to_token_span,
)

from methods import get_pos


def sha256(path):
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require_file(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def install_role_json():
    require_file(ROLE_JSON)
    os.environ["ROLE_JSON_PATH"] = str(ROLE_JSON)


def build_prompt_package(role_name):
    """
    Build the formal Llama pairwise prompt package.

    The instruction span used by activation patching is the
    span returned directly by the formal prompt generator.
    """
    install_role_json()

    return formal_prompt_generator(
        "adjusted",
        1,
        "instruction_2",
        "output_2",
        "tone_0",
        "passage_first",
        "beginning",
        role_name,
        "True",
        "True",
        "zero_shot_ranking",
    )


def truncate(tokenizer, text, length):
    """
    Same truncation convention as formal PairwiseLlmRanker.truncate().
    """

    return tokenizer.convert_tokens_to_string(
        tokenizer.tokenize(str(text))[:length]
    )


def _single_token_id(tokenizer, candidates):
    for text in candidates:

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) == 1:
            return int(ids[0]), text

    raise RuntimeError(
        f"No single-token candidate among {candidates}"
    )


def choose_yesno_ids(tokenizer):
    """
    Must mirror the candidate order in the completed formal Pairwise
    implementation.
    """

    yes_id, yes_text = _single_token_id(
        tokenizer,
        [
            "Yes",
            " Yes",
            "yes",
            " yes",
        ],
    )

    no_id, no_text = _single_token_id(
        tokenizer,
        [
            "No",
            " No",
            "no",
            " no",
        ],
    )

    return (
        yes_id,
        no_id,
        yes_text,
        no_text,
    )


def chat_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    if (
        len(ids) > 0
        and isinstance(ids[0], list)
    ):
        ids = ids[0]

    return list(ids)


def chat_tensor(tokenizer, prompt):
    """
    Same chat-template convention as existing activation patching.
    """

    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
        truncation=True,
        add_generation_prompt=True,
    )


def get_pairwise_ranges_exact(
    tokenizer,
    prompt,
    role,
    query,
    document1,
    document2,
    inst,
    use_chat=True,
):
    """
    Exact Pairwise counterpart of get_pointwise_ranges().

    doc_range is deliberately:

        [
            Passage-A document-text range,
            Passage-B document-text range,
        ]

    It does NOT make one continuous span from A through B.
    """

    get_length = _make_len_getter(
        tokenizer,
        use_chat,
    )

    role_range = _build_role_range_dict(
        tokenizer,
        prompt,
        role,
        use_chat,
    )

    # --------------------------------------------------------
    # Role
    # --------------------------------------------------------

    role_start = prompt.find(role)

    if role_start < 0:
        raise RuntimeError(
            "Role text not found in Pairwise prompt"
        )

    role_end = role_start + len(role)

    # --------------------------------------------------------
    # Passage A document text
    # --------------------------------------------------------

    marker_a = "Passage A: "

    marker_a_start = prompt.find(
        marker_a,
        role_end,
    )

    if marker_a_start < 0:
        raise RuntimeError(
            "Passage A marker not found"
        )

    doc1_char_start = (
        marker_a_start
        + len(marker_a)
    )

    doc1_char_end = (
        doc1_char_start
        + len(document1)
    )

    actual_doc1 = prompt[
        doc1_char_start:doc1_char_end
    ]

    if actual_doc1 != document1:
        raise RuntimeError(
            "Passage A document-text mismatch"
        )

    # --------------------------------------------------------
    # Passage B document text
    # --------------------------------------------------------

    marker_b = "\nPassage B: "

    marker_b_start = prompt.find(
        marker_b,
        doc1_char_end,
    )

    if marker_b_start < 0:
        raise RuntimeError(
            "Passage B marker not found"
        )

    doc2_char_start = (
        marker_b_start
        + len(marker_b)
    )

    doc2_char_end = (
        doc2_char_start
        + len(document2)
    )

    actual_doc2 = prompt[
        doc2_char_start:doc2_char_end
    ]

    if actual_doc2 != document2:
        raise RuntimeError(
            "Passage B document-text mismatch"
        )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    marker_q = "\nQuery: "

    marker_q_start = prompt.find(
        marker_q,
        doc2_char_end,
    )

    if marker_q_start < 0:
        raise RuntimeError(
            "Query marker not found"
        )

    query_char_start = (
        marker_q_start
        + len(marker_q)
    )

    query_char_end = (
        query_char_start
        + len(query)
    )

    actual_query = prompt[
        query_char_start:query_char_end
    ]

    if actual_query != query:
        raise RuntimeError(
            "Query text mismatch"
        )

    # --------------------------------------------------------
    # Instruction
    # --------------------------------------------------------

    inst_char_start = prompt.find(
        inst,
        query_char_end,
    )

    if inst_char_start < 0:
        raise RuntimeError(
            "Pairwise instruction not found"
        )

    inst_char_end = (
        inst_char_start
        + len(inst)
    )

    # --------------------------------------------------------
    # Convert char spans to EXACT chat-token spans.
    # --------------------------------------------------------

    doc1_span = _char_span_to_token_span(
        prompt,
        doc1_char_start,
        doc1_char_end,
        get_length,
    )

    doc2_span = _char_span_to_token_span(
        prompt,
        doc2_char_start,
        doc2_char_end,
        get_length,
    )

    query_span = _char_span_to_token_span(
        prompt,
        query_char_start,
        query_char_end,
        get_length,
    )

    inst_span = _char_span_to_token_span(
        prompt,
        inst_char_start,
        inst_char_end,
        get_length,
    )

    query_range = [
        query_span
    ]

    # IMPORTANT:
    # two disjoint Passage ranges.
    doc_range = [
        doc1_span,
        doc2_span,
    ]

    inst_range = [
        inst_span
    ]

    return (
        role_range,
        query_range,
        doc_range,
        inst_range,
    )


def normalize_pos(value):
    if isinstance(value, slice):
        return (
            "slice",
            value.start,
            value.stop,
            value.step,
        )

    if isinstance(value, list):
        return tuple(
            int(x)
            for x in value
        )

    if isinstance(value, tuple):
        return tuple(
            int(x)
            for x in value
        )

    if hasattr(value, "item"):
        try:
            return int(
                value.item()
            )
        except Exception:
            pass

    if isinstance(value, int):
        return int(value)

    return repr(value)


def target_position(
    seq_len,
    target,
    ranges,
):
    (
        role_range,
        query_range,
        doc_range,
        inst_range,
    ) = ranges

    value = get_pos(
        seq_len,
        target,
        role_range=role_range,
        query_range=query_range,
        doc_range=doc_range,
        inst_range=inst_range,
        model_name=MODEL_NAME,
    )

    return normalize_pos(value)


def role_span(role_range, key):
    return tuple(
        int(x)
        for x in role_range[key][0]
    )


def document_text(doc):
    if doc is None:
        raise RuntimeError(
            "Document store returned None"
        )

    if hasattr(doc, "text"):
        value = getattr(
            doc,
            "text",
        )

        if value is not None:
            return str(value)

    if hasattr(doc, "body"):
        value = getattr(
            doc,
            "body",
        )

        if value is not None:
            return str(value)

    raise RuntimeError(
        f"Cannot extract text from {type(doc)}"
    )


def query_text(query):
    if hasattr(query, "text"):
        value = getattr(
            query,
            "text",
        )

        if value is not None:
            return str(value)

    if hasattr(query, "query"):
        value = getattr(
            query,
            "query",
        )

        if value is not None:
            return str(value)

    raise RuntimeError(
        f"Cannot extract query text from {type(query)}"
    )


def load_examples(tokenizer):
    """
    Load the exact 54 controlled DL20 pairs.

    Every example always contains BOTH:
      - one judged relevant passage
      - one fixed label-0 irrelevant passage
    """

    import ir_datasets

    require_file(PAIRS_TSV)

    pairs = pd.read_csv(
        PAIRS_TSV,
        sep="\t",
        dtype={
            "qid": str,
            "pos_doc": str,
            "neg_doc": str,
        },
    )

    required_columns = {
        "qid",
        "pos_doc",
        "pos_label",
        "neg_doc",
        "neg_label",
    }

    missing = (
        required_columns
        - set(pairs.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing pairs.tsv columns: "
            f"{sorted(missing)}"
        )

    if len(pairs) != EXPECTED_QIDS:
        raise RuntimeError(
            f"Expected {EXPECTED_QIDS} pairs, "
            f"got {len(pairs)}"
        )

    if (
        pairs["qid"].nunique()
        != EXPECTED_QIDS
    ):
        raise RuntimeError(
            "Controlled DL20 qids are not unique"
        )

    if not (
        pairs["neg_label"]
        .astype(int)
        .eq(0)
        .all()
    ):
        raise RuntimeError(
            "Every negative passage must have label 0"
        )

    dataset = ir_datasets.load(
        IR_DATASET
    )

    query_map = {}

    for q in dataset.queries_iter():
        query_map[
            str(q.query_id)
        ] = query_text(q)

    store = dataset.docs_store()

    examples = []

    for sample_index, row in pairs.iterrows():

        qid = str(row["qid"])

        pos_docid = str(
            row["pos_doc"]
        )

        neg_docid = str(
            row["neg_doc"]
        )

        if qid not in query_map:
            raise RuntimeError(
                f"Missing query qid={qid}"
            )

        pos_obj = store.get(
            pos_docid
        )

        neg_obj = store.get(
            neg_docid
        )

        if pos_obj is None:
            raise RuntimeError(
                f"Missing positive doc={pos_docid}"
            )

        if neg_obj is None:
            raise RuntimeError(
                f"Missing negative doc={neg_docid}"
            )

        query = truncate(
            tokenizer,
            query_map[qid],
            QUERY_LENGTH,
        )

        pos_text = truncate(
            tokenizer,
            document_text(pos_obj),
            PASSAGE_LENGTH,
        )

        neg_text = truncate(
            tokenizer,
            document_text(neg_obj),
            PASSAGE_LENGTH,
        )

        examples.append(
            {
                "sample": int(sample_index),
                "qid": qid,
                "query": query,
                "pos_docid": pos_docid,
                "neg_docid": neg_docid,
                "pos_label": int(
                    row["pos_label"]
                ),
                "neg_label": int(
                    row["neg_label"]
                ),
                "pos_text": pos_text,
                "neg_text": neg_text,
            }
        )

    if len(examples) != EXPECTED_QIDS:
        raise RuntimeError(
            "Unexpected controlled-example count"
        )

    return examples


def condition_view(
    example,
    condition,
):
    """
    Formal Pairwise patching conditions.

    relevance:
        A = relevant
        B = irrelevant
        correct = Yes

    irrelevance:
        A = irrelevant
        B = relevant
        correct = No
    """

    if condition == "relevance":

        return {
            "doc1":
                example["pos_text"],
            "doc2":
                example["neg_text"],

            "doc1_id":
                example["pos_docid"],
            "doc2_id":
                example["neg_docid"],

            "doc1_label":
                example["pos_label"],
            "doc2_label":
                example["neg_label"],

            "correct":
                "Yes",
            "wrong":
                "No",
        }

    if condition == "irrelevance":

        return {
            "doc1":
                example["neg_text"],
            "doc2":
                example["pos_text"],

            "doc1_id":
                example["neg_docid"],
            "doc2_id":
                example["pos_docid"],

            "doc1_label":
                example["neg_label"],
            "doc2_label":
                example["pos_label"],

            "correct":
                "No",
            "wrong":
                "Yes",
        }

    raise ValueError(
        f"Unknown condition={condition}"
    )
