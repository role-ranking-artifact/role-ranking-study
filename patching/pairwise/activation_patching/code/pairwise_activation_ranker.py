from __future__ import annotations

import gc
import json

import pandas as pd
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformer_lens import HookedTransformer

from methods import activation_patch

from pairwise_core import (
    MODEL_NAME,
    MODEL_PATH,
    MAX_LENGTH,
    SOURCE_PAIR_IDS,
    build_prompt_package,
    choose_yesno_ids,
    chat_tensor,
    condition_view,
    get_pairwise_ranges_exact,
)


def _serialize_pos(x):
    if isinstance(x, slice):
        return (
            f"slice({x.start},{x.stop},{x.step})"
        )

    if isinstance(x, list):
        return json.dumps(
            [int(v) for v in x]
        )

    if isinstance(x, tuple):
        return json.dumps(
            [int(v) for v in x]
        )

    return str(x)


class PairwiseActivationPatchingRanker:
    """
    Llama DL20 Pairwise Yes/No activation patching.

    One model load for the whole experiment.
    """

    def __init__(self):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available"
            )

        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Exactly one visible CUDA GPU is required; "
                f"got {torch.cuda.device_count()}"
            )

        print(
            "[GPU]",
            torch.cuda.get_device_name(0),
            flush=True,
        )

        config = AutoConfig.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            trust_remote_code=True,
        )

        if config.model_type != "llama":
            raise RuntimeError(
                f"Expected llama, got {config.model_type}"
            )

        print(
            "[LOAD] tokenizer",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            use_fast=False,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.tokenizer.use_default_system_prompt = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.tokenizer.pad_token_id = (
            self.tokenizer.eos_token_id
        )

        self.tokenizer.padding_side = "left"

        (
            self.yes_id,
            self.no_id,
            self.yes_text,
            self.no_text,
        ) = choose_yesno_ids(
            self.tokenizer
        )

        print(
            "[TOKENS] "
            f"Yes={self.yes_text!r}:{self.yes_id} "
            f"No={self.no_text!r}:{self.no_id}",
            flush=True,
        )

        print(
            "[LOAD] HuggingFace Llama",
            flush=True,
        )

        self.hf_model = (
            AutoModelForCausalLM.from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map={"": 0},
            )
        )

        self.hf_model.eval()

        for p in self.hf_model.parameters():
            if (
                p.device.type != "cuda"
                or p.device.index != 0
            ):
                raise RuntimeError(
                    "HF model is not entirely on CUDA:0"
                )

        print(
            "[LOAD] HookedTransformer",
            flush=True,
        )

        self.model = (
            HookedTransformer
            .from_pretrained_no_processing(
                MODEL_NAME,
                hf_model=self.hf_model,
                tokenizer=self.tokenizer,
                device="cuda",
                dtype="bfloat16",
            )
        )

        self.model.eval()

        if self.model.cfg.n_layers != 32:
            raise RuntimeError(
                "Expected 32 Llama layers, got "
                f"{self.model.cfg.n_layers}"
            )

        if self.model.cfg.n_heads != 32:
            raise RuntimeError(
                "Expected 32 Llama heads, got "
                f"{self.model.cfg.n_heads}"
            )

        print(
            "[LOAD] model ready "
            f"layers={self.model.cfg.n_layers} "
            f"heads={self.model.cfg.n_heads}",
            flush=True,
        )

    def _correct_wrong_ids(
        self,
        condition,
    ):
        if condition == "relevance":
            return (
                self.yes_id,
                self.no_id,
            )

        if condition == "irrelevance":
            return (
                self.no_id,
                self.yes_id,
            )

        raise ValueError(condition)

    def _build_dataloaders(
        self,
        examples,
        role_pair,
        condition,
    ):
        clean_role = (
            f"role_{role_pair}"
        )

        corrupt_role = (
            f"role_{role_pair + 10}"
        )

        clean_pkg = build_prompt_package(
            clean_role,
        )

        corrupt_pkg = build_prompt_package(
            corrupt_role,
        )

        if clean_pkg[0] != "output_2":
            raise RuntimeError(
                "Clean prompt is not output_2"
            )

        if corrupt_pkg[0] != "output_2":
            raise RuntimeError(
                "Corrupt prompt is not output_2"
            )

        clean_template = clean_pkg[1]
        clean_role_text = clean_pkg[2]
        instruction = clean_pkg[3]

        corrupt_template = corrupt_pkg[1]
        corrupt_role_text = corrupt_pkg[2]

        clean_loader = []
        corrupt_loader = []
        sample_meta = []

        for i, example in enumerate(examples):

            view = condition_view(
                example,
                condition,
            )

            clean_prompt = (
                clean_template.format(
                    query=example["query"],
                    doc1=view["doc1"],
                    doc2=view["doc2"],
                )
            )

            corrupt_prompt = (
                corrupt_template.format(
                    query=example["query"],
                    doc1=view["doc1"],
                    doc2=view["doc2"],
                )
            )

            clean_ranges = (
                get_pairwise_ranges_exact(
                    self.tokenizer,
                    clean_prompt,
                    clean_role_text,
                    example["query"],
                    view["doc1"],
                    view["doc2"],
                    instruction,
                    use_chat=True,
                )
            )

            corrupt_ranges = (
                get_pairwise_ranges_exact(
                    self.tokenizer,
                    corrupt_prompt,
                    corrupt_role_text,
                    example["query"],
                    view["doc1"],
                    view["doc2"],
                    instruction,
                    use_chat=True,
                )
            )

            clean_tokens = chat_tensor(
                self.tokenizer,
                clean_prompt,
            )

            corrupt_tokens = chat_tensor(
                self.tokenizer,
                corrupt_prompt,
            )

            if (
                clean_tokens.shape
                != corrupt_tokens.shape
            ):
                raise RuntimeError(
                    "Clean/corrupt tensor-shape "
                    f"mismatch qid={example['qid']} "
                    f"role_pair={role_pair} "
                    f"condition={condition}"
                )

            if (
                clean_tokens.shape[1]
                >= MAX_LENGTH
            ):
                raise RuntimeError(
                    "Unexpected truncation-risk: "
                    f"qid={example['qid']} "
                    f"seq_len={clean_tokens.shape[1]}"
                )

            clean_loader.append(
                (
                    clean_tokens,
                    *clean_ranges,
                )
            )

            corrupt_loader.append(
                (
                    corrupt_tokens,
                    *corrupt_ranges,
                )
            )

            sample_meta.append({
                "sample": i,
                "qid": example["qid"],
                "pos_docid": example["pos_docid"],
                "neg_docid": example["neg_docid"],
                "pos_label": example["pos_label"],
                "neg_label": example["neg_label"],
                "passage_a_docid": view["doc1_id"],
                "passage_b_docid": view["doc2_id"],
                "passage_a_label": view["doc1_label"],
                "passage_b_label": view["doc2_label"],
            })

        return (
            clean_loader,
            corrupt_loader,
            sample_meta,
            clean_role,
            corrupt_role,
        )

    def run_unit(
        self,
        examples,
        role_pair,
        condition,
        activation,
        target,
    ):
        if role_pair not in range(1, 11):
            raise ValueError(
                f"Invalid role_pair={role_pair}"
            )

        if activation not in {
            "resid_pre",
            "z",
        }:
            raise ValueError(
                f"Invalid activation={activation}"
            )

        (
            clean_loader,
            corrupt_loader,
            sample_meta,
            clean_role,
            corrupt_role,
        ) = self._build_dataloaders(
            examples,
            role_pair,
            condition,
        )

        correct_id, wrong_id = (
            self._correct_wrong_ids(
                condition
            )
        )

        if activation == "resid_pre":
            axis_names = (
                "layer",
                "pos",
            )

            axis_values = (
                self.model.cfg.n_layers,
                target,
            )

        else:
            axis_names = (
                "layer",
                "pos",
                "head",
            )

            axis_values = (
                self.model.cfg.n_layers,
                target,
                self.model.cfg.n_heads,
            )

        print(
            "[UNIT] "
            f"condition={condition} "
            f"activation={activation} "
            f"target={target} "
            f"{clean_role}->{corrupt_role} "
            f"correct={correct_id} "
            f"wrong={wrong_id}",
            flush=True,
        )

        patch_df = activation_patch(
            model=self.model,
            clean_dataloader=clean_loader,
            corrupted_dataloader=corrupt_loader,
            correct_token_id=correct_id,
            wrong_token_id=wrong_id,
            activation_name=activation,
            index_axis_names=axis_names,
            index_axis_values=axis_values,
            use_pos=True,
            n_samples=len(examples),
            model_name=MODEL_NAME,
            patch_direction="clean_to_corrupt",
        )

        meta = {
            int(x["sample"]): x
            for x in sample_meta
        }

        patch_df.insert(
            0,
            "qid",
            patch_df["sample"].map(
                lambda x: meta[int(x)]["qid"]
            ),
        )

        patch_df.insert(
            1,
            "condition",
            condition,
        )

        patch_df.insert(
            2,
            "source_pair_id",
            SOURCE_PAIR_IDS[
                role_pair - 1
            ],
        )

        patch_df.insert(
            3,
            "clean_role",
            clean_role,
        )

        patch_df.insert(
            4,
            "corrupt_role",
            corrupt_role,
        )

        patch_df.insert(
            5,
            "patch_target",
            target,
        )

        patch_df.insert(
            6,
            "patch_activation",
            activation,
        )

        patch_df.insert(
            7,
            "correct_token_id",
            correct_id,
        )

        patch_df.insert(
            8,
            "wrong_token_id",
            wrong_id,
        )

        for col in [
            "pos_docid",
            "neg_docid",
            "pos_label",
            "neg_label",
            "passage_a_docid",
            "passage_b_docid",
            "passage_a_label",
            "passage_b_label",
        ]:
            patch_df[col] = (
                patch_df["sample"].map(
                    lambda x, c=col:
                        meta[int(x)][c]
                )
            )

        for col in [
            "pos",
            "src_pos",
            "dest_pos",
            "receiver_pos",
        ]:
            if col in patch_df.columns:
                patch_df[col] = (
                    patch_df[col].apply(
                        _serialize_pos
                    )
                )

        expected = (
            len(examples)
            * self.model.cfg.n_layers
        )

        if activation == "z":
            expected *= (
                self.model.cfg.n_heads
            )

        if len(patch_df) != expected:
            raise RuntimeError(
                f"Unit row-count mismatch: "
                f"expected={expected}, "
                f"got={len(patch_df)}"
            )

        if patch_df["qid"].nunique() != len(examples):
            raise RuntimeError(
                "Unit lost one or more qids"
            )

        self.model.reset_hooks()

        del clean_loader
        del corrupt_loader

        gc.collect()

        return patch_df
