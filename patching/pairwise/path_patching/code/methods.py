import gc
import sys
import os
import random
import time
from pathlib import Path
import torch

def _to_model_device(obj, model):
    """
    Recursively move tensors inside BatchEncoding, dict, list, or tuple objects
    to the model device.
    """
    # Support both model.device and parameter-based device lookup.
    try:
        dev = getattr(model, "device", next(model.parameters()).device)
    except StopIteration:
        dev = getattr(model, "device", None)

    if dev is None:
        return obj

    if torch.is_tensor(obj):
        return obj.to(dev)

    if isinstance(obj, dict):  # Covers HuggingFace BatchEncoding.
        return {k: _to_model_device(v, model) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        moved = [_to_model_device(x, model) for x in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved

    return obj
from torch import Tensor
import numpy as np
import pandas as pd
import einops
from tqdm.auto import tqdm
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import Literal, Callable, Optional, Tuple, Union, List, Dict, Any, Sequence
from functools import partial
from rich.table import Table, Column
from rich import print as rprint
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache, patching
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP


random.seed(42)



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



def get_logit(
    logits: Float[Tensor, "batch pos vocab"],
    token_idx: int,
):
    return logits[:, -1, token_idx].mean().item()


def get_prob(
    logits: Float[Tensor, "batch pos vocab"],
    token_idx: int,
):
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    return probs[:, -1, token_idx].mean().item()


def logit_diff(
    logits: Float[Tensor, "batch pos vocab"],
    correct: int,
    wrong: int,
) -> float:
    logitdiff = get_logit(logits, correct) - get_logit(logits, wrong)
    return logitdiff


def prob_diff(
    logits: Float[Tensor, "batch pos vocab"],
    correct: int,
    wrong: int,
) -> float:
    probdiff = get_prob(logits, correct) - get_prob(logits, wrong)
    return probdiff

AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]
PosNames = Literal["all", "last",
                    "role_first", "role_last", "role_all",
                    "role_adj_phrase", "role_adj", "role_modal", "role_adv", "role_adj_adv",
                    "query_first", "query_last", "query_all",
                    "doc_first", "doc_last", "doc_all",
                    "inst_first", "inst_last", "inst_all",
                    "qd_all", "other"]


def _role_span(role_range, key="all"):
    if isinstance(role_range, dict):
        return role_range[key][0]
    return role_range[0]


def get_pos(
    seqlen: int,
    pos_type: PosNames,
    role_range: Optional[Any] = None,
    query_range: Optional[Tuple[int, int]] = None,
    doc_range: Optional[Tuple[int, int]] = None,
    inst_range: Optional[Tuple[int, int]] = None,
    model_name: Optional[str] = None,
) -> Union[int, slice, list[int]]:
    if pos_type == "all":
        return slice(0, seqlen)
    elif pos_type == "last":
        return seqlen - 1
    elif pos_type == "role_first":
        return _role_span(role_range, "all")[0]
    elif pos_type == "role_last":
        return _role_span(role_range, "last")[1] - 1
    elif pos_type == "role_all":
        return slice(*_role_span(role_range, "all"))
    elif pos_type == "role_adj_phrase":
        return slice(*_role_span(role_range, "adj_phrase"))
    elif pos_type == "role_adj":
        span = _role_span(role_range, "adj")
        return span[1] - 1 if span[1] > span[0] else span[0]
    elif pos_type == "role_modal":
        span = _role_span(role_range, "modal")
        return span[1] - 1 if span[1] > span[0] else span[0]
    elif pos_type == "role_adv":
        span = _role_span(role_range, "adv")
        return span[1] - 1 if span[1] > span[0] else span[0]
    elif pos_type == "role_adj_adv":
        adj_span = _role_span(role_range, "adj")
        adv_span = _role_span(role_range, "adv")

        adj_pos = adj_span[1] - 1 if adj_span[1] > adj_span[0] else adj_span[0]
        adv_pos = adv_span[1] - 1 if adv_span[1] > adv_span[0] else adv_span[0]

        if adj_pos == adv_pos:
            return [adj_pos]
        return [adj_pos, adv_pos]
    elif pos_type == "query_first":
        return query_range[0][0]
    elif pos_type == "query_last":
        return query_range[0][-1] - 1
    elif pos_type == "query_all":
        return slice(*query_range[0])
    elif pos_type == "doc_first":
        if len(doc_range) == 1:
            return doc_range[0][0]
        else:
            return [doc_range[0][0], doc_range[1][0]]
    elif pos_type == "doc_last":
        if len(doc_range) == 1:
            return doc_range[0][-1] - 1
        else:
            return [doc_range[0][-1] - 1, doc_range[1][-1] - 1]
    elif pos_type == "doc_all":
        if len(doc_range) == 1:
            return slice(*doc_range[0])
        else:
            return list(range(*doc_range[0])) + list(range(*doc_range[1]))

    elif pos_type == "qd_all":
        if len(doc_range) == 1:
            return list(range(*query_range[0])) + list(range(*doc_range[0]))
        else:
            return list(range(*query_range[0])) + list(range(*doc_range[0])) + list(range(*doc_range[1]))
    elif pos_type == "inst_first":
        return inst_range[0][0]
    elif pos_type == "inst_last":
        return inst_range[0][-1] - 1
    elif pos_type == "inst_all":
        return slice(*inst_range[0])
    elif pos_type == "other":
        # positions except the above all
        if len(doc_range) == 1:
            return list(set(range(seqlen - 1)) - set(range(*_role_span(role_range, "all"))) - set(range(*query_range[0])) - set(
                range(*doc_range[0])) - set(range(*inst_range[0])))
        else:
            return list(set(range(seqlen - 1)) - set(range(*_role_span(role_range, "all"))) - set(range(*query_range[0])) - set(
                range(*doc_range[0])) - set(range(*doc_range[1])) - set(range(*inst_range[0])))
    else:
        raise ValueError(f"Invalid pos_type: {pos_type}")

PATCH_SETTER = {
    ("layer", "pos"): patching.layer_pos_patch_setter, # which is true of everything that is not an attention pattern shaped tensor.
    ("layer", "pos", "head"): patching.layer_pos_head_vector_patch_setter, #which is true of all attention head vector activations (q, k, v, z, result) but not of attention patterns.
    ("layer", "head"): patching.layer_head_vector_patch_setter, # which is true of all attention head vector activations (q, k, v, z, result) but not of attention patterns.
    ("layer", "head", "dest_pos"): patching.layer_head_pos_pattern_patch_setter,
    ("layer", "head", "dest_pos", "src_pos"): patching.layer_head_dest_src_pos_pattern_patch_setter,
}


def create_index_df(
    dataloader: List[Tuple[Tensor, ...]],
    sample_slice: slice,
    model_name: Optional[str] = None,
    **kwargs,
    ) -> pd.DataFrame:
    """
    Create the parameter grid for activation patching experiments.
    Each row is one sample/layer/head/position condition.
    """
    rows = []

    if "sender_layer_or_heads" in kwargs:
        sender_layer_or_heads = kwargs.pop("sender_layer_or_heads")
        if sender_layer_or_heads is None:
            sender_layers = None
            sender_heads = None
        elif isinstance(sender_layer_or_heads, int):
            sender_layers = {sender_layer_or_heads}
            sender_heads = None
        else:
            sender_layers = set(next(zip(*sender_layer_or_heads)))
            sender_heads = sender_layer_or_heads
    else:
        sender_layers = None
        sender_heads = None

    for i in range(sample_slice.start, sample_slice.stop):
        clean_tokens, role_range, query_range, doc_range, inst_range, *_ = dataloader[i]

        for layer in range(kwargs["layer"]):
            if sender_layers is not None and layer not in sender_layers:
                continue

            if "head" in kwargs:
                for head in range(kwargs["head"]):
                    if sender_heads is not None and (layer, head) not in sender_heads:
                        continue

                    row = {"sample": i, "layer": layer}

                    for pos in ["pos", "src_pos", "dest_pos"]:
                        if pos in kwargs:
                            row[pos] = get_pos(
                                clean_tokens.shape[1],
                                kwargs[pos],
                                role_range=role_range,
                                query_range=query_range,
                                doc_range=doc_range,
                                inst_range=inst_range,
                                model_name=model_name,
                            )

                    row["head"] = head

                    if "receiver_pos" in kwargs:
                        row["receiver_pos"] = get_pos(
                            clean_tokens.shape[1],
                            kwargs["receiver_pos"],
                            role_range=role_range,
                            query_range=query_range,
                            doc_range=doc_range,
                            inst_range=inst_range,
                            model_name=model_name,
                        )
                        row["receiver_layer_or_heads"] = kwargs["receiver_layer_or_heads"]
                        row["receiver_activation_name"] = kwargs["receiver_activation_name"]

                    rows.append(row)

            else:
                row = {"sample": i, "layer": layer}

                for pos in ["pos", "src_pos", "dest_pos"]:
                    if pos in kwargs:
                        row[pos] = get_pos(
                            clean_tokens.shape[1],
                            kwargs[pos],
                            role_range=role_range,
                            query_range=query_range,
                            doc_range=doc_range,
                            inst_range=inst_range,
                            model_name=model_name,
                        )

                if "receiver_pos" in kwargs:
                    row["receiver_pos"] = get_pos(
                        clean_tokens.shape[1],
                        kwargs["receiver_pos"],
                        role_range=role_range,
                        query_range=query_range,
                        doc_range=doc_range,
                        inst_range=inst_range,
                        model_name=model_name,
                    )
                    row["receiver_layer_or_heads"] = kwargs["receiver_layer_or_heads"]
                    row["receiver_activation_name"] = kwargs["receiver_activation_name"]

                rows.append(row)

    df = pd.DataFrame(rows)

    preferred = [
        "sample", "layer", "pos", "src_pos", "dest_pos", "head",
        "receiver_pos", "receiver_layer_or_heads", "receiver_activation_name"
    ]
    ordered = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    return df


@torch.no_grad()
def activation_patch(
    model: HookedTransformer,
    clean_dataloader: List[Tuple[Tensor, ...]],
    corrupted_dataloader: List[Tuple[Tensor, ...]],
    correct_token_id: int,
    wrong_token_id: int,
    activation_name: str,
    index_axis_names: Sequence[AxisNames],
    index_axis_values: Sequence[Union[int, str]],
    use_pos: bool,
    n_samples: int = 10,
    model_name: Optional[str] = None,
    patch_direction: str = "clean_to_corrupt",
) -> pd.DataFrame:
    # Clear hooks from previous runs.
    model.reset_hooks()

    n_samples = min(n_samples, len(clean_dataloader))

    if use_pos:
        sample_slice = slice(0, n_samples)
    else:
        sample_slice = slice(n_samples, 2 * n_samples)

    # Normalize wildcard placeholders to model dimensions.
    axis_kwargs = dict(zip(index_axis_names, index_axis_values))

    if "layer" in axis_kwargs and not isinstance(axis_kwargs["layer"], int):
        axis_kwargs["layer"] = model.cfg.n_layers

    if "head" in axis_kwargs and not isinstance(axis_kwargs["head"], int):
        axis_kwargs["head"] = model.cfg.n_heads

    if "head_index" in axis_kwargs and not isinstance(axis_kwargs["head_index"], int):
        axis_kwargs["head"] = model.cfg.n_heads
        axis_kwargs.pop("head_index", None)

    index_df = create_index_df(
        clean_dataloader,
        sample_slice,
        model_name=model_name,
        **axis_kwargs
    )

    patch_setter = PATCH_SETTER[tuple(index_axis_names)]

    def patching_hook(base_activation, hook, index, source_activation):
        return patch_setter(base_activation, index, source_activation)

    correct_logit = partial(get_logit, token_idx=correct_token_id)
    wrong_logit = partial(get_logit, token_idx=wrong_token_id)
    correct_prob = partial(get_prob, token_idx=correct_token_id)
    wrong_prob = partial(get_prob, token_idx=wrong_token_id)

    metric_dict = {
        f"{input_type}_{metric}": np.zeros(len(index_df))
        for input_type in ["clean", "corrupted", "patched"]
        for metric in ["correct_logit", "wrong_logit", "correct_prob", "wrong_prob", "ld", "prob_diff"]
    }

    for i in tqdm(range(sample_slice.start, sample_slice.stop)):
        clean_tokens = clean_dataloader[i][0]
        corrupted_tokens = corrupted_dataloader[i][0]

        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=[utils.get_act_name(activation_name, j) for j in range(model.cfg.n_layers)],
        )
        corrupted_logits, corrupted_cache = model.run_with_cache(
            corrupted_tokens,
            names_filter=[utils.get_act_name(activation_name, j) for j in range(model.cfg.n_layers)],
        )

        sample_index_df = index_df[index_df["sample"] == i]

        for index_row in list(sample_index_df.iterrows()):
            gi = index_row[0]
            _, *index = index_row[1].to_list()

            current_activation_name = utils.get_act_name(activation_name, layer=index[0])

            if patch_direction == "clean_to_corrupt":
                base_tokens = corrupted_tokens
                source_activation = clean_cache[current_activation_name]
            elif patch_direction == "corrupt_to_clean":
                base_tokens = clean_tokens
                source_activation = corrupted_cache[current_activation_name]
            else:
                raise ValueError(f"Unsupported patch_direction={patch_direction}")

            current_hook = partial(
                patching_hook,
                index=index,
                source_activation=source_activation,
            )

            patched_logits = model.run_with_hooks(
                base_tokens,
                fwd_hooks=[(current_activation_name, current_hook)]
            )

            for prefix, logits in zip(
                ["clean", "corrupted", "patched"],
                [clean_logits, corrupted_logits, patched_logits]
            ):
                for metric, func in zip(
                    ["correct_logit", "wrong_logit", "correct_prob", "wrong_prob"],
                    [correct_logit, wrong_logit, correct_prob, wrong_prob],
                ):
                    metric_dict[f"{prefix}_{metric}"][gi] = func(logits)

                metric_dict[f"{prefix}_ld"][gi] = logit_diff(logits, correct_token_id, wrong_token_id)
                metric_dict[f"{prefix}_prob_diff"][gi] = prob_diff(logits, correct_token_id, wrong_token_id)

    for key, value in metric_dict.items():
        index_df[key] = value

    denom = index_df["clean_ld"] - index_df["corrupted_ld"]
    index_df["normalized_ld"] = np.where(
        np.abs(denom) < 1e-12,
        np.nan,
        (index_df["patched_ld"] - index_df["corrupted_ld"]) / denom
    )
    index_df["ld_recovery"] = index_df["patched_ld"] - index_df["corrupted_ld"]
    index_df["prob_recovery"] = index_df["patched_correct_prob"] - index_df["corrupted_correct_prob"]
    index_df["patch_direction"] = patch_direction

    return index_df
@torch.no_grad()
def get_mean_activations(
    model: HookedTransformer,
    mean_dataloader: List[Tuple[Tensor, ...]],
    n_samples: int = 10,
    use_pos: bool = True,
):
    """
    for each layer:
    note: attn_hook_z: [batch, seq_len, n_heads, d_head]
    note: resid_pre_hook: [batch, seq_len, d_model]
    note: d_head = d_model // n_heads
    note: sum over batch, then mean over seq_len
    mean_activations: {layer: {"z": [n_heads, d_head],
                                "resid_pre": [d_model]}}
    """
    model.reset_hooks()
    # set sample slice
    n_samples = min(n_samples, len(mean_dataloader))
    if use_pos:
        sample_slice = slice(0, n_samples)
    elif not use_pos:
        sample_slice = slice(n_samples, 2 * n_samples)

    # Get sum of activations for each layer over all samples
    mean_activations = {layer: {} for layer in range(model.cfg.n_layers)}
    def get_activation_hook(activation, hook):
        if "z" in hook.name:
            # activation: [batch, seq_len, n_heads, d_head]
            activation_mean = activation.sum(0).mean(0)  # [n_heads, d_head]
            mean_activations[hook.layer()]["z"] = mean_activations[hook.layer()].get("z", 0) + activation_mean
        elif "resid_pre" in hook.name:
            # activation: [batch, seq_len, d_model]
            activation_mean = activation.sum(0).mean(0)  # [d_model]
            mean_activations[hook.layer()]["resid_pre"] = mean_activations[hook.layer()].get("resid_pre", 0) + activation_mean
        return
        # loop over samples
    for i in tqdm(range(sample_slice.start, sample_slice.stop), desc="Computing mean activations"):
        tokens = mean_dataloader[i][0]
        _ = model.run_with_hooks(
            tokens,
            fwd_hooks=
            [(utils.get_act_name("z", layer), get_activation_hook) for layer in range(model.cfg.n_layers)] + \
            [(utils.get_act_name("resid_pre", layer), get_activation_hook) for layer in range(model.cfg.n_layers)],
        )
    # divide by number of samples
    for layer in mean_activations:
        mean_activations[layer]["z"] /= n_samples
        mean_activations[layer]["resid_pre"] /= n_samples

    return mean_activations

@torch.no_grad()
def zero_mean_ablation(
    model: HookedTransformer,
    evaluation_dataloader: List[Tuple[Tensor, ...]],
    ablation_style: Literal["mean", "zero"],
    ablate_on: Literal["heads", "layers"],
    ablate_heads: dict[tuple[int, int], str], #{(layer, head): "pos_name"}
    correct_token_id: int,
    wrong_token_id: int,
    mean_activations: Optional[dict[int, dict[str, Tensor]]] = None,
    mask_ablate_modules: bool = True,
    n_samples: int = 10,
    use_pos: bool = True,
):
    model.reset_hooks()
    # set sample slice
    n_samples = min(n_samples, len(evaluation_dataloader))
    if use_pos:
        sample_slice = slice(0, n_samples)
    elif not use_pos:
        sample_slice = slice(n_samples, 2 * n_samples)

    # get mean activations
    if ablation_style == "mean" and mean_activations is None:
        mean_activations = get_mean_activations(model, evaluation_dataloader, n_samples, use_pos)
    elif ablation_style == "mean" and mean_activations is not None:
        pass # for pairwise ranking, mean_activations is already computed
    elif ablation_style == "zero":
        mean_activations = None

     # loop over samples
    ablated_outputs = []
    for i in tqdm(range(sample_slice.start, sample_slice.stop), desc="Computing ablated outputs"):
        # get lenth of tokens
        tokens, role_range, query_range, doc_range, inst_range, *_ = evaluation_dataloader[i]
        seqlen = tokens.shape[1]
        role_range = list(range(*_role_span(role_range, "all")))
        query_range = list(range(*query_range[0]))
        doc_range = list(range(*doc_range[0])) if len(doc_range) == 1 else list(range(*doc_range[0])) + list(range(*doc_range[1]))
        inst_range = list(range(*inst_range[0]))

        pos_dict = {
            "role_all": role_range,
            "query_all": query_range,
            "doc_all": doc_range,
            "inst_all": inst_range,
            "last": seqlen - 1,
        }

        ablate_pos_dict = {
            "role_all": list(set(range(seqlen)) - set(role_range)),
            "query_all": list(set(range(seqlen)) - set(query_range)),
            "doc_all": list(set(range(seqlen)) - set(doc_range)),
            "inst_all": list(set(range(seqlen)) - set(inst_range)),
            "last": slice(0, seqlen - 1),
        }

        # Compute ablated outputs
        def ablate_head_hook(activation, hook):
            # note: attn_hook_z: [batch, seq_len, n_heads, d_head]
            # note: mean_activations[hook.layer()]["z"]: [n_heads, d_head]
            for head_idx in range(model.cfg.n_heads):
                head = (hook.layer(), head_idx)  # Current head identifier.

                if not mask_ablate_modules:
                    if head not in ablate_heads:
                        if ablation_style == "mean":
                            activation[:, :, head_idx] = mean_activations[hook.layer()]["z"][head_idx].unsqueeze(
                                0).unsqueeze(0)
                        else:
                            activation[:, :, head_idx] = 0
                    else:
                        pos = ablate_pos_dict[ablate_heads[head]]  # Complement positions for this head.
                        if ablation_style == "mean":
                            activation[:, pos, head_idx] = mean_activations[hook.layer()]["z"][head_idx].unsqueeze(
                                0).unsqueeze(0)
                        else:
                            activation[:, pos, head_idx] = 0

                elif mask_ablate_modules and head in ablate_heads:
                    pos = pos_dict[ablate_heads[head]]  # Target positions for this head.
                    if ablation_style == "mean":
                        activation[:, pos, head_idx] = mean_activations[hook.layer()]["z"][head_idx].unsqueeze(
                            0).unsqueeze(0)
                    else:
                        activation[:, pos, head_idx] = 0
            return activation

        # ablate layer
        def ablate_layer_hook(activation, hook):
            # note: activation: [batch, seq_len, d_model]
            # note: mean_activations[hook.layer()]["resid_pre"]: [d_model]
            if ablation_style == "mean":
                activation[...] = mean_activations[hook.layer()]["resid_pre"].unsqueeze(0).unsqueeze(0)
            elif ablation_style == "zero":
                activation[...] = 0
            return activation

        # run model
        if ablate_on == "heads":
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(utils.get_act_name("z", layer), ablate_head_hook) for layer in
                            range(model.cfg.n_layers)],
                return_type="logits"
            )
        elif ablate_on == "layers":
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(utils.get_act_name("resid_pre", layer), ablate_layer_hook) for layer in
                            range(model.cfg.n_layers)],
                return_type="logits"
            )  # [0, -1, :] # [batch, seq_len, vocab]

        # turn logits to scores
        correct_logit = logits[0, -1, correct_token_id]
        wrong_logit = logits[0, -1, wrong_token_id]

        batch_scores = torch.stack([correct_logit, wrong_logit])  # [2]
        batch_scores = torch.nn.functional.softmax(batch_scores, dim=0)  # [2] # prob_score

        #correct_score = batch_scores[0].item()  # [batch]
        #ablated_outputs.append((i, correct_score))
        # ablated_outputs.append(logits.cpu())
        correct_score = batch_scores[0].item()

        pred_label = int(correct_logit > wrong_logit)

        ablated_outputs.append({
            "sample": i,
            "correct_score": correct_score,
            "correct_logit": correct_logit.item(),
            "wrong_logit": wrong_logit.item(),
            "pred_label": pred_label,
        })
    return ablated_outputs


@torch.no_grad()
def zero_shot_ranking(
    model: HookedTransformer,
    evaluation_dataloader: List[Tuple[Tensor, ...]],
    correct_token_id: int,
    wrong_token_id: int,
    n_samples: int = 10,
    use_pos: bool = True,
):
    n_samples = min(n_samples, len(evaluation_dataloader))
    if use_pos:
        sample_slice = slice(0, n_samples)
    elif not use_pos:
        sample_slice = slice(n_samples, 2 * n_samples)
    # Run the model without patching and collect baseline logits.
    outputs = []
    for i in tqdm(range(sample_slice.start, sample_slice.stop), desc="Computing zero-shot ranking outputs"):
        tokens = evaluation_dataloader[i][0]
        # Move tokens to the model device for both HookedTransformer and HF models.
        tokens = _to_model_device(tokens, model)

        if hasattr(model, "run_with_cache"):
            # HookedTransformer path.
            logits, _ = model.run_with_cache(tokens, return_type="logits")
        else:
            # HuggingFace model path.
            with torch.inference_mode():
                # Pass all fields together when tokens are BatchEncoding/dict.
                batch = {"input_ids": tokens} if torch.is_tensor(tokens) else tokens
                batch = _to_model_device(batch, model)
                out = model(**batch)
                logits = out.logits
        correct_logit = logits[0, -1, correct_token_id]
        wrong_logit = logits[0, -1, wrong_token_id]

        batch_scores = torch.stack([correct_logit, wrong_logit]) # [2]
        batch_scores = torch.nn.functional.softmax(batch_scores, dim=0) # [2] # prob_score

        correct_score = batch_scores[0].item()

        pred_label = int(correct_logit > wrong_logit)

        outputs.append({
            "sample": i,
            "correct_score": correct_score,
            "correct_logit": correct_logit.item(),
            "wrong_logit": wrong_logit.item(),
            "pred_label": pred_label,
        })
    return outputs
