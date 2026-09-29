import json
import random
import torch
import transformers
from typing import Dict, List, Tuple

SpanList = List[Tuple[int, int]]
RoleRangeDict = Dict[str, SpanList]


def _make_len_getter(tokenizer, use_chat=False):
    if use_chat:
        return lambda x: len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": x.strip(" ")}],
                continue_final_message=True,
            )
        )
    else:
        return lambda x: len(tokenizer(x.strip(" "))[0])


def _char_span_to_token_span(prompt: str, start_char: int, end_char: int, _get_length):
    prefix = prompt[:start_char]
    span_text = prompt[:end_char]
    start_tok = _get_length(prefix)
    if start_char == 0:
        start_tok -= 1
    end_tok = _get_length(span_text)
    return (start_tok, end_tok)


def _extract_role_component_char_spans(role_text: str):
    role_text = role_text.rstrip("\n")
    all_span = (0, len(role_text))

    adj_phrase_start = role_text.index("You are ") + len("You are ")
    adj_phrase_end = role_text.index(" search assistant")
    adj_phrase = role_text[adj_phrase_start:adj_phrase_end]

    adj_word = adj_phrase.split()[-1]
    adj_start = adj_phrase_start + adj_phrase.rfind(adj_word)
    adj_end = adj_start + len(adj_word)

    modal_start = role_text.index(" search assistant that ") + len(" search assistant that ")
    modal_end = role_text.index(" rank passages")

    adv_start = role_text.index(" rank passages ") + len(" rank passages ")
    adv_end = role_text.index(", based")

    stripped = role_text.rstrip(".")
    last_word = stripped.split()[-1]
    last_start = stripped.rfind(last_word)
    last_end = last_start + len(last_word)

    return {
        "all": all_span,
        "adj_phrase": (adj_phrase_start, adj_phrase_end),
        "adj": (adj_start, adj_end),
        "modal": (modal_start, modal_end),
        "adv": (adv_start, adv_end),
        "last": (last_start, last_end),
    }


def _build_role_range_dict(tokenizer, prompt: str, role: str, use_chat=False) -> RoleRangeDict:
    if role == "":
        zero = [(0, 0)]
        return {"all": zero, "adj_phrase": zero, "adj": zero, "modal": zero, "adv": zero, "last": zero}

    _get_length = _make_len_getter(tokenizer, use_chat)
    role_global_start = prompt.index(role)
    role_global_end = role_global_start + len(role)
    comp_char_spans = _extract_role_component_char_spans(role)

    out: RoleRangeDict = {}
    out["all"] = [_char_span_to_token_span(prompt, role_global_start, role_global_end, _get_length)]
    for key in ["adj_phrase", "adj", "modal", "adv", "last"]:
        local_start, local_end = comp_char_spans[key]
        global_start = role_global_start + local_start
        global_end = role_global_start + local_end
        out[key] = [_char_span_to_token_span(prompt, global_start, global_end, _get_length)]
    return out


def get_pointwise_ranges(tokenizer, prompt, role, query, document, inst, use_chat=False):
    _get_length = _make_len_getter(tokenizer, use_chat)
    role_range = _build_role_range_dict(tokenizer, prompt, role, use_chat)

    prefix, *_ = prompt.split(query)
    query_start = _get_length(prefix)
    query_end = _get_length(prefix + query)
    query_range = [(query_start, query_end)]

    prefix, *_ = prompt.split(document)
    doc_start = _get_length(prefix)
    doc_end = _get_length(prefix + document)
    doc_range = [(doc_start, doc_end)]

    prefix, *_ = prompt.split(inst)
    inst_start = _get_length(prefix)
    if len(prefix) == 0:
        inst_start -= 1
    inst_end = _get_length(prefix + inst)
    inst_range = [(inst_start, inst_end)]

    return role_range, query_range, doc_range, inst_range


def get_pairwise_ranges(tokenizer, prompt, role, query, document1, document2, inst, use_chat=False):
    _get_length = _make_len_getter(tokenizer, use_chat)
    role_range = _build_role_range_dict(tokenizer, prompt, role, use_chat)

    prefix, *_ = prompt.split(query)
    query_start = _get_length(prefix)
    query_end = _get_length(prefix + query)
    query_range = [(query_start, query_end)]

    prefix, *_ = prompt.split(document1)
    doc_start = _get_length(prefix)
    doc_end = _get_length(prefix + document1)
    doc_range_1 = (doc_start, doc_end)

    prefix, *_ = prompt.split(document2)
    doc_start = _get_length(prefix.strip())
    doc_end = _get_length(prefix + document2)
    doc_range_2 = (doc_start, doc_end)
    doc_range = [(doc_range_1[0], doc_range_2[1])]

    prefix, *_ = prompt.split(inst)
    inst_start = _get_length(prefix)
    if len(prefix) == 0:
        inst_start -= 1
    inst_end = _get_length(prefix + inst)
    inst_range = [(inst_start, inst_end)]

    return role_range, query_range, doc_range, inst_range


def get_loaders(
    data_path: str,
    pos_role: str,
    neg_role: str,
    tokenizer: transformers.PreTrainedTokenizer,
    format: str,
    role_play: str,
    prompt_template: str,
    role_position: str,
    use_chat: bool = True,
    use_pos: bool = None,
    use_hard_neg: bool = False,
    nsamples: int = 100,
    seqlen: int = 2048,
):
    random.seed(42)
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    if role_play == "positive":
        role = pos_role
    elif role_play == "negative":
        role = neg_role
    elif role_play == "none":
        role = ""
    else:
        raise ValueError(f"Unsupported role_play={role_play}")

    if format == "pointwise":
        inst = "Does the document answer the query?\nAnswer 'Yes' or 'No'.\nAnswer: "
    elif format == "pairwise":
        inst = "Is document A more relevant to the query?\nAnswer 'Yes' or 'No'.\nAnswer: "
    else:
        raise ValueError(f"Unsupported format={format}")

    if format == "pointwise":
        if role_position == "front":
            if prompt_template == "doc_first":
                prompt_template = role + "Document: {document}\nQuery: {query}\n" + inst
            elif prompt_template == "query_first":
                prompt_template = role + "Query: {query}\nDocument: {document}\n" + inst
            else:
                raise ValueError(f"Unsupported prompt_template={prompt_template}")
        elif role_position == "back":
            if prompt_template == "doc_first":
                prompt_template = "Document: {document}\nQuery: {query}\n" + role + inst
            elif prompt_template == "query_first":
                prompt_template = "Query: {query}\nDocument: {document}\n" + role + inst
            else:
                raise ValueError(f"Unsupported prompt_template={prompt_template}")
        else:
            raise ValueError(f"Unsupported role_position={role_position}")
    elif format == "pairwise":
        if role_position == "front":
            if prompt_template == "doc_first":
                prompt_template = role + "Document A: {document1}\nDocument B: {document2}\nQuery: {query}\n" + inst
            elif prompt_template == "query_first":
                prompt_template = role + "Query: {query}\nDocument A: {document1}\nDocument B: {document2}\n" + inst
            else:
                raise ValueError(f"Unsupported prompt_template={prompt_template}")
        elif role_position == "back":
            if prompt_template == "doc_first":
                prompt_template = "Document A: {document1}\nDocument B: {document2}\nQuery: {query}\n" + role + inst
            elif prompt_template == "query_first":
                prompt_template = "Query: {query}\nDocument A: {document1}\nDocument B: {document2}\n" + role + inst
            else:
                raise ValueError(f"Unsupported prompt_template={prompt_template}")
        else:
            raise ValueError(f"Unsupported role_position={role_position}")

    dataloader = []
    for item in data:
        if format == "pointwise":
            query = item["query"]
            if use_pos is True:
                document = item["positive_document"]
                label = 1
            elif use_pos is False:
                document = random.choice(item["hard_negative_document"] if use_hard_neg else item["random_negative_document"])
                label = 0
            else:
                raise ValueError("use_pos is None is not supported")
            prompt = prompt_template.format(query=query, document=document)
            role_range, query_range, doc_range, inst_range = get_pointwise_ranges(tokenizer, prompt, role, query, document, inst, use_chat)
        else:
            query = item["query"]
            document1 = item["positive_document"]
            document2 = random.choice(item["hard_negative_document"] if use_hard_neg else item["random_negative_document"])
            if use_pos is False:
                document1, document2 = document2, document1
            elif use_pos is not True:
                raise ValueError("use_pos is None is not supported")
            document1, document2 = document1.strip(), document2.strip()
            label = None
            prompt = prompt_template.format(query=query, document1=document1, document2=document2)
            role_range, query_range, doc_range, inst_range = get_pairwise_ranges(tokenizer, prompt, role, query, document1, document2, inst, use_chat)

        if use_chat:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", padding="longest", max_length=seqlen, truncation=True, add_generation_prompt=True)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt", padding="longest", max_length=seqlen, truncation=True).input_ids

        dataloader.append((input_ids, role_range, query_range, doc_range, inst_range, label))
    return dataloader


def truncate_to_equal_length(clean_dataloader, corrupted_dataloader):
    new_clean_dataloader = []
    new_corrupted_dataloader = []
    for (clean_tokens, clean_role_range, clean_query_range, clean_doc_range, clean_inst_range, _), \
        (corrupted_tokens, corrupted_role_range, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, _) in zip(clean_dataloader, corrupted_dataloader):

        clean_query_start, clean_query_end = clean_query_range[0]
        corrupted_query_start, corrupted_query_end = corrupted_query_range[0]
        clean_doc_start, clean_doc_end = clean_doc_range[0]
        corrupted_doc_start, corrupted_doc_end = corrupted_doc_range[0]

        assert torch.equal(clean_tokens[0, clean_query_start:clean_query_end], corrupted_tokens[0, corrupted_query_start:corrupted_query_end])

        if clean_tokens.shape[1] > corrupted_tokens.shape[1]:
            new_clean_tokens = corrupted_tokens.clone()
            new_clean_tokens[:, corrupted_doc_start:corrupted_doc_end - 1] = clean_tokens[:, clean_doc_start:clean_doc_start + (corrupted_doc_end - corrupted_doc_start) - 1]
            new_clean_dataloader.append((new_clean_tokens, clean_role_range, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, None))
            new_corrupted_dataloader.append((corrupted_tokens, corrupted_role_range, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, None))
        else:
            new_corrupted_tokens = clean_tokens.clone()
            new_corrupted_tokens[:, clean_doc_start:clean_doc_end - 1] = corrupted_tokens[:, corrupted_doc_start:corrupted_doc_start + (clean_doc_end - clean_doc_start) - 1]
            new_clean_dataloader.append((clean_tokens, clean_role_range, clean_query_range, clean_doc_range, clean_inst_range, None))
            new_corrupted_dataloader.append((new_corrupted_tokens, corrupted_role_range, clean_query_range, clean_doc_range, clean_inst_range, None))

    return new_clean_dataloader, new_corrupted_dataloader
