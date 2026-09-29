import json
import random
import numpy as np
import torch
import transformers
from typing import Optional


def get_pointwise_ranges(tokenizer, prompt, role, query, document, inst, use_chat=False):
    # Preserve newlines; remove surrounding spaces only.
    if use_chat:
        _get_length = lambda x: len(tokenizer.apply_chat_template([{"role": "user", "content": x.strip(' ')}], continue_final_message=True))
    else:
        _get_length = lambda x: len(tokenizer(x.strip(' '))[0])
    # role
    if role == "":
        role_range = [(0, 0)]
    else:
        prefix, *_ = prompt.split(role)
        role_start = _get_length(prefix)
        if len(prefix) == 0:
            role_start -= 1
        role_end = _get_length(prefix + role)
        role_range = [(role_start, role_end)]
    # query
    prefix, *_ = prompt.split(query)
    query_start = _get_length(prefix)
    query_end = _get_length(prefix + query)
    query_range = [(query_start, query_end)]
    # document
    prefix, *_ = prompt.split(document)
    doc_start = _get_length(prefix)
    doc_end = _get_length(prefix + document)
    doc_range = [(doc_start, doc_end)]
    # inst
    prefix, *_ = prompt.split(inst)
    inst_start = _get_length(prefix)
    if len(prefix) == 0:
        inst_start -= 1
    inst_end = _get_length(prefix + inst)
    inst_range = [(inst_start, inst_end)]

    return role_range, query_range, doc_range, inst_range

def get_pairwise_ranges(tokenizer, prompt, role, query, document1, document2, inst, use_chat=False):
    if use_chat:
        _get_length = lambda x: len(tokenizer.apply_chat_template([{"role": "user", "content": x.strip(' ')}], continue_final_message=True))
    else:
        _get_length = lambda x: len(tokenizer(x.strip(' '))[0])

    # role
    if role == "":
        role_range = [(0, 0)]
    else:
        prefix, *_ = prompt.split(role)
        role_start = _get_length(prefix)
        if len(prefix) == 0:
            role_start -= 1
        role_end = _get_length(prefix + role)
        role_range = [(role_start, role_end)]
    # query
    prefix, *_ = prompt.split(query)
    query_start = _get_length(prefix)
    query_end = _get_length(prefix + query)
    query_range = [(query_start, query_end)]
    # document1
    prefix, *_ = prompt.split(document1)
    doc_start = _get_length(prefix)
    doc_end = _get_length(prefix + document1)
    doc_range_1 = (doc_start, doc_end)
    # document2
    prefix, *_ = prompt.split(document2)
    doc_start = _get_length(prefix.strip())
    doc_end = _get_length(prefix + document2)
    doc_range_2 = (doc_start, doc_end)
    # Keep document 1 and document 2 lengths separate.
    # doc_range = [(doc_range_1[0], doc_range_1[1]), (doc_range_2[0], doc_range_2[1])]
    # Merge document 1 and document 2 ranges.
    doc_range = [(doc_range_1[0], doc_range_2[1])]


    # inst
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
    role_position: str, # front, back
    use_chat: bool = True,
    use_pos: bool = None, 
    use_hard_neg: bool = False, 
    nsamples: int = 100,
    seqlen: int = 2048
):
    random.seed(42)  # for reproducibility

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f] #[:nsamples]
        # nsamples is used in methods.py to limit the sample count.

    # role 
    if role_play == "positive":
        role = pos_role
    elif role_play == "negative":
        role = neg_role
    elif role_play == "none":
        role = ""

    # instruction
    if format == "pointwise":
        inst = "Does the document answer the query?\nAnswer 'Yes' or 'No'.\nAnswer: "
    elif format == "pairwise":
        inst = "Is document A more relevant to the query?\nAnswer 'Yes' or 'No'.\nAnswer: "

    # prompt_template and role_position
    if format == "pointwise":
        if role_position == "front":
            if prompt_template == "doc_first": # table 1, 4
                prompt_template = role + "Document: {document}\nQuery: {query}\n" + inst
            elif prompt_template == "query_first": # table 2
                prompt_template = role + "Query: {query}\nDocument: {document}\n" + inst
        elif role_position == "back":
            if prompt_template == "doc_first": # table 3
                prompt_template = "Document: {document}\nQuery: {query}\n" + role + inst
            elif prompt_template == "query_first":
                prompt_template = "Query: {query}\nDocument: {document}\n" + role + inst
    elif format == "pairwise":
        if role_position == "front":
            if prompt_template == "doc_first": # table 1, 4
                prompt_template = role + "Document A: {document1}\nDocument B: {document2}\nQuery: {query}\n" + inst
            elif prompt_template == "query_first": # table 2
                prompt_template = role + "Query: {query}\nDocument A: {document1}\nDocument B: {document2}\n" + inst
        elif role_position == "back":
            if prompt_template == "doc_first": # table 3
                prompt_template = "Document A: {document1}\nDocument B: {document2}\nQuery: {query}\n" + role + inst
            elif prompt_template == "query_first":
                prompt_template = "Query: {query}\nDocument A: {document1}\nDocument B: {document2}\n" + role + inst

    # get query, doc and label to construct prompt and ranges
    dataloader = []
    for i, item in enumerate(data):
        if format == "pointwise":
            query = item["query"]
            if use_pos is True:            
                document = item["positive_document"]
                label = 1
            elif use_pos is False:
                if use_hard_neg:
                    document = random.choice(item["hard_negative_document"])
                else:
                    document = random.choice(item["random_negative_document"])
                label = 0
            else:
                raise ValueError("use_pos is None is not supported")
            prompt = prompt_template.format(query=query, document=document)
            role_range, query_range, doc_range, inst_range = get_pointwise_ranges(tokenizer, prompt, role, query, document, inst, use_chat)

        elif format == "pairwise":
            query = item["query"]
            document1 = item["positive_document"]
            if use_hard_neg:
                document2 = random.choice(item["hard_negative_document"])
            else:
                document2 = random.choice(item["random_negative_document"])
            # decide document 1 and document 2
            if use_pos is True:
                pass
            elif use_pos is False:
                document1, document2 = document2, document1
            else:
                raise ValueError("use_pos is None is not supported")
            document1, document2 = document1.strip(), document2.strip()

            label = None
            prompt = prompt_template.format(
                query=query,
                document1=document1,
                document2=document2,
            )
            role_range, query_range, doc_range, inst_range = get_pairwise_ranges(tokenizer, prompt, role, query, document1, document2, inst, use_chat)

        if use_chat:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding="longest",
                max_length=seqlen,
                truncation=True,
                add_generation_prompt=True
            )
        else:
            input_ids = tokenizer(prompt, return_tensors="pt", padding="longest", max_length=seqlen, truncation=True).input_ids

        dataloader.append((input_ids, role_range, query_range, doc_range, inst_range, label))

    return dataloader

def truncate_to_equal_length(clean_dataloader, corrupted_dataloader):
    new_clean_dataloader = []
    new_corrupted_dataloader = []
    for (clean_tokens, clean_query_range, clean_doc_range, clean_inst_range, _), \
        (corrupted_tokens, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, _) in zip(clean_dataloader, corrupted_dataloader):

        clean_query_start, clean_query_end = clean_query_range[0]
        corrupted_query_start, corrupted_query_end = corrupted_query_range[0]
        clean_doc_start, clean_doc_end = clean_doc_range[0]
        corrupted_doc_start, corrupted_doc_end = corrupted_doc_range[0]

        assert torch.equal(clean_tokens[0, clean_query_start:clean_query_end], corrupted_tokens[0, corrupted_query_start:corrupted_query_end])

        if clean_tokens.shape[1] > corrupted_tokens.shape[1]:
            new_clean_tokens = corrupted_tokens.clone()
            new_clean_tokens[:, corrupted_doc_start:corrupted_doc_end - 1] = clean_tokens[:, clean_doc_start:clean_doc_start + (corrupted_doc_end - corrupted_doc_start) - 1]
            new_clean_dataloader.append((new_clean_tokens, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, None))
            new_corrupted_dataloader.append((corrupted_tokens, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, None))
        else:
            new_corrupted_tokens = clean_tokens.clone()
            new_corrupted_tokens[:, clean_doc_start:clean_doc_end - 1] = corrupted_tokens[:, corrupted_doc_start:corrupted_doc_start + (clean_doc_end - clean_doc_start) - 1]
            new_clean_dataloader.append((clean_tokens, clean_query_range, clean_doc_range, clean_inst_range, None))
            new_corrupted_dataloader.append((new_corrupted_tokens, clean_query_range, clean_doc_range, clean_inst_range, None))
