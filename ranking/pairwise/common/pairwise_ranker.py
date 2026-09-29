from typing import List
import copy
import json
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from ranker import LlmRanker, SearchResult


class Text2TextGenerationDataset:
    """
    Compatibility class required by pointwise_ranker import.
    Pairwise smoke tests do not directly use it, but main.py imports pointwise_ranker.
    """
    def __init__(self, data, tokenizer):
        self.data = tokenizer(data)

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, item):
        return {
            "input_ids": self.data["input_ids"][item],
            "attention_mask": self.data["attention_mask"][item],
        }



def _load_role_dict(experiment_type: str):
    if experiment_type == "baseline_without_role":
        return {"role_0": ""}

    role_json_path = os.environ.get("ROLE_JSON_PATH", "").strip()
    if not role_json_path:
        raise ValueError(
            "ROLE_JSON_PATH is required for role-based pairwise runs. "
            "For baseline use --experiment_type baseline_without_role."
        )

    with open(role_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    role_dict = {}

    if "pos_roles" in obj and "neg_roles" in obj:
        pos = obj["pos_roles"]
        neg = obj["neg_roles"]

        # Runtime convention:
        # pos role_1..role_10, neg role_11..role_20
        for i, key in enumerate(sorted(pos.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else x), 1):
            role_dict[f"role_{i}"] = pos[key]

        # Keep negative roles at 11..20 even in the five-pair last chunk.
        offset = 10
        for i, key in enumerate(sorted(neg.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else x), 1):
            role_dict[f"role_{offset + i}"] = neg[key]

    else:
        # Already flat dict
        role_dict = obj

    return role_dict


def prompt_generator(
    prompt_type,
    original_prompt_number,
    instruction,
    output,
    tone,
    order,
    position,
    role,
    query_before_instruction,
    role_at_beginning,
    experiment_type,
):
    if prompt_type == "original":
        return (
            "Given a query: {query}, which of the following two passages is more relevant to the query?\n"
            "Passage A: {doc1}\n"
            "Passage B: {doc2}\n"
            "Answer with A or B.\n"
            "Answer:"
        )

    if prompt_type != "adjusted":
        raise NotImplementedError(f"prompt_type={prompt_type}")

    instruction_dict = {
        "instruction_1": 'Given a query "{query}", which of the following two passages is more relevant to the query?\n',
        "instruction_2": "Is document A more relevant to the query?\n",
    }

    output_dict = {
        # Important: use A/B as the immediate next token, not "Passage A/B",
        # because "Passage A" first token is usually shared.
        "output_1": "Answer with A or B.\nAnswer:",
        "output_2": "Answer \'Yes\' or \'No\'.\nAnswer: ",
    }

    tone_dict = {
        "tone_0": "",
        "tone_1": "Please ",
        "tone_2": "Only ",
        "tone_3": "Must ",
        "tone_4": "You better get this right or you will be punished. ",
        "tone_5": "Only respond with the ranking result, do not explain. ",
    }

    role_dict = _load_role_dict(experiment_type)
    if role not in role_dict:
        raise KeyError(f"{role} not found in role_dict. Available examples: {list(role_dict.keys())[:10]}")

    role_text = role_dict[role]
    passages = "Passage A: {doc1}\nPassage B: {doc2}\n"

    if query_before_instruction == "True":
        prompt_instruction = "Query: {query}\n" + instruction_dict[instruction]
    else:
        prompt_instruction = instruction_dict[instruction] + "Query: {query}\n"

    if order == "query_first":
        core = prompt_instruction + passages
    elif order == "passage_first":
        core = passages + prompt_instruction
    else:
        raise NotImplementedError(f"order={order}")

    if role_at_beginning == "True":
        if position == "beginning":
            prompt = role_text + core + tone_dict[tone] + output_dict[output]
        elif position == "ending":
            prompt = role_text + tone_dict[tone] + output_dict[output] + core
        else:
            raise NotImplementedError(f"position={position}")
    else:
        if position == "beginning":
            prompt = role_text + core + tone_dict[tone] + output_dict[output]
        elif position == "ending":
            prompt = tone_dict[tone] + output_dict[output] + role_text + core
        else:
            raise NotImplementedError(f"position={position}")

    return [output, prompt, role_text, instruction_dict[instruction]]


class PairwiseLlmRanker(LlmRanker):
    def __init__(
        self,
        model_name,
        model_path,
        tokenizer_name_or_path,
        device,
        method="heapsort",
        batch_size=1,
        k=10,
        hf_token=None,
        prompt_type="adjusted",
        original_prompt_number=1,
        instruction="instruction_2",
        output="output_2",
        tone="tone_0",
        order="query_first",
        position="beginning",
        role="role_0",
        query_before_instruction="True",
        role_at_beginning="True",
        experiment_type="baseline_without_role",
        ablation_style="zero",
        ablate_on="heads",
        patching_pos="last",
        top_k=10,
        data_format="pairwise",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.device = device
        self.method = method
        self.batch_size = batch_size
        self.k = k

        self.prompt_type = prompt_type
        self.original_prompt_number = original_prompt_number
        self.instruction = instruction
        self.output = output
        self.tone = tone
        self.order = order
        self.position = position
        self.role = role
        self.query_before_instruction = query_before_instruction
        self.role_at_beginning = role_at_beginning
        self.experiment_type = experiment_type
        self.data_format = data_format
        self.HF_TOKEN = hf_token

        if self.experiment_type == "zero_mean_ablation":
            raise NotImplementedError("This clean pairwise_ranker.py is for ranking probe only, not ablation.")

        self.config = AutoConfig.from_pretrained(model_path, token=self.HF_TOKEN, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            token=self.HF_TOKEN,
            use_fast=False,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=self.HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

        self.ab_ids = self._choose_pair_token_ids([["A", " A"], ["B", " B"]], "A/B")
        self.yesno_ids = self._choose_pair_token_ids([["Yes", " Yes", "yes", " yes"], ["No", " No", "no", " no"]], "Yes/No")

        print(f"[PAIRWISE INIT] model_type={self.config.model_type}")
        print(f"[PAIRWISE INIT] A/B token ids={self.ab_ids}")
        print(f"[PAIRWISE INIT] Yes/No token ids={self.yesno_ids}")
        print(f"[PAIRWISE INIT] instruction={self.instruction} output={self.output} role={self.role}")

    def _single_token_id(self, candidates):
        for s in candidates:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0], s
        raise ValueError(f"No single-token candidate found among {candidates}")

    def _choose_pair_token_ids(self, pair_candidates, label):
        left_id, left_s = self._single_token_id(pair_candidates[0])
        right_id, right_s = self._single_token_id(pair_candidates[1])
        print(f"[TOKEN CHECK] {label}: {left_s!r}->{left_id}, {right_s!r}->{right_id}")
        return (left_id, right_id)

    def _format_chat(self, prompt):
        conversation = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _next_token_probs(self, prompts, token_pair):
        texts = [self._format_chat(p) for p in prompts]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :]
            pair_logits = logits[:, list(token_pair)]
            probs = torch.softmax(pair_logits, dim=1)

        return probs.detach().float().cpu().tolist()

    def compare(self, query: str, docs: List[str]):
        doc1, doc2 = docs[0], docs[1]

        prompt_package = prompt_generator(
            self.prompt_type,
            self.original_prompt_number,
            self.instruction,
            self.output,
            self.tone,
            self.order,
            self.position,
            self.role,
            self.query_before_instruction,
            self.role_at_beginning,
            self.experiment_type,
        )
        prompt_tem = prompt_package[1]

        prompt0 = prompt_tem.format(query=query, doc1=doc1, doc2=doc2)
        prompt1 = prompt_tem.format(query=query, doc1=doc2, doc2=doc1)

        # Yes/No mode:
        # prompt0 asks whether doc1 as Passage A is better than doc2.
        # prompt1 asks whether doc2 as Passage A is better than doc1.
        if self.output == "output_2" or self.instruction == "instruction_2":
            probs = self._next_token_probs([prompt0, prompt1], self.yesno_ids)
            yes0 = probs[0][0]
            yes1 = probs[1][0]
            return {"mode": "yesno", "yes0": yes0, "yes1": yes1, "margin": yes0 - yes1}

        # A/B mode:
        # prompt0: doc1=A, doc2=B. doc1 wins if A>B.
        # prompt1: doc2=A, doc1=B. doc1 wins if B>A.
        probs = self._next_token_probs([prompt0, prompt1], self.ab_ids)
        a0, b0 = probs[0]
        a1, b1 = probs[1]
        margin = (a0 - b0) + (b1 - a1)
        return {"mode": "ab", "a0": a0, "b0": b0, "a1": a1, "b1": b1, "margin": margin}

    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and arr[l] > arr[i]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr, k):
        n = len(arr)
        ranked = 0
        for i in range(n // 2, -1, -1):
            self.heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self.heapify(arr, i, 0)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)

        if self.method != "heapsort":
            raise NotImplementedError(f"Only heapsort is implemented in clean pairwise ranker, got {self.method}")

        class ComparableDoc:
            def __init__(self, docid, text, ranker):
                self.docid = docid
                self.text = text
                self.ranker = ranker

            def __gt__(self, other):
                out = self.ranker.compare(query, [self.text, other.text])
                return out["margin"] > 0

        arr = [ComparableDoc(docid=doc.docid, text=doc.text, ranker=self) for doc in ranking]
        self.heapSort(arr, self.k)

        ranking = [SearchResult(docid=doc.docid, score=-i, text=None) for i, doc in enumerate(reversed(arr))]

        results = []
        top_doc_ids = set()
        rank = 1

        for doc in ranking[: self.k]:
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1

        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
