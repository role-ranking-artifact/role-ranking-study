from typing import List, Tuple, Optional
from ranker import LlmRanker, SearchResult
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from pairwise_ranker import Text2TextGenerationDataset
import torch
import sys
import os
import json
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_structure import get_pointwise_ranges
from methods import zero_mean_ablation, zero_shot_ranking, activation_patch


def prompt_generator(prompt_type, original_prompt_number, instruction,
                     output, tone, order, position, role,
                     query_before_instruction, role_at_beginning, experiment_type):
    if prompt_type == "original":
        original_prompt_dict = {
            1: ["output_1", "Query: {query}\n Passage: {text}\n Does the passage answer the query?  Answer 'Yes' or 'No'"],
            2: ["output_2", "Passage: {text}\n Query: {query}\n Is this passage relevant to the query? Please answer True/False. Answer:"],
            3: ["output_3", "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.\nQuery: {query} Document:{text} Output:"],
            4: ["output_4", "From a scale of 0 to 4, judge the relevance between the query and the document.'.\n Query: {query} Document:{text} Output:"],
        }
        return original_prompt_dict[original_prompt_number]

    instruction_dict = {
        "instruction_1": "Does the passage answer the query?\n",
        "instruction_2": "Is this passage relevant to the query?\n",
        "instruction_3": "For the following query and document, judge whether they are relevant.\n",
        "instruction_4": "Judge the relevance between the query and the document.\n"
    }
    output_dict = {
        "output_1": "Answer 'Yes' or 'No'. ",
        "output_2": "Answer True/False. ",
        "output_3": "Judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'. ",
        "output_4": "From a scale of 0 to 4, judge the relevance between the query and the document.",
        "output_5": "Answer 'Yes' or 'No'.\nAnswer: ",
    }
    tone_dict = {
        "tone_0": "",
        "tone_1": "Please ",
        "tone_2": "Only ",
        "tone_3": "Must ",
        "tone_4": "You better get this right or you will be punished. ",
        "tone_5": "Only response the ranking results, do not say any word or explain. "
    }

    role_dict_path = os.environ.get("ROLE_JSON_PATH")

    role_dict = {"role_0": ""}
    if experiment_type in ["normal_run", "zero_mean_ablation", "zero_shot_ranking", "activation_patching"]:
        if not role_dict_path:
            raise RuntimeError(
                "ROLE_JSON_PATH must be set for role-conditioned experiments."
            )
        with open(role_dict_path, "r") as f:
            dict_loaded = json.load(f)
        pos = dict_loaded["pos_roles"]
        neg = dict_loaded["neg_roles"]
        for i in range(1, 11):
            role_dict[f"role_{i}"] = pos[f"role_{i}"]
            role_dict[f"role_{10+i}"] = neg[f"role_{i}"]
    elif experiment_type == "baseline_without_role":
        role_dict = {"role_0": ""}

    if query_before_instruction == "True":
        prompt_instruction = "Query: {query}\n" + instruction_dict[instruction]
    else:
        prompt_instruction = instruction_dict[instruction] + "Query: {query}\n"
    passages = "\nPassage: {text}\n"

    if role_at_beginning == "True":
        if order == "query_first":
            if position == "beginning":
                prompt = role_dict[role] + prompt_instruction + passages + tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + prompt_instruction + passages
            else:
                raise NotImplementedError
        elif order == "passage_first":
            if position == "beginning":
                prompt = role_dict[role] + passages + "Query: {query}\n" + instruction_dict[instruction] + tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + passages + prompt_instruction
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError("Aligned patching code assumes role_at_beginning=True")

    return [output, prompt, role_dict[role], instruction_dict[instruction]]


def _hashable_pos(x):
    if isinstance(x, slice):
        return f"slice({x.start},{x.stop},{x.step})"
    if isinstance(x, list):
        return "list(" + ",".join(map(str, x)) + ")"
    if isinstance(x, tuple):
        return "tuple(" + ",".join(map(str, x)) + ")"
    return x


def _single_token_id(tokenizer, candidates, label):
    for text in candidates:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        if len(ids) == 1:
            return int(ids[0])

    raise ValueError(
        f"Could not find a single-token representation "
        f"for {label}: {candidates}"
    )


def _yes_no_token_ids(tokenizer, model_name):
    name = str(model_name).lower()

    if "qwen" in name:
        yes_candidates = ["Yes"]
        no_candidates = ["No"]

    elif "mistral" in name or "mixtral" in name:
        yes_candidates = ["Yes", " Yes"]
        no_candidates = ["No", " No"]

    elif "llama" in name:
        yes_candidates = [" Yes"]
        no_candidates = [" No"]

    else:
        raise ValueError(
            "Unsupported model for Yes/No token selection: "
            f"{model_name}"
        )

    yes_id = _single_token_id(
        tokenizer,
        yes_candidates,
        "Yes",
    )
    no_id = _single_token_id(
        tokenizer,
        no_candidates,
        "No",
    )

    return yes_id, no_id


class PointwiseLlmRanker(LlmRanker):
    def __init__(self,
                 model_name,
                 model_path,
                 tokenizer_name_or_path,
                 device,
                 method="yes_no",
                 batch_size=1,
                 hf_token=None,
                 prompt_type="adjusted",
                 original_prompt_number=1,
                 instruction="instruction_1",
                 output="output_5",
                 tone="tone_1",
                 order="query_first",
                 position="beginning",
                 role="role_0",
                 query_before_instruction="False",
                 role_at_beginning="True",
                 experiment_type="normal_run",
                 ablation_style="zero",
                 ablate_on="heads",
                 patching_pos="last",
                 top_k=10,
                 data_format="pointwise",
                 doc_source="relevance",
                 patch_activation="resid_pre",
                 patch_target=None,
                 n_samples=100,
                 clean_role=None,
                 corrupt_role=None,
                 patch_index_axis_names=("layer", "pos"),
                 patch_direction="clean_to_corrupt",
                 query_doc_limit=1):
        self.HF_TOKEN = hf_token
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.device = device
        self.method = method
        self.batch_size = batch_size
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
        self.ablation_style = ablation_style
        self.ablate_on = ablate_on
        self.patching_pos = patching_pos
        self.top_k = top_k
        self.data_format = data_format
        self.patch_activation = patch_activation
        self.patch_target = patch_target
        self.n_samples = n_samples
        self.clean_role = clean_role
        self.corrupt_role = corrupt_role
        self.patch_index_axis_names = patch_index_axis_names
        self.patch_direction = patch_direction
        self.query_doc_limit = query_doc_limit
        self.doc_source = doc_source

        self.config = AutoConfig.from_pretrained(model_path, token=self.HF_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=self.HF_TOKEN,
            use_fast=False,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=self.HF_TOKEN,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.tokenizer.use_default_system_prompt = False
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize HookedTransformer once for the full run.
        print("[INFO] Loading HookedTransformer once...")
        self.hooked_model = HookedTransformer.from_pretrained_no_processing(
            self.model_name,
            hf_model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            dtype="bfloat16",
        )

    def _build_hooked_model(self):
        return self.hooked_model
    def _prompt_pkg(self, role_name: str):
        return prompt_generator(
            self.prompt_type,
            self.original_prompt_number,
            self.instruction,
            self.output,
            self.tone,
            self.order,
            self.position,
            role_name,
            self.query_before_instruction,
            self.role_at_beginning,
            self.experiment_type,
        )

    def patch_query(self, qid: str, query: str, ranking: List[SearchResult]) -> pd.DataFrame:
        assert self.clean_role is not None and self.corrupt_role is not None, \
            "activation_patching requires clean_role and corrupt_role"

        clean_pkg = self._prompt_pkg(self.clean_role)
        corrupt_pkg = self._prompt_pkg(self.corrupt_role)

        clean_prompt_tem, clean_role_text, instruction = clean_pkg[1], clean_pkg[2], clean_pkg[3]
        corrupt_prompt_tem, corrupt_role_text = corrupt_pkg[1], corrupt_pkg[2]

        if clean_pkg[0] not in ["output_1", "output_5"]:
            raise ValueError("Aligned patching currently expects Yes/No outputs")

        yes_id, no_id = _yes_no_token_ids(
            self.tokenizer,
            self.model_name,
        )

        # Default direction: Yes - No.
        correct_token_id = yes_id
        wrong_token_id = no_id
        # Use No - Yes for irrelevance head patching so helpful head effects are positive.
        if getattr(self, "doc_source", None) == "irrelevance" and self.patch_activation == "z":
            correct_token_id = no_id
            wrong_token_id = yes_id

        docs = ranking[:self.query_doc_limit]
        clean_dataloader, corrupt_dataloader = [], []

        for doc in docs:
            clean_prompt = clean_prompt_tem.format(text=doc.text, query=query)
            corrupt_prompt = corrupt_prompt_tem.format(text=doc.text, query=query)

            clean_ranges = get_pointwise_ranges(
                self.tokenizer, clean_prompt, clean_role_text, query, doc.text, instruction, use_chat=True
            )
            corrupt_ranges = get_pointwise_ranges(
                self.tokenizer, corrupt_prompt, corrupt_role_text, query, doc.text, instruction, use_chat=True
            )

            clean_tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": clean_prompt}],
                return_tensors="pt",
                padding="longest",
                max_length=2048,
                truncation=True,
                add_generation_prompt=True,
            )
            corrupt_tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": corrupt_prompt}],
                return_tensors="pt",
                padding="longest",
                max_length=2048,
                truncation=True,
                add_generation_prompt=True,
            )

            clean_dataloader.append((clean_tokens, *clean_ranges))
            corrupt_dataloader.append((corrupt_tokens, *corrupt_ranges))

        hooked_model = self._build_hooked_model()

        if self.patch_activation == "z":
            # Head patching needs an explicit position axis to distinguish targets.
            if tuple(self.patch_index_axis_names) == ("layer", "pos", "head"):
                index_axis_values = (
                    hooked_model.cfg.n_layers,
                    self.patch_target,
                    hooked_model.cfg.n_heads,
                )
            elif tuple(self.patch_index_axis_names) == ("layer", "head"):
                # This mode patches heads across all positions.
                index_axis_values = (
                    hooked_model.cfg.n_layers,
                    hooked_model.cfg.n_heads,
                )
            else:
                raise ValueError(
                    f"Unsupported patch_index_axis_names for z patching: {self.patch_index_axis_names}"
                )
        else:
            index_axis_values = (hooked_model.cfg.n_layers, self.patch_target)

        patch_df = activation_patch(
            model=hooked_model,
            clean_dataloader=clean_dataloader,
            corrupted_dataloader=corrupt_dataloader,
            correct_token_id=correct_token_id,
            wrong_token_id=wrong_token_id,
            activation_name=self.patch_activation,
            index_axis_names=self.patch_index_axis_names,
            index_axis_values=index_axis_values,
            use_pos=True,
            n_samples=min(self.n_samples, len(clean_dataloader)),
            model_name=self.model_name,
            patch_direction=self.patch_direction,
        )

        meta_cols = [
            ("clean_role", self.clean_role),
            ("corrupt_role", self.corrupt_role),
            ("patch_target", self.patch_target),
            ("patch_activation", self.patch_activation),
            ("patch_direction", self.patch_direction),
        ]

        for col, val in meta_cols:
            patch_df[col] = val

        # Convert non-hashable position values before summary aggregation.
        for col in ["pos", "src_pos", "dest_pos", "receiver_pos"]:
            if col in patch_df.columns:
                patch_df[col] = patch_df[col].apply(_hashable_pos)

        # Keep metadata columns first.
        ordered_cols = [c for c, _ in meta_cols]
        other_cols = [c for c in patch_df.columns if c not in ordered_cols]
        patch_df = patch_df[ordered_cols + other_cols]

        return patch_df

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        if self.method != "yes_no":
            raise ValueError("Only yes_no is supported")

        prompt_package = self._prompt_pkg(self.role)
        prompt_tem = prompt_package[1]

        yes_id, no_id = _yes_no_token_ids(
            self.tokenizer,
            self.model_name,
        )

        if self.experiment_type == "normal_run":
            conversation = [{"role": "user", "content": prompt_tem}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )[17:]
            data = [prompt.format(text=doc.text, query=query) for doc in ranking]
            dataset = Text2TextGenerationDataset(data, self.tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=DataCollatorWithPadding(self.tokenizer, max_length=2048, padding='longest'),
                shuffle=False,
                drop_last=False,
                num_workers=0
            )
            current_id = 0
            with torch.no_grad():
                for batch_inputs in tqdm(dataloader):
                    batch_inputs = batch_inputs.to(self.model.device)
                    logits = self.model(
                        input_ids=batch_inputs['input_ids'],
                        attention_mask=batch_inputs['attention_mask']
                    ).logits[:, -1, :]
                    yes_scores = logits[:, yes_id]
                    no_scores = logits[:, no_id]
                    batch_scores = torch.stack((yes_scores, no_scores), dim=1)
                    batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                    scores = batch_scores[:, 0]
                    for score in scores:
                        ranking[current_id].score = score.item()
                        current_id += 1
            ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
            return ranking

        raise NotImplementedError("Use patch_query() for activation_patching in the aligned pipeline")

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
