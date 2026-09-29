from typing import List, Tuple
import transformers
from ranker import LlmRanker, SearchResult

# T5 dependencies are imported only when a T5 model is requested.
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def load_ranker(model_name_or_path, method=None, **kwargs):
    model_name_lc = model_name_or_path.lower()

    if model_name_lc.startswith("t5") or "t5" in model_name_lc:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tok = T5Tokenizer.from_pretrained(model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        model.eval()
        return tok, model, "t5"

    # Default causal language model path.
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()
    return tok, model, "causal"

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from pairwise_ranker import Text2TextGenerationDataset
import torch
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_structure import get_pointwise_ranges
from tqdm import tqdm
import json
# ---- OPTIONAL: transformer_lens (only needed for ablation/patching) ----
TL_AVAILABLE = False
_TL_IMPORT_ERR = None
try:
    from transformer_lens import HookedTransformer  # type: ignore
    TL_AVAILABLE = True
except Exception as e:
    HookedTransformer = None  # type: ignore
    _TL_IMPORT_ERR = repr(e)
# ----------------------------------------------------------------------
from methods import zero_mean_ablation, zero_shot_ranking


def role_dict_generator():
    # all combinations of 10 adjectives, 3 modals, 10 adverbs for positive and negative
    # pos words
    pos_adjective = ["an able", "an expert", "a superb", "a capable", "a reliable",
                     "a strong", "a brilliant", "a logical", "a focused", "knowledgeable"]
    pos_modal = ["can", "will", "shall"]
    pos_adverb = ["brightly", "wisely", "swiftly", "nicely", "accurately",
                  "firmly", "clearly", "carefully", "sharply", "perfectly"]
    # neg words
    neg_adjective = ["a faulty", "a confused", "a clumsy", "a messy", "an incorrect",
                     "an awful", "an unreliable", "a problematic", "a hopeless", "a flawed"]
    neg_modal = ["might", "will", "could"]
    neg_adverb = ["poorly", "wrongly", "mistakenly", "improperly", "falsely",
                  "terribly", "badly", "incorrectly", "sadly", "horribly"]
    # role template
    role_template = "You are {adjective} search assistant that {modal} rank passages {adverb}, basedon their relevance to a query. \n"

    role_dict = {}
    i = 0
    for adjective in pos_adjective:
        for modal in pos_modal:
            for adverb in pos_adverb:
                role_dict[f"role_{i}"] = role_template.format(adjective=adjective, modal=modal, adverb=adverb)
                i += 1
    for adjective in neg_adjective:
        for modal in neg_modal:
            for adverb in neg_adverb:
                role_dict[f"role_{i}"] = role_template.format(adjective=adjective, modal=modal, adverb=adverb)
                i += 1
    return role_dict


def prompt_generator(prompt_type, original_prompt_number, instruction,
                     output, tone, order, position, role,
                     query_before_instruction, role_at_beginning, experiment_type):
    # gives a list of information [0] is flag, [1] is content, which is prompt_package
    if prompt_type == "original":
        original_prompt_dict = {
            1: ["output_1",
                "Query: {query}\n Passage: {text}\n Does the passage answer the query?  Answer 'Yes' or 'No'"],

            2: ["output_2", "Passage: {text}\n Query: {query}\n "
                            "Is this passage relevant to the query? Please answer True/False. Answer:"],

            3: ["output_3",
                "For the following query and document, judge whether they are 'Highly Relevant', 'Somewhat "
                "Relevant', or 'Not Relevant'.\n"
                "Query: {query} Document:{text} Output:"],

            4: ["output_4", "From a scale of 0 to 4, judge the relevance between the query and the document.'.\n "
                            "Query: {query} Document:{text} Output:"],
        }
        return original_prompt_dict[original_prompt_number]

    elif prompt_type == "adjusted":
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
        # Define role_dict based on experiment type.
        # Allow launchers to override the role file without editing code.
        role_dict_path = os.environ.get(
            "ROLE_JSON_PATH",
            os.path.join(parent_dir, "role_dict", "meta-llama-20-random.json")
        )

        # baseline
        role_dict = {"role_0": ""}

        if experiment_type in ["normal_run", "zero_mean_ablation", "zero_shot_ranking"]:
            with open(role_dict_path, "r") as f:
                dict_loaded = json.load(f)

            assert "pos_roles" in dict_loaded and "neg_roles" in dict_loaded, \
                f"Bad role json format: must have pos_roles/neg_roles. Got keys={list(dict_loaded.keys())}"

            pos = dict_loaded["pos_roles"]
            neg = dict_loaded["neg_roles"]
            n_roles = len(pos)
            if n_roles not in (5, 10) or len(neg) != n_roles:
                raise ValueError(f"Expected 5 or 10 matched role pairs in {role_dict_path}")

            # Read by key order to avoid nondeterministic .values() ordering.
            # Load 10 roles per polarity, or 5 in the final chunk.
            for i in range(1, n_roles + 1):
                k = f"role_{i}"
                if k not in pos:
                    raise KeyError(f"Missing {k} in pos_roles of {role_dict_path}")
                if not isinstance(pos[k], str) or len(pos[k].strip()) == 0:
                    raise ValueError(f"Empty text for pos_roles[{k}] in {role_dict_path}")
                role_dict[k] = pos[k]

            # Keep negative roles at role_11..role_20 (or role_11..role_15).
            for j in range(1, n_roles + 1):
                k = f"role_{j}"
                if k not in neg:
                    raise KeyError(f"Missing {k} in neg_roles of {role_dict_path}")
                if not isinstance(neg[k], str) or len(neg[k].strip()) == 0:
                    raise ValueError(f"Empty text for neg_roles[{k}] in {role_dict_path}")
                role_dict[f"role_{10 + j}"] = neg[k]

        elif experiment_type == "baseline_without_role":
            role_dict = {"role_0": ""}

        else:
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
                    raise NotImplementedError(f"Position {position} is not implemented.")
            elif order == "passage_first":
                if position == "beginning":
                    prompt = role_dict[role] + passages + "Query: {query}\n" + instruction_dict[instruction] + \
                             tone_dict[tone] + output_dict[output]
                elif position == "ending":
                    prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + passages + prompt_instruction
                else:
                    raise NotImplementedError(f"Position {position} is not implemented.")
            else:
                raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")
    else:  # role directly before instruction
        if order == "query_first":
            if position == "beginning":
                prompt = role_dict[role] + prompt_instruction + passages + tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = tone_dict[tone] + output_dict[output] + role_dict[role] + prompt_instruction + passages
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")
        elif order == "passage_first":
            if position == "beginning":
                prompt = passages + "Query: {query}\n" + role_dict[role] + instruction_dict[instruction] + \
                         tone_dict[tone] + output_dict[output]
            elif position == "ending":
                prompt = role_dict[role] + instruction_dict[instruction] + tone_dict[tone] + output_dict[
                    output] + passages + "Query: {query}\n"
            else:
                raise NotImplementedError(f"Position {position} is not implemented.")

    prompt_package = [output, prompt, role_dict[role], instruction_dict[instruction]]

    return prompt_package


class PointwiseLlmRanker(LlmRanker):
    """
    rerank each query on 100 passages
    """

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
                 output="output_1",
                 tone="tone_1",
                 order="query_first",
                 position="beginning",
                 role="False",
                 query_before_instruction="False",
                 role_at_beginning="True",
                 experiment_type="normal_run",
                 ablation_style="zero",
                 ablate_on="heads",
                 patching_pos="last",
                 top_k=10,
                 data_format="pointwise"):

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
        self.config = AutoConfig.from_pretrained(model_path, token=self.HF_TOKEN)
        if self.config.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                           token=self.HF_TOKEN,
                                                           use_fast=False,
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                              token=self.HF_TOKEN,
                                                              device_map='auto',
                                                              torch_dtype=torch.bfloat16,
                                                              low_cpu_mem_usage=True,
                                                              trust_remote_code=True,
                                                              )
            self.tokenizer.use_default_system_prompt = False
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:

        if self.method == "yes_no":

            if self.config.model_type == 'llama':
                # Prepare prompt.
                prompt_package = prompt_generator(self.prompt_type, self.original_prompt_number, self.instruction,
                                                  self.output, self.tone, self.order, self.position, self.role,
                                                  self.query_before_instruction, self.role_at_beginning,
                                                  self.experiment_type)
                if self.experiment_type in ["zero_mean_ablation", "zero_shot_ranking", "baseline_without_role"]:
                    prompt_tem = prompt_package[1]  # pass prompt content
                    role = prompt_package[2]
                    instruction = prompt_package[3]
                    # Prepare Yes/No token ids.
                    if prompt_package[0] == "output_1" or prompt_package[0] == "output_5":
                        yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                        no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                    else:
                        raise ValueError(f"Invalid type of output {prompt_package[0]}")

                    # Rerank with zero or mean ablation.
                    ##prepare dataloader for each query ##
                    data = []
                    dataloader = []
                    for doc in ranking:
                        prompt = prompt_tem.format(text=doc.text, query=query)
                        data.append(prompt)
                        role_range, query_range, doc_range, inst_range = get_pointwise_ranges(self.tokenizer, prompt,
                                                                                              role, query, doc.text,
                                                                                              instruction,
                                                                                              use_chat=True)

                        conversation = [{"role": "user", "content": prompt}]
                        input_ids = self.tokenizer.apply_chat_template(
                            conversation,
                            return_tensors="pt",
                            padding="longest",
                            max_length=2048,
                            truncation=True,
                            add_generation_prompt=True)
                        dataloader.append((input_ids, role_range, query_range, doc_range, inst_range))
                    # prepare ablate_heads #
                    attention_head_path = os.path.join(parent_dir, "attention_head_top_k.json")
                    with open(attention_head_path, "r") as f:
                        attention_head_top_k = json.load(f)
                    top_heads = attention_head_top_k[self.data_format]
                    # ablate_heads = {}
                    if self.patching_pos != 'mix':  # single position
                        ablate_heads = {
                            (layer, head): self.patching_pos
                            for layer, head in top_heads[self.patching_pos][:self.top_k]
                        }
                    else:  # mix all positions
                        ablate_heads = {}
                        for pos in top_heads.keys():
                            for layer, head in top_heads[pos][:self.top_k]:
                                ablate_heads[(layer, head)] = pos
                    if self.experiment_type == "zero_mean_ablation":
                        # call zero_mean_ablation method #
                        ablated_outputs = zero_mean_ablation(self.model,
                                                             dataloader,
                                                             self.ablation_style,
                                                             self.ablate_on,
                                                             ablate_heads,
                                                             correct_token_id=yes_id,
                                                             wrong_token_id=no_id,
                                                             mask_ablate_modules=True,  # fixed
                                                             n_samples=100,  # fixed
                                                             use_pos=True,  # fixed:all top 100 from first stage ranking
                                                             )

                    elif self.experiment_type in ["zero_shot_ranking", "baseline_without_role"]:
                        # call zero_shot_ranking method #
                        ablated_outputs = zero_shot_ranking(self.model,
                                                            dataloader,
                                                            correct_token_id=yes_id,
                                                            wrong_token_id=no_id,
                                                            n_samples=100,  # fixed
                                                            use_pos=True,  # fixed:all top 100 from first stage ranking
                                                            )
                    else:
                        raise ValueError(f"Invalid experiment type {self.experiment_type}")
                    for (i, score) in ablated_outputs:
                        ranking[i].score = score
                    # Rerank for normal runs.
                elif self.experiment_type == "normal_run":
                    prompt_tem = prompt_package[1]
                    conversation = [{"role": "user", "content": prompt_tem}]
                    prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                                add_generation_prompt=True)[17:]
                    data = [prompt.format(text=doc.text, query=query) for doc in ranking]
                    dataset = Text2TextGenerationDataset(data, self.tokenizer)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        collate_fn=DataCollatorWithPadding(
                            self.tokenizer,
                            max_length=2048,
                            padding='longest',
                        ),
                        shuffle=False,
                        drop_last=False,
                        num_workers=0
                    )

                    current_id = 0
                    with torch.no_grad():
                        for batch_inputs in tqdm(dataloader):
                            batch_inputs = batch_inputs.to(self.llm.device)
                            logits = self.model(input_ids=batch_inputs['input_ids'],
                                                attention_mask=batch_inputs['attention_mask'])

                            logits = logits.logits[:, -1, :]  # [batch_size, vocab_size]

                            yes_scores = logits[:, yes_id]
                            no_scores = logits[:, no_id]

                            batch_scores = torch.stack((yes_scores, no_scores), dim=1)  # [batch_size, 2]
                            batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)

                            scores = batch_scores[:, 0]
                            for score in scores:
                                ranking[current_id].score = score.item()
                                current_id += 1

        ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return ranking

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
