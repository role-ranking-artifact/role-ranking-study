from typing import List, Tuple
from ranker import LlmRanker, SearchResult
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
import copy
import torch
import json
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_structure import get_pairwise_ranges
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except Exception:
    T5Tokenizer = None
    T5ForConditionalGeneration = None
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
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
from methods import zero_mean_ablation, zero_shot_ranking, get_mean_activations
import itertools


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
    neg_adverb = ["poorly", "wrongly","mistakenly", "improperly", "falsely",
                "terribly", "badly", "incorrectly", "sadly", "horribly"]
    # role template
    role_template = "You are {adjective} search assistant that {modal} rank passages {adverb}, basedon their relevance to a query. \n"

    role_dict = {}
    i = 0
    for adjective in pos_adjective:
        for modal in pos_modal:
            for adverb in pos_adverb:
                role_dict[f"role_{i}"] = role_template.format(adjective=adjective, modal=modal,adverb=adverb)
                i += 1
    for adjective in neg_adjective:
        for modal in neg_modal:
            for adverb in neg_adverb:
                role_dict[f"role_{i}"] = role_template.format(adjective=adjective, modal=modal,adverb=adverb)
                i += 1
    return role_dict

def prompt_generator(prompt_type, original_prompt_number, instruction, 
                        output, tone, order, position, role, 
                        query_before_instruction, role_at_beginning, experiment_type):
    if prompt_type == "original":
        original_prompt_dict = {
            1: "Given a query: {query}, which of the following two passages is more relevant to the query? \n"
               "Passage A: {doc1} \n"
               "Passage B: {doc2} \n"
               "Output Passage A or Passage B:"
        }
        return original_prompt_dict[original_prompt_number]

    elif prompt_type == "adjusted":
        instruction_dict = {
            "instruction_1": 'Given a query "{query}", '
                             'which of the following two passages is more relevant to the query?\n',
            "instruction_2": "Is document A more relevant to the query?\n"
        }

        output_dict = {
            "output_1": "Output Passage A or Passage B. ",
            "output_2": "Answer 'Yes' or 'No'.\nAnswer: ",
        }

        tone_dict = {
            "tone_0": "",
            "tone_1": "Please ",
            "tone_2": "Only ",
            "tone_3": "Must ",
            "tone_4": "You better get this right or you will be punished. ",
            "tone_5": "Only response the ranking results, do not say any word or explain. "
        }

        #### define role_dict based on experiment type
        if experiment_type == "normal_run":
            role_dict = role_dict_generator()
        elif experiment_type in ["zero_mean_ablation", "zero_shot_ranking"]:
            role_dict_path = os.path.join(parent_dir, "role_dict", "meta-llama.json")
            with open(role_dict_path, "r") as f:
                dict = json.load(f)
            role_dict = dict["pos_roles"].copy()
            for i, key in enumerate(dict['neg_roles'].keys()):
                role_dict[f"role_{i+1+len(dict['pos_roles'].keys())}"] = dict['neg_roles'][key]
        elif experiment_type == "baseline_without_role":
            role_dict = {"role_0": ""}

        if query_before_instruction == "True":
            prompt_instruction = "Query: {query}\n" + instruction_dict[instruction]
        else:
            prompt_instruction = instruction_dict[instruction] + "Query: {query}\n"
        
        passages = "\nPassage A: {doc1} \nPassage B: {doc2} \n"

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
                    prompt = role_dict[role] + passages + "Query: {query}\n" + instruction_dict[instruction] + tone_dict[tone] + output_dict[output]
                elif position == "ending":
                    prompt = role_dict[role] + tone_dict[tone] + output_dict[output] + passages + prompt_instruction
                else:
                    raise NotImplementedError(f"Position {position} is not implemented.")
            else:
                raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")
        else: # role directly before instruction
            if order == "query_first":
                if position == "beginning":
                    prompt = role_dict[role] + prompt_instruction + passages + tone_dict[tone] + output_dict[output]
                elif position == "ending":
                    prompt = tone_dict[tone] + output_dict[output] + role_dict[role] + prompt_instruction + passages
                else:
                    raise NotImplementedError(f"Position {position} is not implemented.")
            elif order == "passage_first":
                if position == "beginning": # doc + query + role + instruction +output
                    prompt = passages + "Query: {query}\n" + role_dict[role] + instruction_dict[instruction] + tone_dict[tone] + output_dict[output]
                elif position == "ending":
                    prompt = role_dict[role] + instruction_dict[instruction] + tone_dict[tone] + output_dict[output] + passages + "Query: {query}\n"
                else:
                    raise NotImplementedError(f"Order {order} is not implemented, position is {position}.")

        # return prompt
        prompt_package = [output, prompt, role_dict[role], instruction_dict[instruction]]

        return prompt_package

class Text2TextGenerationDataset(Dataset):
    def __init__(self, data: List[str], tokenizer):
        self.data = tokenizer(data)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, item):
        return {'input_ids': self.data['input_ids'][item],
                'attention_mask': self.data['attention_mask'][item]}


class PairwiseLlmRanker(LlmRanker):
    def __init__(self,
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
                 order="passage_first",
                 position="beginning",
                 role="False",
                 query_before_instruction="True",
                 role_at_beginning="True",
                 experiment_type="normal_run",
                 ablation_style="zero",
                 ablate_on="heads",
                 patching_pos="last",
                 top_k=10,
                 data_format="pairwise"
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
        self.ablation_style = ablation_style
        self.ablate_on = ablate_on
        self.patching_pos = patching_pos
        self.top_k = top_k
        self.data_format = data_format
        self.HF_TOKEN = hf_token
        self.config = AutoConfig.from_pretrained(model_path, token=self.HF_TOKEN)

        self.mean_activations = None

        if self.config.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                           token=self.HF_TOKEN,
                                                           use_fast=False,
                                                           trust_remote_code=True)
            self.tokenizer.use_default_system_prompt = False
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                            token=self.HF_TOKEN,
                                                            device_map='auto',
                                                            torch_dtype=torch.bfloat16)
            if self.experiment_type == "zero_mean_ablation":
                self.model = HookedTransformer.from_pretrained_no_processing(
                    model_name,
                    dtype="bfloat16",
                    device="cuda",
                    hf_model=self.model,
                    tokenizer=self.tokenizer)
        else:
            raise NotImplementedError

    def mean_activations_for_each_query(
        self,
        model: HookedTransformer,
        query: str,
        rankings: List[SearchResult],
        k: int # top k docs for reranking
    ):
        """
        Dataloader: 
        for each query, get the mean activations of the top k docs pairwise ranking results,
        for each pair of docs, also count the swapping results;
        e.g. for query 1, top 3 docs: there are 3 pairs of docs: (doc1, doc2), (doc1, doc3), (doc2, doc3),
        and 3 swapping results: (doc2, doc1), (doc3, doc1), (doc3, doc2);
        Return:
        for each layer: 
        note: attn_hook_z: [batch, seq_len, n_heads, d_head]
        note: resid_pre_hook: [batch, seq_len, d_model]
        note: d_head = d_model // n_heads
        note: sum over the batch dimension, then average over sequence length
        mean_activations: {layer: {"z": [n_heads, d_head], 
                                    "resid_pre": [d_model]}}
        """
        ### construct dataloader ###
        prompt_package = prompt_generator(self.prompt_type, self.original_prompt_number, self.instruction,
                                         self.output, self.tone, self.order, self.position, self.role,
                                         self.query_before_instruction, self.role_at_beginning,
                                         self.experiment_type)
        prompt_tem = prompt_package[1]
        role = prompt_package[2]
        instruction = prompt_package[3]
        dataloader = []
        # all the possible pairs of docs of top k docs: 
        # consider orders: permutation of top k docs; 
        # do not consider order: combinations of top k docs
        for doc_1, doc_2 in itertools.combinations(rankings[:k], 2):
            prompt0 = prompt_tem.format(query=query, doc1=doc_1.text, doc2=doc_2.text)
            prompt1 = prompt_tem.format(query=query, doc1=doc_2.text, doc2=doc_1.text)
            role_range_0, query_range_0, doc_range_0, inst_range_0 = get_pairwise_ranges(self.tokenizer, prompt0, role, query, doc_1.text, doc_2.text, instruction, use_chat=True)
            role_range_1, query_range_1, doc_range_1, inst_range_1 = get_pairwise_ranges(self.tokenizer, prompt1, role, query, doc_2.text, doc_1.text, instruction, use_chat=True)
            conversation0 = [{"role": "user", "content": prompt0}]
            conversation1 = [{"role": "user", "content": prompt1}]
            input_ids_0 = self.tokenizer.apply_chat_template(conversation0, return_tensors="pt", padding="longest", max_length=2048, truncation=True, add_generation_prompt=True)
            input_ids_1 = self.tokenizer.apply_chat_template(conversation1, return_tensors="pt", padding="longest", max_length=2048, truncation=True, add_generation_prompt=True)
            dataloader.append((input_ids_0, role_range_0, query_range_0, doc_range_0, inst_range_0))
            dataloader.append((input_ids_1, role_range_1, query_range_1, doc_range_1, inst_range_1))
            
        #### get mean activations ####
        self.mean_activations = get_mean_activations(
                                            model, 
                                            dataloader,
                                            n_samples=len(dataloader),
                                            use_pos=True) # fixed
        return

    def compare(self, query: str, docs: List):

        if self.config.model_type == 'llama':

            prompt_package = prompt_generator(self.prompt_type, self.original_prompt_number, self.instruction,
                                         self.output, self.tone, self.order, self.position, self.role,
                                         self.query_before_instruction, self.role_at_beginning,
                                         self.experiment_type)
            if self.experiment_type == "zero_mean_ablation":
                prompt_tem = prompt_package[1]  # pass prompt content
                role = prompt_package[2]
                instruction = prompt_package[3]

                # Prepare Yes/No token ids.
                yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                
                # Prepare dataloader for swapped document orders.
                dataloader_swap = []   
                doc1, doc2 = docs[0], docs[1]
                prompt0 = prompt_tem.format(query=query, doc1=doc1, doc2=doc2)
                prompt1 = prompt_tem.format(query=query, doc1=doc2, doc2=doc1)
                role_range_0, query_range_0, doc_range_0, inst_range_0 = get_pairwise_ranges(self.tokenizer, prompt0, role, query, doc1, doc2, instruction, use_chat=True)
                role_range_1, query_range_1, doc_range_1, inst_range_1 = get_pairwise_ranges(self.tokenizer, prompt1, role, query, doc2, doc1, instruction, use_chat=True)
                conversation0 = [{"role": "user", "content": prompt0}]
                conversation1 = [{"role": "user", "content": prompt1}]
                input_ids_0 = self.tokenizer.apply_chat_template(
                            conversation0, 
                            return_tensors="pt",
                            padding="longest",
                            max_length=2048,
                            truncation=True,
                            add_generation_prompt=True)
                input_ids_1 = self.tokenizer.apply_chat_template(
                            conversation1, 
                            return_tensors="pt",
                            padding="longest",
                            max_length=2048,
                            truncation=True,
                            add_generation_prompt=True)
                dataloader_swap.append((input_ids_0, role_range_0, query_range_0, doc_range_0, inst_range_0))
                dataloader_swap.append((input_ids_1, role_range_1, query_range_1, doc_range_1, inst_range_1))

                # Prepare ablated heads.
                attention_head_path = os.path.join(parent_dir, "attention_head_top_k.json")
                with open(attention_head_path, "r") as f:
                    attention_head_top_k = json.load(f)
                top_heads = attention_head_top_k[self.data_format]
                if self.patching_pos != 'mix': # single position
                    ablate_heads = {
                        (layer, head): self.patching_pos
                        for layer, head in top_heads[self.patching_pos][:self.top_k]
                    }
                else: # mix all positions
                    ablate_heads = {}
                    for pos in top_heads.keys():
                        for layer, head in top_heads[pos][:self.top_k]:
                            ablate_heads[(layer, head)] = pos
                if self.experiment_type == "zero_mean_ablation":
                    # call zero_ablation method #
                    if self.ablation_style == "zero":
                        ablated_outputs = zero_mean_ablation(self.model, 
                                       dataloader_swap, 
                                       self.ablation_style, 
                                       self.ablate_on, 
                                       ablate_heads, 
                                       correct_token_id=yes_id,
                                       wrong_token_id=no_id,
                                       mask_ablate_modules=True, #fixed
                                       n_samples=100, # fixed
                                       use_pos=True, # fixed:all top 100 from first stage ranking
                                       )
                        
                    elif self.ablation_style == "mean":
                        ablated_outputs = zero_mean_ablation(self.model, 
                                       dataloader_swap, 
                                       self.ablation_style, 
                                       self.ablate_on, 
                                       ablate_heads, 
                                       correct_token_id=yes_id,
                                       wrong_token_id=no_id,
                                       mean_activations=self.mean_activations,
                                       mask_ablate_modules=True, #fixed
                                       n_samples=100, # fixed
                                       use_pos=True, # fixed
                                       )
                    scores = [score for (_, score) in ablated_outputs]
                    assert len(scores) == len(dataloader_swap) == 2
                    return scores   
            

            elif self.experiment_type in ["normal_run", "zero_shot_ranking", "baseline_without_role"]:
                doc1, doc2 = docs[0], docs[1]
                input_texts = [prompt_package[1].format(query=query, doc1=doc1, doc2=doc2),
                               prompt_package[1].format(query=query, doc1=doc2, doc2=doc1)]

                conversation0 = [{"role": "user", "content": input_texts[0]}]
                conversation1 = [{"role": "user", "content": input_texts[1]}]

                prompt0 = self.tokenizer.apply_chat_template(conversation0, tokenize=False, add_generation_prompt=True)
                prompt1 = self.tokenizer.apply_chat_template(conversation1, tokenize=False, add_generation_prompt=True)

                tokenized = self.tokenizer([prompt0, prompt1], return_tensors="pt", padding="longest")
                input_ids = tokenized.input_ids.to(self.device)
                attention_mask = tokenized.attention_mask.to(self.device)

                yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

                with torch.no_grad():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = logits.logits[:, -1, :] # [2, vocab_size] last token

                    yes_scores = logits[:, yes_id]
                    no_scores = logits[:, no_id] 

                    batch_scores = torch.stack((yes_scores, no_scores), dim=1)
                    batch_scores = torch.nn.functional.softmax(batch_scores, dim=1) 
                    scores = batch_scores[:, 0].tolist() 

                return scores

        else:
            raise NotImplementedError

        return scores

    def heapify(self, arr, n, i):
        # Find largest among root and children
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[i]: 
            largest = l

        if r < n and arr[r] > arr[largest]:
            largest = r

        # If root is not largest, swap with largest and continue heapifying
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // 2, -1, -1):
            self.heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        # for mean ablation: compute mean activations for each query
        if self.experiment_type == "zero_mean_ablation" and self.ablation_style == "mean":
            self.mean_activations_for_each_query(self.model, query, ranking, self.k)

        if self.method == "heapsort":
            class ComparableDoc:
                def __init__(self, docid, text, ranker):
                    self.docid = docid
                    self.text = text
                    self.ranker = ranker

                def __gt__(self, other): 
                    out = self.ranker.compare(query, [self.text, other.text])
                    if self.ranker.instruction == "instruction_2":
                        prompt0_yes_score, prompt1_yes_score = out[0], out[1]
                        if prompt0_yes_score > prompt1_yes_score:
                            return True
                        else:
                            return False
                    else:
                        # Original Passage A/B format
                        if out[0] == "Passage A" and out[1] == "Passage B":
                            return True
                        else:
                            return False

            arr = [ComparableDoc(docid=doc.docid, text=doc.text, ranker=self) for doc in ranking]
            self.heapSort(arr, self.k)
            ranking = [SearchResult(docid=doc.docid, score=-i, text=None) for i, doc in enumerate(reversed(arr))]

        elif self.method == "bubblesort":
            k = min(self.k, len(ranking))

            last_end = len(ranking) - 1
            for i in range(k):
                current_ind = last_end
                is_change = False
                while True:
                    if current_ind <= i:
                        break
                    doc1 = ranking[current_ind]
                    doc2 = ranking[current_ind - 1]
                    output = self.compare(query, [doc1.text, doc2.text])
                    if output[0] == "Passage A" and output[1] == "Passage B":
                        ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]

                        if not is_change:
                            is_change = True
                            if last_end != len(ranking) - 1:  # skip unchanged pairs at the bottom
                                last_end += 1
                    if not is_change:
                        last_end -= 1
                    current_ind -= 1
        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')
        results = []
        top_doc_ids = set()
        rank = 1
        # Add the top-k documents to the result list using negative rank as score.
        for i, doc in enumerate(ranking[:self.k]):
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
