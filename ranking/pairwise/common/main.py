import os
import csv
import logging
import ir_datasets
from ranker import LlmRanker, SearchResult
from pointwise_ranker import PointwiseLlmRanker
from pairwise_ranker import PairwiseLlmRanker
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
from dataclasses import dataclass

random.seed(929)
logger = logging.getLogger(__name__)


def write_run_file(path, results, tag):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):
    experiment_type = args.experiment_type

    output_file_path = f"{args.save_path}/results_{args.role}.csv"
    if os.path.exists(output_file_path):
        return

    if args.data_format == 'pointwise':
        ranker_kwargs = {
            'model_name': args.model_name,
            'model_path': args.model_path,
            'tokenizer_name_or_path': args.tokenizer_name_or_path,
            'device': args.device,
            'method': args.method,
            'batch_size': args.batch_size,
            'hf_token': args.hf_token,
            'prompt_type': args.prompt_type,
            'original_prompt_number': args.original_prompt_number,
            'instruction': args.instruction,
            'output': args.output,
            'tone': args.tone,
            'order': args.order,
            'position': args.position,
            'role': args.role,
            'query_before_instruction': args.query_before_instruction,
            'role_at_beginning': args.role_at_beginning,
            'experiment_type': args.experiment_type,
            'data_format': args.data_format
        }

        if args.experiment_type == "zero_mean_ablation":
            ranker_kwargs.update({
                'ablation_style': args.ablation_style,
                'ablate_on': args.ablate_on,
                'patching_pos': args.patching_pos,
                'top_k': args.top_k
            })

        elif args.experiment_type == "activation_patching":
            ranker_kwargs.update({
                'patch_activation': args.patch_activation,
                'patch_target': args.patch_target,
                'n_samples': args.n_samples
            })

        ranker = PointwiseLlmRanker(**ranker_kwargs)

    elif args.data_format == 'pairwise':
        if args.method != 'allpair':
            args.batch_size = 2
            logger.info('Setting batch_size to 2.')

        ranker_kwargs = {
            'model_name': args.model_name,
            'model_path': args.model_path,
            'tokenizer_name_or_path': args.tokenizer_name_or_path,
            'device': args.device,
            'method': args.method,
            'batch_size': args.batch_size,
            'k': args.k,
            'hf_token': args.hf_token,
            'prompt_type': args.prompt_type,
            'original_prompt_number': args.original_prompt_number,
            'instruction': args.instruction,
            'output': args.output,
            'tone': args.tone,
            'order': args.order,
            'position': args.position,
            'role': args.role,
            'query_before_instruction': args.query_before_instruction,
            'role_at_beginning': args.role_at_beginning,
            'experiment_type': args.experiment_type,
            'data_format': args.data_format
        }

        if args.experiment_type == "zero_mean_ablation":
            ranker_kwargs.update({
                'ablation_style': args.ablation_style,
                'ablate_on': args.ablate_on,
                'patching_pos': args.patching_pos,
                'top_k': args.top_k
            })

        # Activation patching is supported only for pointwise ranking here.
        elif args.experiment_type == "activation_patching":
            raise ValueError("activation_patching is currently only supported for --data_format pointwise.")

        ranker = PairwiseLlmRanker(**ranker_kwargs)

    else:
        raise ValueError('Must specify either --data_format pointwise or --data_format pairwise.')

    if experiment_type == "zero_mean_ablation":
        print(f"Running zero/mean ablation experiment with style: {args.ablation_style}")
        print(f"Ablating on: {args.ablate_on}")
        if hasattr(args, 'patching_pos') and args.patching_pos:
            print(f"Patching position: {args.patching_pos}")
        if hasattr(args, 'top_k') and args.top_k:
            print(f"Top-k heads: {args.top_k}")

    elif experiment_type == "activation_patching":
        print("Running activation patching experiment")
        print(f"Patch activation: {args.patch_activation}")
        print(f"Patch target: {args.patch_target}")
        print(f"n_samples: {args.n_samples}")

    elif experiment_type == "zero_shot_ranking":
        print("Running zero-shot ranking experiment")

    elif experiment_type == "baseline_without_role":
        print("Running baseline without role experiment")

    elif experiment_type == "normal_run":
        print("Running normal ranking experiment")

    # Load query_map and docstore from the dataset name.
    query_map = {}
    if args.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.ir_dataset_name)
        for query in dataset.queries_iter():
            qid = query.query_id
            text = query.text
            query_map[qid] = ranker.truncate(text, args.query_length)
        dataset = ir_datasets.load(args.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        topics = get_topics(args.pyserini_index + '-test')
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.pyserini_index + '.flat')

    # Load and filter first-stage ranking results.
    logger.info(f'Loading first stage run from {args.run_path}.')
    first_stage_rankings = []
    with open(args.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()

            if args.ir_dataset_name == "msmarco-passage/trec-dl-2020":
                current_path = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(current_path, "queries.dl2020.tsv")
                with open(file_path, 'r') as f2:
                    reader = csv.reader(f2, delimiter='\t')
                    first_column = [row[0] for row in reader]

                if qid != current_qid:
                    if current_qid is not None and current_qid in first_column:
                        first_stage_rankings.append(
                            (current_qid, query_map[current_qid], current_ranking[:args.hits])
                        )
                    current_ranking = []
                    current_qid = qid

                if len(current_ranking) >= args.hits:
                    continue

                if args.ir_dataset_name is not None:
                    text = docstore.get(docid).text
                    if 'title' in dir(docstore.get(docid)):
                        text = f'{docstore.get(docid).title} {text}'
                else:
                    data = json.loads(docstore.doc(docid).raw())
                    text = data['text']
                    if 'title' in data:
                        text = f'{data["title"]} {text}'

                text = ranker.truncate(text, args.passage_length)
                current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))

            else:
                if qid != current_qid:
                    if current_qid is not None:
                        first_stage_rankings.append(
                            (current_qid, query_map[current_qid], current_ranking[:args.hits])
                        )
                    current_ranking = []
                    current_qid = qid

                if len(current_ranking) >= args.hits:
                    continue

                if args.ir_dataset_name is not None:
                    text = docstore.get(docid).text
                    if 'title' in dir(docstore.get(docid)):
                        text = f'{docstore.get(docid).title} {text}'
                else:
                    data = json.loads(docstore.doc(docid).raw())
                    text = data['text']
                    if 'title' in data:
                        text = f'{data["title"]} {text}'

                text = ranker.truncate(text, args.passage_length)
                current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))

        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.hits]))

    reranked_results = []
    # Run reranking.
    for i, (qid, query, ranking) in enumerate(first_stage_rankings):
        print(f"Processing query {i + 1} / {len(first_stage_rankings)}")
        if args.shuffle_ranking is not None:
            if args.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.shuffle_ranking}.')
        reranked_results.append((qid, query, ranker.rerank(query, ranking)))

    write_run_file(f"{args.save_path}/results_{args.role}.csv", reranked_results, 'LLMRankers')
    print(f'Reranked run saved to {args.save_path} \n')


def add_common_args(parser):
    parser.add_argument('--run_path', type=str, required=True,
                        help='Path to the first stage run file (TREC format) to rerank.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the reranked run file (TREC format).')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Path to the pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                        help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    parser.add_argument('--ir_dataset_name', type=str, default=None)
    parser.add_argument('--pyserini_index', type=str, default=None)
    parser.add_argument('--hits', type=int, default=100)
    parser.add_argument('--query_length', type=int, default=20)
    parser.add_argument('--passage_length', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    parser.add_argument('--hf_token', type=str, default=None)

    parser.add_argument('--data_format', type=str, required=True, choices=['pointwise', 'pairwise'],
                        help='Data format: pointwise or pairwise')
    parser.add_argument('--method', type=str, required=True,
                        help='yes_no for pointwise, allpair, heapsort, bubblesort for pairwise')
    parser.add_argument('--batch_size', type=int, default=1, help='2 for normal run, 1 for patching ablation')
    parser.add_argument('--k', type=int, default=10, help='k for pairwise only')

    parser.add_argument('--prompt_type', type=str, default="adjusted", choices=['original', 'adjusted'])
    parser.add_argument('--original_prompt_number', type=int, default=1)
    parser.add_argument('--instruction', type=str, default="instruction_1")
    parser.add_argument('--output', type=str, default="output_1")
    parser.add_argument('--tone', type=str, default="tone_1")
    parser.add_argument('--order', type=str, default="query_first", choices=['query_first', 'passage_first'])
    parser.add_argument('--position', type=str, default="beginning", choices=['beginning', 'ending'])
    parser.add_argument('--role', type=str, default="role_0")
    parser.add_argument('--query_before_instruction', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--role_at_beginning', type=str, default="True", choices=['True', 'False'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Ranking System with Different Experiment Types')
    commands = parser.add_subparsers(dest='experiment_type', required=True,
                                     help='Choose experiment type')

    # 1. Normal Run
    normal_run_parser = commands.add_parser("normal_run", help="Normal ranking run without ablation")
    add_common_args(normal_run_parser)

    # 2. Zero/Mean Ablation
    zero_ablation_parser = commands.add_parser("zero_mean_ablation", help="Zero or mean ablation experiments")
    add_common_args(zero_ablation_parser)
    zero_ablation_parser.add_argument("--ablation_style", type=str, default="zero", choices=["mean", "zero"],
                                      help="Ablation style: mean or zero")
    zero_ablation_parser.add_argument("--ablate_on", type=str, default="heads", choices=["heads", "layers"],
                                      help="Ablate on heads or layers")
    zero_ablation_parser.add_argument("--patching_pos", type=str, default="last", help="Position for patching")
    zero_ablation_parser.add_argument("--top_k", type=int, default=10, help="Top k heads to be ablated")

    # 3. Activation Patching
    activation_patching_parser = commands.add_parser("activation_patching", help="Activation patching experiments")
    add_common_args(activation_patching_parser)
    activation_patching_parser.add_argument("--patch_activation", type=str, default="resid_pre",
                                            choices=["resid_pre", "z"])
    activation_patching_parser.add_argument("--patch_target", type=str, required=True,
                                            choices=["role_adj", "role_adv"])
    activation_patching_parser.add_argument("--n_samples", type=int, default=100)

    # 4. Zero Shot Ranking
    zero_shot_ranking_parser = commands.add_parser("zero_shot_ranking", help="Zero shot ranking experiments")
    add_common_args(zero_shot_ranking_parser)

    # 5. Baseline Without Role
    baseline_without_role_parser = commands.add_parser("baseline_without_role",
                                                       help="Baseline without role experiments")
    add_common_args(baseline_without_role_parser)

    args = parser.parse_args()

    if hasattr(args, 'ir_dataset_name') and hasattr(args, 'pyserini_index'):
        if args.ir_dataset_name is not None and args.pyserini_index is not None:
            raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    main(args)
