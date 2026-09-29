import os
import logging
import ir_datasets
from ranker import SearchResult
from pointwise_ranker import PointwiseLlmRanker
from pairwise_ranker import PairwiseLlmRanker
from tqdm import tqdm
import argparse
import random
import pandas as pd

random.seed(929)
logger = logging.getLogger(__name__)


def write_run_file(path, results, tag):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                f.write(f"{qid}\tQ0\t{doc.docid}\t{rank}\t{doc.score}\t{tag}\n")
                rank += 1


def save_patch_outputs(save_path: str, pair_name: str, patch_df: pd.DataFrame) -> None:
    os.makedirs(save_path, exist_ok=True)
    raw_csv = os.path.join(save_path, f"patch_metrics_{pair_name}.csv")
    summary_csv = os.path.join(save_path, f"patch_summary_{pair_name}.csv")
    patch_df.to_csv(raw_csv, index=False)

    group_cols = [c for c in ["layer", "pos", "head"] if c in patch_df.columns]
    if not group_cols:
        summary = patch_df.copy()
    else:
        summary = patch_df.groupby(group_cols, as_index=False).agg(
            clean_ld=("clean_ld", "mean"),
            corrupted_ld=("corrupted_ld", "mean"),
            patched_ld=("patched_ld", "mean"),
            normalized_ld=("normalized_ld", "mean"),
            clean_correct_prob=("clean_correct_prob", "mean"),
            corrupted_correct_prob=("corrupted_correct_prob", "mean"),
            patched_correct_prob=("patched_correct_prob", "mean"),
            ld_recovery=("ld_recovery", "mean"),
            prob_recovery=("prob_recovery", "mean"),
        )
    summary.to_csv(summary_csv, index=False)
    print(f"Saved patch metrics to {raw_csv}")
    print(f"Saved patch summary to {summary_csv}")


def load_pairs_tsv(pairs_path: str, doc_source: str):
    """
    TSV format:
    qid    pos_doc    pos_label    neg_doc    neg_label

    doc_source:
    - relevance   -> [pos_doc], pos_label
    - irrelevance -> [neg_doc], neg_label
    - combined    -> [pos_doc, neg_doc], {"pos_label": ..., "neg_label": ...}
    """
    df = pd.read_csv(pairs_path, sep="\t")

    required_cols = {"qid", "pos_doc", "pos_label", "neg_doc", "neg_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {pairs_path}: {missing}")

    qid_to_docids = {}
    qid_to_label = {}

    for _, row in df.iterrows():
        qid = str(row["qid"])

        if doc_source == "relevance":
            qid_to_docids[qid] = [str(row["pos_doc"])]
            qid_to_label[qid] = int(row["pos_label"])

        elif doc_source == "irrelevance":
            qid_to_docids[qid] = [str(row["neg_doc"])]
            qid_to_label[qid] = int(row["neg_label"])

        elif doc_source == "combined":
            qid_to_docids[qid] = [str(row["pos_doc"]), str(row["neg_doc"])]
            qid_to_label[qid] = {
                "pos_label": int(row["pos_label"]),
                "neg_label": int(row["neg_label"]),
            }

        else:
            raise ValueError(f"Unsupported doc_source={doc_source}")

    return qid_to_docids, qid_to_label

def main(args):
    experiment_type = args.experiment_type

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
            'data_format': args.data_format,
            'doc_source': args.doc_source,
        }

        if args.experiment_type == "zero_mean_ablation":
            ranker_kwargs.update({
                'ablation_style': args.ablation_style,
                'ablate_on': args.ablate_on,
                'patching_pos': args.patching_pos,
                'top_k': args.top_k,
            })
        elif args.experiment_type == "activation_patching":
            ranker_kwargs.update({
                'patch_activation': args.patch_activation,
                'patch_target': args.patch_target,
                'n_samples': args.n_samples,
                'clean_role': args.clean_role,
                'corrupt_role': args.corrupt_role,
                'patch_index_axis_names': tuple(args.patch_index_axis_names.split(',')),
                'patch_direction': args.patch_direction,
                'query_doc_limit': args.query_doc_limit,
            })

        ranker = PointwiseLlmRanker(**ranker_kwargs)

    elif args.data_format == 'pairwise':
        if args.method != 'allpair':
            args.batch_size = 2
            logger.info('Setting batch_size to 2.')
        ranker = PairwiseLlmRanker(
            model_name=args.model_name,
            model_path=args.model_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            device=args.device,
            method=args.method,
            batch_size=args.batch_size,
            k=args.k,
            hf_token=args.hf_token,
            prompt_type=args.prompt_type,
            original_prompt_number=args.original_prompt_number,
            instruction=args.instruction,
            output=args.output,
            tone=args.tone,
            order=args.order,
            position=args.position,
            role=args.role,
            query_before_instruction=args.query_before_instruction,
            role_at_beginning=args.role_at_beginning,
            experiment_type=args.experiment_type,
            data_format=args.data_format,
        )
    else:
        raise ValueError('Must specify either --data_format pointwise or --data_format pairwise.')

    # -------------------------
    # load dataset queries/docs
    # -------------------------
    query_map = {}
    if args.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.ir_dataset_name)
        for query in dataset.queries_iter():
            query_map[str(query.query_id)] = ranker.truncate(query.text, args.query_length)
        dataset = ir_datasets.load(args.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        raise ValueError("This aligned patching script expects --ir_dataset_name")

    # Controlled pairs.tsv input.
    if args.pairs_path is not None:
        logger.info(f'Loading controlled pairs from {args.pairs_path} (doc_source={args.doc_source}).')

        qid_to_docids, qid_to_label = load_pairs_tsv(args.pairs_path, args.doc_source)

        first_stage_rankings = []
        for qid, docids in qid_to_docids.items():
            if qid not in query_map:
                print(f"[WARN] qid={qid} not found in dataset queries, skipping.")
                continue

            ranking = []
            skip_this_qid = False

            for docid in docids:
                try:
                    doc = docstore.get(docid)
                except Exception as e:
                    print(f"[WARN] failed to fetch docid={docid} for qid={qid}: {e}")
                    skip_this_qid = True
                    break

                text = doc.text
                if hasattr(doc, "title"):
                    text = f"{doc.title} {text}"
                text = ranker.truncate(text, args.passage_length)

                # Score is a placeholder for combined inputs.
                if args.doc_source == "combined":
                    score = 0.0
                else:
                    score = float(qid_to_label[qid])

                ranking.append(SearchResult(docid=docid, score=score, text=text))

            if skip_this_qid:
                continue

            first_stage_rankings.append((qid, query_map[qid], ranking))

        print(f"[INFO] Loaded {len(first_stage_rankings)} controlled query-doc sets.")

    # BM25 run input.
    else:
        logger.info(f'Loading first stage run from {args.run_path}.')
        first_stage_rankings = []
        with open(args.run_path, 'r') as f:
            current_qid = None
            current_ranking = []
            for line in tqdm(f):
                qid, _, docid, _, score, _ = line.strip().split()
                qid = str(qid)

                if qid != current_qid:
                    if current_qid is not None:
                        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.hits]))
                    current_ranking = []
                    current_qid = qid

                if len(current_ranking) >= args.hits:
                    continue

                doc = docstore.get(docid)
                text = doc.text
                if hasattr(doc, "title"):
                    text = f"{doc.title} {text}"
                text = ranker.truncate(text, args.passage_length)
                current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))

            if current_qid is not None:
                first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.hits]))

    # -------------------------
    # run experiment
    # -------------------------
    if experiment_type == "activation_patching":
        source_tag = "pairs" if args.pairs_path is not None else "run"
        doc_tag = args.doc_source if args.pairs_path is not None else "topk"
        pair_name = (
            f"{source_tag}_{doc_tag}__"
            f"clean_{args.clean_role}__corrupt_{args.corrupt_role}__"
            f"{args.patch_target}__{args.patch_activation}"
        )

        all_patch_dfs = []
        for i, (qid, query, ranking) in enumerate(first_stage_rankings):
            print(f"Processing patching query {i + 1} / {len(first_stage_rankings)}")
            patch_df = ranker.patch_query(qid, query, ranking)
            patch_df.insert(0, "qid", qid)
            if args.pairs_path is not None:
                patch_df.insert(1, "doc_source", args.doc_source)

                label_info = qid_to_label[qid]
                if args.doc_source == "combined":
                    patch_df.insert(2, "pos_label", label_info["pos_label"])
                    patch_df.insert(3, "neg_label", label_info["neg_label"])
                else:
                    patch_df.insert(2, "doc_label", label_info)
            all_patch_dfs.append(patch_df)

        full_df = pd.concat(all_patch_dfs, ignore_index=True)
        save_patch_outputs(args.save_path, pair_name, full_df)
        return

    reranked_results = []
    for i, (qid, query, ranking) in enumerate(first_stage_rankings):
        print(f"Processing query {i + 1} / {len(first_stage_rankings)}")
        reranked_results.append((qid, query, ranker.rerank(query, ranking)))

    os.makedirs(args.save_path, exist_ok=True)
    write_run_file(f"{args.save_path}/results_{args.role}.csv", reranked_results, 'LLMRankers')
    print(f'Reranked run saved to {args.save_path}')


def add_common_args(parser):
    parser.add_argument('--run_path', type=str, required=False, default=None)

    parser.add_argument('--pairs_path', type=str, default=None,
                        help='Optional controlled pairs TSV. If set, use this instead of run_path.')
    parser.add_argument('--doc_source', type=str, default='relevance',
                        choices=['relevance', 'irrelevance', 'combined'],
                        help='When using pairs_path, choose relevance (best relevant), irrelevance (fixed random label0), or combined (both pos_doc and neg_doc).')

    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--tokenizer_name_or_path', type=str, default=None)
    parser.add_argument('--ir_dataset_name', type=str, default=None)
    parser.add_argument('--hits', type=int, default=100)
    parser.add_argument('--query_length', type=int, default=20)
    parser.add_argument('--passage_length', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    parser.add_argument('--hf_token', type=str, default=None)

    parser.add_argument('--data_format', type=str, required=True, choices=['pointwise', 'pairwise'])
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)

    parser.add_argument('--prompt_type', type=str, default="adjusted", choices=['original', 'adjusted'])
    parser.add_argument('--original_prompt_number', type=int, default=1)
    parser.add_argument('--instruction', type=str, default="instruction_1")
    parser.add_argument('--output', type=str, default="output_5")
    parser.add_argument('--tone', type=str, default="tone_1")
    parser.add_argument('--order', type=str, default="query_first", choices=['query_first', 'passage_first'])
    parser.add_argument('--position', type=str, default="beginning", choices=['beginning', 'ending'])
    parser.add_argument('--role', type=str, default="role_0")
    parser.add_argument('--query_before_instruction', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--role_at_beginning', type=str, default="True", choices=['True', 'False'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Ranking System with Different Experiment Types')
    commands = parser.add_subparsers(dest='experiment_type', required=True)

    normal_run_parser = commands.add_parser("normal_run")
    add_common_args(normal_run_parser)

    zero_ablation_parser = commands.add_parser("zero_mean_ablation")
    add_common_args(zero_ablation_parser)
    zero_ablation_parser.add_argument("--ablation_style", type=str, default="zero", choices=["mean", "zero"])
    zero_ablation_parser.add_argument("--ablate_on", type=str, default="heads", choices=["heads", "layers"])
    zero_ablation_parser.add_argument("--patching_pos", type=str, default="last")
    zero_ablation_parser.add_argument("--top_k", type=int, default=10)

    zero_shot_ranking_parser = commands.add_parser("zero_shot_ranking")
    add_common_args(zero_shot_ranking_parser)

    baseline_without_role_parser = commands.add_parser("baseline_without_role")
    add_common_args(baseline_without_role_parser)

    activation_patching_parser = commands.add_parser("activation_patching")
    add_common_args(activation_patching_parser)
    activation_patching_parser.add_argument('--clean_role', type=str, required=True)
    activation_patching_parser.add_argument('--corrupt_role', type=str, required=True)
    activation_patching_parser.add_argument('--patch_activation', type=str, default='resid_pre',
                                           choices=['resid_pre', 'resid_post', 'attn_out', 'mlp_out', 'z'])
    activation_patching_parser.add_argument('--patch_target', type=str, required=True,
                                           choices=['role_adj', 'role_adv', 'role_adj_adv', 'role_modal',
                                                    'role_all', 'query_all', 'doc_all', 'inst_all', 'last'])
    activation_patching_parser.add_argument('--patch_index_axis_names', type=str, default='layer,pos')
    activation_patching_parser.add_argument('--patch_direction', type=str, default='clean_to_corrupt',
                                           choices=['clean_to_corrupt', 'corrupt_to_clean'])
    activation_patching_parser.add_argument('--n_samples', type=int, default=100)
    activation_patching_parser.add_argument('--query_doc_limit', type=int, default=1,
                                           help='uses one document per query for pointwise patching')

    args = parser.parse_args()

    if args.pairs_path is None and args.run_path is None:
        raise ValueError("You must provide either --run_path or --pairs_path.")

    main(args)
