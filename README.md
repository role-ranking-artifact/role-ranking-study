# Role Ranking Artifact

This repository contains the code and supporting materials for the ECIR artifact accompanying "Tracing Role Effects in LLM Rankers: From Lexical Sensitivity to Internal Dependencies."

## Repository Structure

- `ranking/pointwise/`: pointwise Yes/No ranking experiments.
- `ranking/pairwise/`: pairwise Yes/No ranking experiments.
- `patching/pointwise/`: pointwise activation-patching and three-stage path-patching code.
- `patching/pairwise/`: Llama DL20 pairwise activation-patching and path-patching code.
- `roles/`: role prompt files used by the ranking and patching launchers.
- `data/pairs/`: controlled query-document pairs for mechanistic experiments.
- `data/runs/`: first-stage retrieval runs used as ranking inputs.
- `figures/supplementary/`: supplementary result figures not included in the main paper.

Pointwise ranking keeps small model-specific adapter directories under
`ranking/pointwise/{llama,mistral,qwen}/`, while pairwise ranking uses a
shared implementation under `ranking/pairwise/common/`; in both cases the
model is selected through `--model {llama,qwen,mistral}` in the launcher.

## Environment

The experiments require Python 3.10 or later and the packages listed in `requirements.txt`. Large language models are loaded from Hugging Face or from local model paths supplied through the launcher arguments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some ranking scripts use Pyserini-backed indexes or BEIR/TREC data resources. Those external indexes and model weights are not included in this repository.

## Running Experiments

Pointwise ranking:

```bash
python ranking/pointwise/run_pointwise.py --model llama --dataset dl20 --experiment normal_run --chunk 1
```

Pairwise ranking:

```bash
python ranking/pairwise/run_pairwise.py --model llama --dataset dl20 --experiment normal_run --chunk 1
```

Pointwise activation patching:

```bash
python patching/pointwise/activation_patching/run_activation_patching.py --model llama --dataset dl20 --condition relevance --kind head
```

Pairwise path patching audit:

```bash
python patching/pairwise/path_patching/run_path_patching.py --audit-only
```

The launchers expose additional options for model paths, output paths, and partial runs. Use `--help` on each script for the full argument list.

## Notes

- The released scripts use the model names reported in the paper: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, and Qwen2.5-7B-Instruct.
- Query length is set to 20 tokens in the released ranking and patching launchers.
- Supplementary figures are provided under `figures/supplementary/`.
