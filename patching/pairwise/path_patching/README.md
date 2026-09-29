# Pairwise Q-only path patching (Llama / DL20)

Place this `patching/pairwise/path_patching/` directory inside `role-ranking-artifact/`.
It follows the entry-point and candidate-file layout of
`patching/pointwise/path_patching/`. The per-example three-stage implementation from
the completed Pairwise experiment has been merged into `run_path_patching.py`.

## Files in this directory

- `run_path_patching.py`: runs the fixed 68 causal paths across 10 role pairs
  and 54 DL20 queries, including the original C0–C7 interventions, resume
  handling, and summary generation.
- `code/methods.py`: the terminal-token position logic used in the completed
  run. Every selected Llama adjective and adverb is one token, so for these
  roles a terminal position is also its full span.
- `candidates/llama/dl20.csv`: the exact 68-path candidate file from that run
  (SHA-256 `c950d30451da91ef92a497e785c9c5d53c171ef73ade340809b6756de5148cea`).

The script reuses `patching/pairwise/activation_patching/code/` (model, loaders,
`pairwise_core.py`, and data structures), `ranking/pairwise/common/` (formal
Pairwise prompt), the pairwise selected20 roles, and
`data/pairs/pointwise/dl20.tsv`. It does not use the pointwise prompt or the
fullspan instruction variant.

## Check files and run

From the repository root, with the same Python environment used for your
pairwise activation-patching experiment:

```bash
python patching/pairwise/path_patching/run_path_patching.py --audit-only

python patching/pairwise/path_patching/run_path_patching.py \
  --model-path /path/to/local/Llama-3.1-8B-Instruct
```

`--audit-only` needs only Python's standard library; the full run uses the
dependencies from the pairwise activation-patching environment (including
`pandas`, `torch`, and `transformer_lens`).

Default output is `results/pairwise_path_patching/llama/dl20/`. If desired,
pass `--output-dir /path/to/experiment-output`. The optional `--pairs-path`,
`--role-json-path`, and `--candidates-path` flags override the defaults;
the candidate SHA-256 remains fixed to the completed Q-only experiment.

The runner verifies the Pairwise question-only `inst_all` field and the
selected single-token roles before running paths. There is no standalone
preflight certificate; input and token-range checks occur in the runner.
Add `/results/` to the repository-root `.gitignore` before tracking this code
if outputs will be written under the default directory.
