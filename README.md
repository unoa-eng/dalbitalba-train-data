# dalbitalba-train-data

Training, evaluation, RunPod, and research repository for the dalbitalba writing stack.

## Boundary

- `dalbitalba`: service repo for web, backend, Flutter, deployment, and production database code
- `dalbitalba-train-data`: datasets, RunPod launchers, blind-eval tooling, benchmark artifacts, and research archives

## Active layout

- root `*.jsonl`: curated corpora and seed datasets
- `train_*.py`, `chain_train.sh`: RunPod training entrypoints
- `scripts/launch_train_pod.py`: start a training pod
- `scripts/launch_eval_pod.py`: start an evaluation pod
- `scripts/generate_samples.py`: generate AI blind-test samples from the uploaded adapter repo
- `scripts/run_eval.sh`: end-to-end blind evaluation runner inside RunPod
- `eval/`: blind-set builder, three-judge runner, and report renderer
- `runs/`: Git-persisted outputs written back by RunPod jobs

## Imported artifacts

- `bench/`: local benchmark snapshots copied from the service repo
- `turing-test/`: manual blind quiz artifacts
- `research/obsidian-export/`: Obsidian-ready board and community research vault
- `archive/omni-audit/`: historical OMNI audit outputs
- `archive/runpod/`: local orchestration logs
- `archive/legacy-omni-audit/`: old scheduler and git-hook assets kept for reference

## RunPod flow

1. Launch training with `python scripts/launch_train_pod.py`.
2. Training uploads adapters to Hugging Face and pushes `runs/latest-train.json` plus `runs/train-run-*` branches back to GitHub.
3. Launch evaluation with `python scripts/launch_eval_pod.py`.
4. Evaluation generates samples, runs judges, renders reports, and pushes `runs/latest-eval.json` plus `runs/eval-run-*` branches back to GitHub.

## Research note

`research/obsidian-export/` is preserved for manual analysis and board/community content review.
It is not production input and should not be fed into training without an explicit curation step.
