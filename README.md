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

## Bring-up on another environment

1. Clone this repository.
2. Copy `.env.local.example` to `.env.local`.
3. Fill the shared keys you may already have in `dalbitalba/apps/web/.env.local`:
   `RUNPOD_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
4. Add the train-repo-only keys:
   `HF_USERNAME`, `GITHUB_TOKEN`, and for local eval launches `HF_ADAPTER_REPO`.
5. Optionally set `NTFY_TOPIC`, `BASE_MODEL`, `GPU_TYPE`, and `CONTAINER_IMAGE`.
6. Run `python scripts/check_env.py --target train` or `python scripts/check_env.py --target eval`.
7. Launch with `python scripts/launch_train_pod.py` or `python scripts/launch_eval_pod.py`.

This means another machine can reconstruct the pipeline quickly, but resuming an old run
from the exact previous checkpoint still depends on whether the adapter or checkpoint was
already uploaded to Hugging Face or another persistent storage target.

## Service repo bridge

`unoa-eng/dalbitalba` can manually dispatch these workflows through its `Train Data Bridge`
workflow. The bridge keeps GPU and evaluation execution here while letting service-side
operators start a train or eval run from the app repository.

## Required GitHub configuration

Repository secrets used here:

- `RUNPOD_API_KEY`
- `HF_TOKEN`
- `HF_USERNAME` for training uploads
- `TRAIN_REPO_PUSH_TOKEN` for Git clone/push from inside RunPod
- `ANTHROPIC_API_KEY` for eval judging
- `OPENAI_API_KEY` for eval judging
- `NTFY_TOPIC` optional notification topic

Repository variables used here:

- `BASE_MODEL`
- `GPU_TYPE`
- `CONTAINER_IMAGE`

Additional service-repo secret:

- `dalbitalba` needs `TRAIN_REPO_DISPATCH_TOKEN` so its bridge workflow can dispatch
  these train-repo workflows.

If you are operating only through GitHub Actions, these values should be placed in
repository secrets and variables instead of a local `.env.local` file.

## Research note

`research/obsidian-export/` is preserved for manual analysis and board/community content review.
It is not production input and should not be fed into training without an explicit curation step.
