# Final RunPod Go/No-Go Audit

Date: 2026-04-28  
Scope: simulated launch only, no pod created

## Overall Verdict

**NO-GO**

The **training pod path is ready**: `recipes/budget30_v2.env` loads cleanly, the required train secrets are present, `scripts/launch_train_pod.py --dry-run` passes, the verifier gate is already `PASS`, and the critical Python entrypoints compile.

The **full budget30 -> phase6 loop is not ready**: local eval readiness currently fails because there is no adapter repo configured in `.env.local`, and `scripts/check_env.py` is still wired to the legacy eval contract rather than the live `phase6` contract.

## Commands Run

- `python3 scripts/check_env.py`
- `python3 -c "import py_compile; ..."`
- `set -a; source recipes/budget30_v2.env; set +a; python3 scripts/launch_train_pod.py --dry-run`
- `set -a; source recipes/budget30_v2.env; set +a; python3 scripts/launch_eval_pod.py --dry-run`
- `wc -l cpt_corpus.v3.jsonl cpt_context_stream.jsonl sft_pairs.v2.jsonl val_set.v2.jsonl`
- budget arithmetic check with `rows=48054`, `effective_batch=16`, `sec_per_step=18.43`, `usd_per_hour=0.79`

## Criteria

| # | Criterion | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Source `recipes/budget30_v2.env` and verify required env vars are set | **PASS (train path)** | The recipe defines the expected budget30 train knobs: `BASE_MODEL`, `GPU_TYPE=L40S`, `CONTAINER_IMAGE`, `BUDGET_PROFILE=budget30`, `CPT_LR=1e-4`, `CPT_NUM_EPOCHS=1`, `SFT_NUM_EPOCHS=0`, `SKIP_SFT=1`, timeout values, `EVAL_MODE=phase6`, and `RUN_MAUVE=0` in [recipes/budget30_v2.env](/Users/unoa/projects/dalbitalba-train-data/recipes/budget30_v2.env:3). After sourcing the recipe and loading `.env.local`, the train-launch secrets were all present: `RUNPOD_API_KEY`, `HF_TOKEN`, `HF_USERNAME`, `GITHUB_TOKEN`. |
| 2 | Run `python3 scripts/check_env.py` and report result | **FAIL** | The command exits with code `1`. Train keys are all `OK`, but eval readiness fails with `HF_ADAPTER_REPO MISSING` and `ANTHROPIC_API_KEY MISSING`. The current checker hardcodes those eval requirements in [scripts/check_env.py](/Users/unoa/projects/dalbitalba-train-data/scripts/check_env.py:17). |
| 3 | Compile core Python entrypoints | **PASS** | `train_cpt.py`, `train_sft.py`, `scripts/launch_train_pod.py`, `scripts/phase6_eval.py`, `scripts/phase6_generate.py`, and `scripts/poll_pod.py` all passed `py_compile` with exit code `0`. |
| 4 | Verify `chain_train.sh` uses v3 filenames | **PASS** | `chain_train.sh` prefers `cpt_corpus.v3.jsonl` when present and still uses `sft_pairs.v2.jsonl` and `val_set.v2.jsonl` as defaults, which matches the current artifact naming in [chain_train.sh](/Users/unoa/projects/dalbitalba-train-data/chain_train.sh:41). |
| 5 | Verify budget: 1 epoch of 48054 rows at batch=16, 18.43 sec/step, $0.79/hr | **FAIL** | The dataset sizes are real: `cpt_corpus.v3.jsonl=48054`, `cpt_context_stream.jsonl=48054`, `sft_pairs.v2.jsonl=44474`, `val_set.v2.jsonl=2451`. `train_cpt.py` uses `BATCH_SIZE=1`, `GRAD_ACCUM=16`, so effective batch is `16` in [train_cpt.py](/Users/unoa/projects/dalbitalba-train-data/train_cpt.py:68), and steps are `ceil(len(train_ds)/16)` in [train_cpt.py](/Users/unoa/projects/dalbitalba-train-data/train_cpt.py:296). The arithmetic gives `3004` steps, `15.3788` hours, and about **`$12.15`**, not `$12.17`. The estimate is close, but the exact figure in the task summary is not reproducible from the current inputs. |
| 6 | Verify eval gate thresholds in `phase6_eval.py` | **PASS** | The gate is exactly `bigram_jsd <= 0.08` and `domain_keyword_alignment >= 0.50`, with the rest of the phase6 thresholds defined in [scripts/phase6_eval.py](/Users/unoa/projects/dalbitalba-train-data/scripts/phase6_eval.py:318). |
| 7 | Check if `.env.local` has all required keys | **FAIL** | `.env.local` currently contains only `RUNPOD_API_KEY`, `HF_TOKEN`, `HF_USERNAME`, `GITHUB_TOKEN`, `GITHUB_REPO`, and `NTFY_TOPIC`. It does **not** contain `HF_ADAPTER_REPO` or `SFT_ADAPTER_REPO`. That is not just a checker issue: the actual eval launcher in `phase6` mode requires `SFT_ADAPTER_REPO` (falling back to `HF_ADAPTER_REPO`) in [scripts/launch_eval_pod.py](/Users/unoa/projects/dalbitalba-train-data/scripts/launch_eval_pod.py:169), and `scripts/phase6_generate.py` hard-fails without it in [scripts/phase6_generate.py](/Users/unoa/projects/dalbitalba-train-data/scripts/phase6_generate.py:12). Running `python3 scripts/launch_eval_pod.py --dry-run` after sourcing the recipe fails immediately with `[ERROR] missing env: SFT_ADAPTER_REPO`. |
| 8 | Simulate the full launch sequence without launching | **FAIL (end-to-end)** | The **train** side is green: `python3 scripts/launch_train_pod.py --dry-run` passes, shows `[gate] verifier PASS`, and serializes a payload targeting `NVIDIA L40S`; the verifier file is `PASS` with `severe_count=0`. The verifier default inputs are already `v3`-aware in [scripts/local_verification_loop.py](/Users/unoa/projects/dalbitalba-train-data/scripts/local_verification_loop.py:30). The **eval** side is red: `python3 scripts/launch_eval_pod.py --dry-run` exits with `[ERROR] missing env: SFT_ADAPTER_REPO`. |

## Brutally Honest Readout

- The repo is **ready to launch the budget30 CPT train pod**.
- The repo is **not ready to execute the full post-train phase6 loop from the current local env**.
- `scripts/check_env.py` is also **out of sync with the live phase6 contract**. In `phase6`, `launch_eval_pod.py` does **not** require `ANTHROPIC_API_KEY`; it requires `SFT_ADAPTER_REPO` or `HF_ADAPTER_REPO` instead. Right now the script reports one real blocker (`HF_ADAPTER_REPO` missing) and one stale blocker (`ANTHROPIC_API_KEY`).

## Minimum Fixes Before Go

1. Add `SFT_ADAPTER_REPO` to `.env.local` once the adapter repo name is known, or add `HF_ADAPTER_REPO` and keep the fallback behavior.
2. Update `scripts/check_env.py` so `--target eval` matches `EVAL_MODE=phase6` instead of always enforcing the legacy Anthropic path.
3. Correct any documentation or dashboards that still present the budget30 CPT estimate as `$12.17`; the current math yields about `$12.15`.
