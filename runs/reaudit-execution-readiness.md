# Re-audit Execution Readiness

Date: 2026-04-28  
Scope: worker-3 re-audit after fixes, repo state only

## Overall Verdict

**GO for the `budget30_v2` train pod launch.**

The train launch path is mechanically ready: the recipe now targets `cpt_context_stream.jsonl`, `chain_train.sh` prefers that corpus, all Python entrypoints compile, the exact CPT-only budget is well under `$30`, `phase6_eval.py` and `phase6_generate.py` have the expected post-train contracts, `scripts/check_env.py` no longer hard-requires `ANTHROPIC_API_KEY`, `python3 scripts/check_env.py --target train` passes, `python3 scripts/launch_train_pod.py --dry-run` passes, and `python3 scripts/local_verification_loop.py --strict --profile budget30` returns `PASS`.

This is **not** a full train->phase6 GO. Local eval env is still incomplete because `SFT_ADAPTER_REPO|HF_ADAPTER_REPO` is unset, and the repo’s local verifier still audits `cpt_corpus.v3.jsonl` by default rather than the newly promoted `cpt_context_stream.jsonl`.

## Commands Run

- `sed -n '1,260p' AUTOPILOT_LOOP.md`
- `set -a; source recipes/budget30_v2.env; set +a; python3 scripts/check_env.py --target both`
- `set -a; source recipes/budget30_v2.env; set +a; python3 scripts/check_env.py --target train`
- `set -a; source recipes/budget30_v2.env; set +a; python3 scripts/launch_train_pod.py --dry-run`
- `set -a; source recipes/budget30_v2.env; set +a; python3 scripts/local_verification_loop.py --strict --profile budget30`
- `python3 -m py_compile train_cpt.py train_sft.py $(rg --files scripts -g '*.py' | sort)`
- `wc -l cpt_context_stream.jsonl cpt_corpus.v3.jsonl val_set.v2.jsonl`

## Criteria

| # | Criterion | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | `recipes/budget30_v2.env` points `TRAIN_CPT_JSONL` at `cpt_context_stream.jsonl` | **PASS** | The recipe explicitly sets `TRAIN_CPT_JSONL=cpt_context_stream.jsonl` in [recipes/budget30_v2.env](/Users/unoa/projects/dalbitalba-train-data/recipes/budget30_v2.env:7). After sourcing the recipe, `python3 scripts/check_env.py --target train` exits `0`, and `python3 scripts/launch_train_pod.py --dry-run` serializes a pod payload with `TRAIN_CPT_JSONL`/`INPUT_JSONL` populated from that recipe. |
| 2 | `chain_train.sh` prefers `cpt_context_stream.jsonl` | **PASS** | `chain_train.sh` upgrades its default CPT input from `cpt_corpus.v2.jsonl` to `cpt_corpus.v3.jsonl`, then to `cpt_context_stream.jsonl` when that file exists, before syncing `TRAIN_CPT_JSONL` and `INPUT_JSONL` in [chain_train.sh](/Users/unoa/projects/dalbitalba-train-data/chain_train.sh:41). |
| 3 | All critical Python scripts compile | **PASS** | `python3 -m py_compile train_cpt.py train_sft.py $(rg --files scripts -g '*.py' | sort)` exited `0`. This is stricter than the task asked for: it compiled every Python file under `scripts/` plus both root trainers. |
| 4 | Exact CPT budget for `cpt_context_stream.jsonl`, `1` epoch, batch `16` | **PASS** | `wc -l cpt_context_stream.jsonl` reports `43,635` rows. `train_cpt.py` uses `BATCH_SIZE=1`, `GRAD_ACCUM=16`, so effective batch is `16` in [train_cpt.py](/Users/unoa/projects/dalbitalba-train-data/train_cpt.py:68), and steps are `ceil(len(train_ds)/(BATCH_SIZE*GRAD_ACCUM))` in [train_cpt.py](/Users/unoa/projects/dalbitalba-train-data/train_cpt.py:296). Exact math: `ceil(43635 / 16) = 2728` steps. Using the repo’s verifier defaults `sec_per_step=18.43` and `hourly_usd=0.79` in [scripts/local_verification_loop.py](/Users/unoa/projects/dalbitalba-train-data/scripts/local_verification_loop.py:597), this yields `13.9658444444` hours and `$11.0330171111`, i.e. about `13.97h` / `$11.03`. |
| 5 | `phase6_eval.py` gate is `bigram_jsd <= 0.08` and `domain_keyword_alignment >= 0.50` | **PASS** | The gate table defines `bigram_jsd: ("le", 0.08)` and `domain_keyword_alignment: ("ge", 0.50)` in [scripts/phase6_eval.py](/Users/unoa/projects/dalbitalba-train-data/scripts/phase6_eval.py:338). |
| 6 | `phase6_generate.py` emits `kind` | **PASS** | `phase6_generate.py` derives `kind` from row metadata / heuristics and writes JSON lines shaped as `{"text": ..., "seed": ..., "kind": kind}` in [scripts/phase6_generate.py](/Users/unoa/projects/dalbitalba-train-data/scripts/phase6_generate.py:41) and [scripts/phase6_generate.py](/Users/unoa/projects/dalbitalba-train-data/scripts/phase6_generate.py:148). |
| 7 | `scripts/check_env.py` no longer requires `ANTHROPIC_API_KEY` | **PASS** | `ANTHROPIC_API_KEY` is listed only under `OPTIONAL_KEYS["eval"]`, not `REQUIRED_KEYS`, and the script only mentions it when `EVAL_MODE=legacy` in [scripts/check_env.py](/Users/unoa/projects/dalbitalba-train-data/scripts/check_env.py:24) and [scripts/check_env.py](/Users/unoa/projects/dalbitalba-train-data/scripts/check_env.py:97). Running `python3 scripts/check_env.py --target both` fails only on `SFT/HF_ADAPTER_REPO`, not on Anthropic. |
| 8 | Final verdict for train pod launch | **GO** | `python3 scripts/launch_train_pod.py --dry-run` passes and reports `[gate] verifier PASS`. `python3 scripts/local_verification_loop.py --strict --profile budget30` also returns `{"verdict":"PASS"}` and wrote [report.md](/Users/unoa/projects/dalbitalba-train-data/runs/local-verification-20260428-081447/report.md:1). From an execution-readiness standpoint, the train pod can be launched. |

## Caveats

- `python3 scripts/check_env.py --target both` still exits `1` because local eval env is missing `SFT_ADAPTER_REPO|HF_ADAPTER_REPO`. That does **not** block the train pod, but it **does** block local phase6 eval launch after training.
- `scripts/local_verification_loop.py` still audits `cpt_corpus.v3.jsonl` by default, not `cpt_context_stream.jsonl`, via `DEFAULT_FILES["cpt"]` in [scripts/local_verification_loop.py](/Users/unoa/projects/dalbitalba-train-data/scripts/local_verification_loop.py:30). The generated PASS report therefore validates a `42,473`-row CPT corpus, while the actual `budget30_v2` launch path now targets the `43,635`-row context stream. The mechanical launch path is correct; the verifier input default is lagging behind it.

## Bottom Line

If the question is strictly **“can we launch the `budget30_v2` CPT train pod now?”**, my answer is **GO**.

If the question is **“is the entire train->phase6 local readiness story perfectly aligned?”**, the answer is **not yet** because the eval adapter repo is still unset and the local verifier default corpus has not caught up to the launch corpus.
