# paper8b RunPod Readiness - 2026-05-12

- Verdict: `PASS`
- Profile: `paper8b` full pipeline, no feature-loss budget mode
- Local verifier: `PASS` / severe `0` / warnings `0`
- Mac mini smoke: `PASS`
- Train/eval dry-run: preflight `PASS`, eval smoke `PASS`, launch rc `0`

## Recipe

- GPU: `NVIDIA L40S`
- Base model: `Qwen/Qwen3-8B-Base`
- CPT phase 1: `cpt_enriched.jsonl`
- CPT phase 2: `cpt_corpus.v3.jsonl` (`style_signal`)
- SFT epochs: `2`
- Budget cap: `$60`

## Cost

- CPT: `28.75h`, `$22.71`
- SFT: `6.56h`, `$5.18`
- Total train: `35.31h`, `$27.89`
- Train/eval/upload timeout cap: `61.0h`, `$48.19`

## Mac Mini Smoke Checks

- bash_n_chain_train_round2: PASS (rc=0)
- bash_n_chain_train: PASS (rc=0)
- py_compile: PASS (rc=0)
- round2_integrity: PASS (rc=0)
- prelaunch_research: PASS (rc=0)
- sft_format_smoke: PASS (rc=0)
- local_verification_paper8b: PASS (rc=0)
- train_eval_process_dry_run: PASS (rc=0)

## Artifacts

- Local verifier report: `runs/local-verification-20260512-062257-508389/report.md`
- Mac mini smoke report: `runs/macmini-smoke-20260512-062239-603856/report.md`
- Train/eval state: `.state/train-eval-process/20260512T062249884856Z`
