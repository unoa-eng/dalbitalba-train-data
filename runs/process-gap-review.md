# Process Gap Review

Date: `2026-04-29`  
Repo: `/Users/unoa/projects/dalbitalba-train-data`  
Branch: `budget30-pre-launch`  
Docs reviewed end-to-end: `TRAINING_DESIGN_V3.md`, `AUTOPILOT_LOOP.md`

## Overall Verdict

The repo can mechanically launch the current `budget30_v2` CPT-only train pod on `cpt_context_stream.jsonl`.

The repo is **not** ready to call the full v3 structured design complete. The structured `v3-data/` path is built, but it is not promoted, its verifier is still failing, its SFT output is not wired into the active trainers, blind eval is not the default promotion gate, and nearly all v3 artifacts are still uncommitted.

Status legend:

- `DONE`: implemented and evidenced in the current repo.
- `IN PROGRESS`: partially implemented or evidenced, but not yet a clean acceptance state.
- `MISSING`: still only in design docs or not wired into the active path.

## Live Evidence Snapshot

- Raw source exists at `/Users/unoa/Downloads/crawled-data-v2`.
- Active tracked launch corpus exists: `cpt_context_stream.jsonl` = `42,137` lines.
- Candidate structured CPT exists: `v3-data/cpt_structured_v3.jsonl` = `53,675` lines.
- Candidate 5-task SFT exists: `v3-data/sft_5task_v3.jsonl` = `58,476` lines.
- `python3 scripts/check_env.py --target train` = `PASS`.
- `python3 scripts/check_env.py --target both` = `FAIL` on missing `SFT_ADAPTER_REPO|HF_ADAPTER_REPO`.
- `bash -n chain_train.sh` = `PASS`.
- `python3 -m py_compile ...` for core train/eval/build scripts = `PASS`.
- `source recipes/budget30_v2.env && python3 scripts/launch_train_pod.py --dry-run` = `PASS` with `[gate] verifier PASS`.
- `source recipes/budget30_v2.env && python3 scripts/launch_eval_pod.py --dry-run` = `FAIL` with missing `SFT_ADAPTER_REPO`.

## End-to-End Step Ledger

| # | Category | Step | Status | Evidence / current gap |
| --- | --- | --- | --- | --- |
| 1 | Data pipeline | Raw crawl is present and treated as source of truth | `DONE` | `AUTOPILOT_LOOP.md` defines raw as truth source; `/Users/unoa/Downloads/crawled-data-v2` exists with the crawl batches. |
| 2 | Data pipeline | Raw profiling pass exists | `DONE` | `scripts/profile_raw_crawl.py` exists; `runs/raw-profile-20260427.json` is present. |
| 3 | Data pipeline | Cleaning/filtering code exists for v3 structured build | `DONE` | `scripts/build_structured_cpt_v3.py` and `scripts/build_5task_sft_v3.py` exist and were used to produce `v3-data/` outputs. |
| 4 | Data pipeline | Structured CPT v3 build (`raw -> clean -> structured`) | `DONE` | `v3-data/cpt_structured_individual.jsonl`, `v3-data/cpt_structured_threads.jsonl`, `v3-data/cpt_structured_v3.jsonl` exist. |
| 5 | Data pipeline | 5-task SFT v3 build | `DONE` | `v3-data/sft_5task_v3.jsonl` exists with summary counts in `v3-data/sft_v3_summary.json`. |
| 6 | Data pipeline | Tokenizer special-token extension script from the v3 design | `MISSING` | The design explicitly requires tokenizer extension for `<|post|>` / `<|comment ...|>` tokens, but repo search finds only design references plus builder-side text insertion, not a tokenizer-update script. |
| 7 | Data pipeline | Synthetic augmentation / EntiGraph stage from the v3 design | `MISSING` | Mentioned in `TRAINING_DESIGN_V3.md`; no augmentation artifact or script wiring found in the active path. |
| 8 | Data pipeline | Replay / anti-forgetting Korean mix from the v3 design | `MISSING` | The design calls for replay data, but no such dataset or recipe wiring is present in the current repo path. |
| 9 | Local validation | Reply/context structure audit exists | `DONE` | `runs/reply-audit.json` reports `reply_chain` sample connection `20/20` and `context_comment_parent_ref` `PASS`. |
| 10 | Local validation | Term-gap audit exists | `DONE` | `scripts/refinement_loop.py` exists; `runs/refinement-v3-report.md` and `runs/final-eval-readiness.md` show repeated local gap diagnostics. |
| 11 | Local validation | Term-gap closure is clean enough for final acceptance | `IN PROGRESS` | Old `v3` patch reports improved gaps, but the active context-stream run still reports `17` moderate gaps in `runs/final-eval-readiness.md` / `runs/reaudit-goal-alignment.md`. |
| 12 | Local validation | Promo contamination audit exists | `DONE` | `runs/promo-final-audit.json` and `runs/reaudit-goal-alignment.md` explicitly measure promo/operator residue. |
| 13 | Local validation | Promo/operator contamination is actually reduced to an acceptable level | `IN PROGRESS` | Active `cpt_context_stream.jsonl` still has material promo residue in prior audits; candidate `v3` structured corpus still fails verifier on contact/minor heuristics. |
| 14 | Local validation | Verifier pass for candidate structured v3 corpus | `IN PROGRESS` | `runs/v3-structured-build-report.md` says `v3-data/cpt_structured_v3.jsonl` still fails with `phone_like=4` and `minor_sexual_proximity=1`; `runs/latest-local-verification.json` is still `FAIL`. |
| 15 | Local validation | Artifact/report count consistency | `IN PROGRESS` | Current files and older reports disagree: `cpt_context_stream.jsonl` is now `42,137` lines, while older reports cite `42,473`, `43,635`, and `48,054`; `v3-data/cpt_structured_v3.jsonl` is `53,675` lines while `v3-data/cpt_v3_summary.json` / `runs/v3-structured-build-report.md` still say `53,680`. |
| 16 | Local LLM experiment | Baseline vs fine-tuned local sanity work exists at all | `DONE` | `TRAINING_DESIGN_V3.md` includes a Qwen3-4B baseline failure readout and a Qwen3-0.6B MLX LoRA result on earlier data. |
| 17 | Local LLM experiment | Local MLX test specifically on the new v3 structured data | `MISSING` | The assigned `runs/mlx-v3-test/` output does not exist. No v3-specific local adapter samples or log directory were found. |
| 18 | Local LLM experiment | Baseline vs fine-tuned comparison for the actual v3 candidate | `MISSING` | There is no current repo artifact comparing baseline vs fine-tuned generations for `v3-data/cpt_structured_v3.jsonl`. |
| 19 | Eval pipeline | Deterministic `phase6` scripts exist and compile | `DONE` | `scripts/phase6_generate.py`, `scripts/phase6_eval.py`, and related scripts compile successfully. |
| 20 | Eval pipeline | Kind-stratified deterministic eval is implemented | `DONE` | Current reports and code search show `phase6_generate.py` emits `kind`, and `phase6_eval.py` consumes kind metadata. |
| 21 | Eval pipeline | Domain metric evaluator for v3 exists and was run | `DONE` | `scripts/eval_domain_metrics_v3.py` exists; `runs/v3-structured-eval.json` and `runs/v3-structured-individual-eval.json` exist. |
| 22 | Eval pipeline | Candidate structured v3 passes domain/distribution similarity gate | `IN PROGRESS` | `runs/v3-structured-eval.json` says `bigram_jsd=0.249792`, `length_kl=0.042703`, `overall_verdict=NEEDS_IMPROVEMENT`. |
| 23 | Eval pipeline | Blind eval toolchain exists | `DONE` | `eval/make_eval_samples.py`, `eval/judge_3way.py`, and `eval/native_eval_kit.py` exist. |
| 24 | Eval pipeline | Blind eval is the default promotion gate | `MISSING` | The active recipe uses `EVAL_MODE=phase6`; blind eval remains a side path, not the default paid acceptance gate. |
| 25 | Eval pipeline | Generated reply/thread coherence is a first-class promotion metric | `MISSING` | There is structure auditing on data, but no default generated-output gate for thread coherence or reply-structure realism in the active paid loop. |
| 26 | Recipe finalization | Active CPT file is decided for the current train pod path | `DONE` | `recipes/budget30_v2.env` points `TRAIN_CPT_JSONL` to `cpt_context_stream.jsonl`. |
| 27 | Recipe finalization | Final decision that structured `v3-data/cpt_structured_v3.jsonl` is the launch CPT | `MISSING` | The build report explicitly says `recipes/budget30_v2.env` was **not** updated to the structured v3 file. |
| 28 | Recipe finalization | Current launch hyperparameters are defined for the active recipe | `DONE` | `recipes/budget30_v2.env` defines `CPT_LR=1e-4`, `CPT_NUM_EPOCHS=1`, `SFT_NUM_EPOCHS=0`, `SKIP_SFT=1`, timeout knobs, and `EVAL_MODE=phase6`. |
| 29 | Recipe finalization | Active SFT recipe uses the new 5-task v3 SFT corpus | `MISSING` | Search shows `sft_5task_v3.jsonl` is only produced and reported; the active trainers still default to legacy `sft_pairs.v2.jsonl`. |
| 30 | Recipe finalization | v3-specific train/eval entrypoints from the design (`train_cpt_v3.py`, `train_sft_v3.py`, `eval_v3.py`) | `MISSING` | Repo search finds only design-doc mentions of these files, not actual implementations. |
| 31 | Pre-launch | Train-only env check passes | `DONE` | `python3 scripts/check_env.py --target train` passes with required train keys present. |
| 32 | Pre-launch | Full train+eval env check passes | `MISSING` | `python3 scripts/check_env.py --target both` fails on missing `SFT_ADAPTER_REPO|HF_ADAPTER_REPO`. |
| 33 | Pre-launch | Training wrapper syntax and core Python entrypoints compile | `DONE` | `bash -n chain_train.sh` and `python3 -m py_compile ...` both pass. |
| 34 | Pre-launch | Train pod dry-run passes | `DONE` | `scripts/launch_train_pod.py --dry-run` passes under `recipes/budget30_v2.env` and reports verifier `PASS`. |
| 35 | Pre-launch | Eval pod dry-run passes | `MISSING` | `scripts/launch_eval_pod.py --dry-run` currently fails on missing adapter repo env. |
| 36 | Pre-launch | Live hourly price drift is hard-blocked by launcher logic | `MISSING` | Existing audits note no hard runtime price ceiling enforcement in the launcher. |
| 37 | Git state | V3 docs/scripts/data/reports are committed to git | `MISSING` | `git ls-files` only returns `AUTOPILOT_LOOP.md` from the v3 set queried. `TRAINING_DESIGN_V3.md`, `recipes/budget30_v2.env`, `v3-data/`, v3 build scripts, and v3 reports are not committed. |
| 38 | Git state | Dirty tracked-file risk is understood before launch | `DONE` | `git status --short --branch` shows a heavily dirty branch; the risk is identifiable and should be treated as a launch blocker until the branch is cleaned or intentionally snapshotted. |

## Requested Check Summary

| Requested area | Status | What is actually true right now |
| --- | --- | --- |
| (a) Data pipeline: raw -> clean -> structured -> verified | `IN PROGRESS` | Raw and structured build are real. Verification is real. The structured v3 candidate still fails its own verifier and is not the promoted training corpus. |
| (b) Local validation: term gaps, structure, promo contamination | `IN PROGRESS` | All three are being measured locally. Structure checks are the healthiest. Gap closure and promo cleanup are still not clean enough to declare finished. |
| (c) Local LLM experiment: baseline vs fine-tuned comparison | `IN PROGRESS` | Legacy/local experimentation exists, but the specific v3 MLX test requested for the current cycle has not been done. |
| (d) Eval pipeline: phase6 metrics, domain metrics, blind eval | `IN PROGRESS` | `phase6` and domain metrics exist. Blind eval exists as scripts but is not the default promotion gate and is not wired into the current paid loop. |
| (e) Recipe finalization: CPT file, hyperparams, eval gates | `IN PROGRESS` | A current CPT-only budget30 recipe exists, but it launches `cpt_context_stream.jsonl`, not the structured v3 corpus, skips SFT, and does not make blind eval mandatory. |
| (f) Pre-launch checklist: env vars, budget, script compilation, file naming | `IN PROGRESS` | Train launch is mechanically ready. Full eval readiness is not. Budget math and filenames are still muddied by stale reports and mixed corpus generations. |
| (g) Git state: v3 artifacts committed, dirty-worktree risk | `MISSING` | The repo is not in a clean, durable, reviewable state for v3. Most v3 artifacts are still uncommitted and could be lost by reset/clean/rebase mistakes. |

## Current Git Risk

Before adding this review file, the branch was already dirty.

Key tracked files already modified:

- `chain_train.sh`
- `train_cpt.py`
- `scripts/check_env.py`
- `scripts/local_verification_loop.py`
- `scripts/phase6_eval.py`
- `scripts/launch_train_pod.py`
- `scripts/build_thread_aware_datasets.py`
- `scripts/cycle_report.py`
- `scripts/recipe_mutator.py`
- `HANDOFF_CODEX.md`
- `docs/MACMINI_BOOTSTRAP.md`
- `docs/RECIPE_MUTATION_RULEBOOK.md`

Key v3 artifacts currently untracked:

- `TRAINING_DESIGN_V3.md`
- `recipes/budget30_v2.env`
- `recipes/round2-cycle1.env`
- `v3-data/`
- `scripts/build_structured_cpt_v3.py`
- `scripts/build_5task_sft_v3.py`
- `scripts/eval_domain_metrics_v3.py`
- `scripts/build_context_cpt.py`
- `scripts/phase6_eval_v2.py`
- most `runs/` v3 readiness / audit reports

Practical implication:

- A careless `git clean`, branch reset, or rebase could wipe out most of the v3 work.
- There is not yet a durable git snapshot that cleanly answers “what exactly is the current v3 launch candidate?”

## Bottom Line

If the question is **“can the repo launch the current budget30 CPT-only train pod?”**, the answer is **yes**.

If the question is **“is the full v3 structured process complete from raw data through a launch-ready, reviewable, reproducible training recipe?”**, the answer is **no**.

The highest-signal blockers are:

1. `v3-data/cpt_structured_v3.jsonl` still fails local verification and is not the promoted CPT file.
2. `sft_5task_v3.jsonl` exists but is not used by the active SFT trainer or recipe.
3. The v3-design-only pieces are still missing: tokenizer special-token setup, v3 train/eval entrypoints, and blind-eval-default gating.
4. Eval launch still fails locally because `SFT_ADAPTER_REPO|HF_ADAPTER_REPO` is unset.
5. The repo’s v3 work is not committed, and report/file counts are inconsistent across artifacts.
