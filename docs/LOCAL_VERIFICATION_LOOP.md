# Local Verification Loop

This repo must pass local verification before any RunPod training launch or
final Hugging Face promotion. The default paid target is `paper8b`: full
round2 CPT phase 1/2 + SFT + integrated phase6 eval. `budget30` remains a
legacy CPT-only probe, not the final quality target.

## 1. Local gate

Run:

```bash
python scripts/local_verification_loop.py --strict --profile paper8b
```

Expected result before spending GPU money:

- `Severe: 0`
- `Warnings: 0`
- JSONL parse errors: `0`
- direct PII signals: `0`
- all Python training/eval scripts compile
- local SFT format smoke passes against `tokenizer_v4`
- `scripts/run_eval.sh` persists `metrics.json`
- Obsidian scope is present: `research/obsidian-ref`, `research/obsidian-export`,
  and `runs/round2-obsidian-synthesis/persona-30-extracted.json`

The verifier now also runs a local tokenizer-only smoke for SFT masking and
ChatML formatting. This remains a control-plane check only; it does not load
the 8B base model and does not replace the paid RunPod smoke.

The verifier also checks Obsidian-derived reference coverage mechanically. Raw
Obsidian markdown remains reference/export material; the training/eval path uses
the curated persona metadata list. Cycle-1 synthesized persona rows are fully
populated metadata entries, not launch-blocking placeholders.

Warnings are not accepted for gated paid profiles. `scripts/launch_train_pod.py`
refuses `paper8b`, `budget30`, or `smoke` launch unless the latest matching
verifier report is `PASS` with `Severe: 0` and `Warnings: 0`. Duplicate-rate
escalation, JSONL parse errors, PII signals, launch-contract failures, and
budget overruns remain blocking severe findings.

## 2. Existing HF artifact gate

Run with a read token in `HF_TOKEN`:

```bash
python scripts/local_verification_loop.py \
  --hf-cpt-repo UNOA/dalbitalba-qwen3-cpt-20260424-0618 \
  --hf-sft-repo UNOA/dalbitalba-qwen3-sft-20260424-0618 \
  --strict
```

The 20260424-0618 artifacts are expected to fail this gate because the auditable
CPT checkpoint is incomplete and the SFT repo has no adapter payload.

## 3. Smoke run before full run

Load `recipes/smoke.env` before launching the next paid GPU job. This limits the
job to a tiny CPT/SFT run whose only purpose is to prove the whole chain:

`CPT -> merge -> SFT -> HF checkpoint -> phase6 eval -> GitHub run artifacts`.

Do not treat smoke metrics as model quality.

For the no-feature-loss paid run, use `recipes/round2-cycle1.env` (`paper8b`).
It keeps both CPT phases, SFT, and phase6 eval enabled. The expected train cost
is tracked separately from the timeout hard cap; current L40S estimate is about
`$28` train cost and about `$48` timeout-cap exposure under a `$60` cap.

Use `recipes/budget30.env` only when explicitly choosing a CPT-only probe. It
skips SFT, so it is not a substitute for the full research run.

## 4. Promotion rule

Do not promote from smoke to `paper8b`, or to a final Hugging Face model repo,
until all of these hold:

- CPT `trainer_state.json` has `global_step >= max_steps`.
- Smoke SFT repo or round2 phase3 path contains `adapter_model.safetensors`.
- `runs/train-run-*` has `DONE.txt`, `manifest.json`, and component logs.
- Round2 eval artifact exists at `eval/phase5-eval-v2.json`, or classic eval has `metrics.json`.
- The phase6 gate result is available and reviewed.
