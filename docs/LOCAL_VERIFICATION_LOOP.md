# Local Verification Loop

This repo must pass local verification before any RunPod training launch or
final Hugging Face promotion. The default budget assumption is now `$30`, so a
full CPT + SFT run is expected to warn unless a cheaper profile is selected.

## 1. Local gate

Run:

```bash
python scripts/local_verification_loop.py --strict
```

Expected result before spending GPU money:

- `Severe: 0`
- JSONL parse errors: `0`
- direct PII signals: `0`
- all Python training/eval scripts compile
- `scripts/run_eval.sh` persists `metrics.json`

Warnings are allowed only when they are explicitly accepted in the run notes.
Current expected warnings are high duplication from weighted oversampling, short
community snippets, and the full-run estimate exceeding the `$30` ceiling.

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

## 3. Smoke run before budget run

Load `recipes/smoke.env` before launching the next paid GPU job. This limits the
job to a tiny CPT/SFT run whose only purpose is to prove the whole chain:

`CPT -> merge -> SFT -> HF checkpoint -> phase6 eval -> GitHub run artifacts`.

Do not treat smoke metrics as model quality.

For the `$30` ceiling, use `recipes/budget30.env` after the smoke run. It is a
CPT-only profile and intentionally skips SFT via `SKIP_SFT=1`.

## 4. Promotion rule

Do not promote to a final Hugging Face model repo until all of these hold:

- CPT `trainer_state.json` has `global_step >= max_steps`.
- SFT repo contains `adapter_model.safetensors`.
- `runs/train-run-*` has `DONE.txt`, `manifest.json`, and component logs.
- `runs/eval-run-*` has `metrics.json`.
- The phase6 gate result is available and reviewed.
