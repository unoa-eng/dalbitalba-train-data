# Local Verification Loop

This repo must pass local verification before any full RunPod training launch or
final Hugging Face promotion.

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
community snippets, and full-run cost proximity to the budget.

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

## 4. Promotion rule

Do not promote to a final Hugging Face model repo until all of these hold:

- CPT `trainer_state.json` has `global_step >= max_steps`.
- SFT repo contains `adapter_model.safetensors`.
- `runs/train-run-*` has `DONE.txt`, `manifest.json`, and component logs.
- `runs/eval-run-*` has `metrics.json`.
- The phase6 gate result is available and reviewed.
