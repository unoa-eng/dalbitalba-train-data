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
- local SFT format smoke passes against `tokenizer_v4`
- `scripts/run_eval.sh` persists `metrics.json`
- Obsidian scope is present: `research/obsidian-ref`, `research/obsidian-export`,
  and `runs/round2-obsidian-synthesis/persona-30-extracted.json`

The verifier now also runs a local tokenizer-only smoke for SFT masking and
ChatML formatting. This remains a control-plane check only; it does not load
the 8B base model and does not replace the paid RunPod smoke.

The verifier also checks Obsidian-derived reference coverage mechanically. Raw
Obsidian markdown remains reference/export material; the training/eval path uses
the curated persona metadata list and treats synthesized placeholder personas as
non-blocking metadata until the cycle-2 source persona JSON is imported.

Warnings are allowed only when they are explicitly accepted in the run notes.
For the active round2/budget30 path, the expected non-blocking warning is
`cpt_corpus.v3.jsonl: many very short rows`. Duplicate-rate escalation, JSONL
parse errors, PII signals, launch-contract failures, and budget overruns remain
blocking severe findings.

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

Do not promote from smoke to budget30, or to a final Hugging Face model repo,
until all of these hold:

- CPT `trainer_state.json` has `global_step >= max_steps`.
- Smoke SFT repo or round2 phase3 path contains `adapter_model.safetensors`.
- `runs/train-run-*` has `DONE.txt`, `manifest.json`, and component logs.
- Round2 eval artifact exists at `eval/phase5-eval-v2.json`, or classic eval has `metrics.json`.
- The phase6 gate result is available and reviewed.
