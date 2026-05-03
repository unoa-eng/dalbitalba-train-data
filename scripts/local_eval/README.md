# Local Eval & Audit Tools (mac-mini)

Companion scripts for Round-2 prep + offline analysis.

## Environment

```bash
cd ~/projects/dalbitalba-train-data
source .venv-local/bin/activate    # or recreate via: uv venv .venv-local --python 3.11
```

Verified specs (M4 16GB, MPS bf16):
- Qwen3-0.6B-Base inference: ~10.9 tok/s (80 tok in 7.3s)
- Qwen3-8B-Base inference: NOT FEASIBLE (>16GB unified memory)

## Tools

- `local_inference_smoke.py` — 0.6B base smoke test (~30s end-to-end)
- `audit_ad_patterns.py` — broad ad/leakage pattern audit across CPT/SFT corpora
- `audit_ad_targeted.py` — precise hard-ad markers (학원광고/공식어/대량판매)
- `verify_metrics.py` — L1+L3 sanity check on metrics.json + L2 5-sample dump
- `print_metrics.py` — pretty-print metrics.json + per-term domain alignment

## Round-2 0.6B local eval (post-RunPod-train)

After RunPod produces Round-2 0.6B adapter on HF Hub:

```bash
EVAL_MAX_ROWS=256 \
SFT_ADAPTER_REPO=UNOA/<round2-adapter-repo> \
BASE_MODEL=Qwen/Qwen3-0.6B-Base \
INPUT_PATH=val_set.v3.jsonl \
OUTPUT_PATH=ai_generated.jsonl \
python scripts/phase6_generate.py
python scripts/phase6_eval.py --ai ai_generated.jsonl --raw val_set.v3.jsonl --out metrics.json --skip-mauve
```

Cost: $0. Wall: ~80 min for N=256.
