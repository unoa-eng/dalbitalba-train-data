# HANDOFF — Codex Autonomous Loop

**Active as of**: 2026-04-24 06:40 UTC
**Owner Claude is handing off**: Phase 5 (pod7 nnoci5ebwqluog) → Phase 6 eval → conditional Phase 7 ORPO → final PR

You are Codex CLI taking over an active Korean-LLM fine-tuning run. Claude has completed Phases 0–2 (data pipeline, MLOps overhaul) and debugged 7 pod launches to reach a working recipe. This document is your single source of truth. Read it end-to-end BEFORE taking any action.

---

## 1. Current live state

| Item | Value |
|------|-------|
| Active pod | `nnoci5ebwqluog` (L40S, $0.79/hr) |
| Pod status | RUNNING (CPT phase) as of 06:40 UTC, created 06:17:59 UTC |
| Expected pod completion | ~09:00 UTC (CPT 48m + merge 7m + SFT 80m + upload 10m) |
| HF CPT repo | `UNOA/dalbitalba-qwen3-cpt-20260424-0617` (filled at save_steps=50) |
| HF SFT repo | `UNOA/dalbitalba-qwen3-sft-20260424-0617` (filled after SFT) |
| GitHub branch | `codex/train-runpod-fixes` @ `364bf16` |
| Budget remaining | $35.01 (cumulative pod burn so far ≈ $0.20) |
| ntfy topic | `dalbit_ai_alert` (push on stage transitions) |
| Raw DB ceiling | `bigram_jsd` baseline 0.019 (from `.planning/calibration/raw-vs-raw.json`) |

## 2. Environment & auth

All credentials live in `.env.local` (gitignored). Load with:

```bash
set -a; source <(grep -v '^#' .env.local | grep '='); set +a
```

Required: `RUNPOD_API_KEY`, `HF_TOKEN`, `HF_USERNAME=UNOA`, `GITHUB_TOKEN`, `GITHUB_REPO=unoa-eng/dalbitalba-train-data`, `NTFY_TOPIC=dalbit_ai_alert`, `WANDB_API_KEY`.

**Security**: never echo, log, or commit these values. Only use shell expansion.

## 3. Locked recipe (do NOT mutate before PASS)

Baseline Phase 5 recipe:
- Base: `Qwen/Qwen3-8B-Base`
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- CPT: r=64 rsLoRA α=64, lr=2e-4, seq_len=1024, 1 epoch, paged_adamw_32bit, bf16
- Merge: fp16 via `scripts/merge_cpt_to_fp16.py`
- SFT: same LoRA config, lr=5e-5, 2 epochs, mix 80% raw continuation + 20% reply pair
- Deps pinned in `chain_train.sh` — do NOT reorder

## 4. Decision tree

Execute this in order. Commit changes with messages beginning `loop(cycle-N):`.

```
WAIT_FOR_POD7_DONE:
  # Poll every 5 min via `scripts/poll_pod.py <pod_id>`. On EXITED:
  - if status=done_ok      → go PHASE6
  - if status=done_cpt_only → go PHASE6 (eval CPT-only variant, SFT recipe stays)
  - if status=done_no_upload→ retry upload once, then PHASE6
  - else                   → diagnose via runs/train-run-<stamp>/ branch logs,
                             fix root cause, relaunch, max 2 retry. If 2 retries
                             fail, escalate via ntfy "ESCALATE: pod7 recovery stuck"
                             and stop loop.

PHASE6 (eval):
  1. Launch eval pod (template below). Budget: ~$0.8 / 40 min.
  2. eval pod runs: sample generation (500 samples, min-p=0.05, T=1.1, max_new=200)
     then phase6_eval.py (pre-built by Claude, outputs JSON gate report).
  3. Commit eval report to runs/eval-run-<stamp>/ branch.
  4. If gate.verdict=="PASS" → go CREATE_PR.
  5. If gate.verdict=="FAIL" → go PHASE7 (at most cycle_cap=3 iterations).

PHASE7 (ORPO, conditional):
  Only runs if PHASE6 failed AND cycle_counter < 3 AND budget_remaining > $4.
  Mutation rules (apply exactly one per cycle, pick highest-impact):
    - violations include "bigram_jsd" > 0.15     → CPT 2 epochs (was 1)
    - violations include "length_kl" > 0.10      → oversample xl/xxl 3× (was 2×)
    - violations include "english_density" > ...  → oversample english 3×
    - violations include "mauve" < 0.80          → enable DoRA (CPT_USE_DORA=1)
    - all metrics close to threshold              → bump LORA_R to 128
  Relaunch pod with new recipe env vars. Retry PHASE6.

CREATE_PR:
  Run `scripts/create_final_pr.sh` (to be authored by Codex on cycle 1,
  template in section 7). Body must cite the final eval report JSON from
  the winning cycle. Stop loop.
```

## 5. Budget & safety guards (enforce in every iteration)

Before launching ANY pod, check:

```bash
SPENT=$(cat .state/budget_spent.json | jq -r '.usd')
if (( $(echo "$SPENT > 25.0" | bc -l) )); then
  curl -d "ESCALATE: budget $SPENT of 25 cap — stop" "https://ntfy.sh/$NTFY_TOPIC"
  exit 1
fi

# Single-pod invariant
RUNNING=$(curl -sH "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://rest.runpod.io/v1/pods" \
  | jq '[.[] | select(.desiredStatus=="RUNNING")] | length')
[[ "$RUNNING" -gt 0 ]] && { echo "existing pod RUNNING, skip"; exit 0; }
```

After pod EXIT, record cost:
```bash
python3 -c "import json; s=json.load(open('.state/budget_spent.json')); s['usd']+=$POD_COST; json.dump(s,open('.state/budget_spent.json','w'))"
```

## 6. Tools already built by Claude

- `scripts/launch_train_pod.py` — launches train pod. Env vars: `BASE_MODEL`, `CPT_LR`, `SFT_LR`, `CPT_NUM_EPOCHS`, `SFT_NUM_EPOCHS`, `GPU_TYPE`, etc. See `.env.local`.
- `scripts/launch_eval_pod.py` — EXISTS BUT legacy SOLAR-oriented. Codex must extend for Qwen3:
  - Update `BASE_MODEL` default to `Qwen/Qwen3-8B-Base`
  - Update adapter path to `UNOA/dalbitalba-qwen3-sft-*`
  - Minor; ~20 lines diff.
- `chain_train.sh` — inside-pod orchestration with preflight smoke test, stage ntfy, log persistence. Re-use via relaunch.
- `scripts/phase6_eval.py` — **already built, deterministic 5-metric gate**. Takes `--ai` + `--raw`, emits JSON + exits 0 (PASS) or 2 (FAIL). Test locally:
  ```bash
  python3 scripts/phase6_eval.py --ai ai_generated.jsonl --raw val_set.v2.jsonl --out runs/eval-*/metrics.json --skip-mauve
  ```
- `scripts/merge_cpt_to_fp16.py` — PEFT merge helper.

## 7. Tools Codex must build (recipe-heavy work)

### 7.1 `scripts/poll_pod.py`

```python
#!/usr/bin/env python3
"""Poll a RunPod pod, output status JSON; exit 0 running, 1 exited, 2 error."""
import os, sys, json, urllib.request
for line in open('.env.local'):
    line=line.strip()
    if '=' in line and not line.startswith('#'):
        k,_,v=line.partition('='); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
pod_id = sys.argv[1]
req = urllib.request.Request(
    f'https://rest.runpod.io/v1/pods/{pod_id}',
    headers={'Authorization': f'Bearer {os.environ["RUNPOD_API_KEY"]}'})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read())
status = data.get('desiredStatus')
print(json.dumps({'pod_id': pod_id, 'status': status, 'last_change': data.get('lastStatusChange')}))
sys.exit(0 if status == 'RUNNING' else 1)
```

### 7.2 `scripts/phase6_generate.py` (eval-pod in-container)

Generate 500 samples from the trained adapter. Prompt source = `val_set.v2.jsonl` first 500 rows, each providing a 1-sentence seed. Sampling: T=1.1, top_p=0.9, min_p=0.05, max_new_tokens=200. Output: `/workspace/ai_generated.jsonl` with `{"text": "..."}` per line.

Skeleton:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json, os
base = os.environ['CPT_MERGED_PATH']  # fp16 merged CPT (from HF)
adapter = os.environ['SFT_ADAPTER_PATH']  # SFT LoRA
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda')
model = PeftModel.from_pretrained(model, adapter).eval()
tok = AutoTokenizer.from_pretrained(base)
# loop val_set.v2.jsonl first 500 rows, generate, write {"text": gen}
```

### 7.3 `scripts/autonomous_loop.sh`

Top-level supervisor. Guards cycle_counter, budget, single-pod. Dispatches to phase6/phase7/pr logic per decision tree.

### 7.4 `scripts/create_final_pr.sh`

```bash
#!/usr/bin/env bash
set -e
cd /mnt/c/Users/mapdr/Downloads/dalbitalba-train-data
BEST_REPORT=$(ls -t runs/eval-run-*/metrics.json | head -1)
gh pr create \
  --base main --head codex/train-runpod-fixes \
  --title "feat(training): Qwen3-8B-Base Korean fine-tune + gated eval" \
  --body-file <(cat <<EOF
## Summary
Phase 0-6 of dalbitalba training infra: raw-vs-raw baseline, v2 data pipeline,
MLOps-hardened Qwen3-8B recipe, 5-metric automated eval gate.

## Final eval report
\`\`\`json
$(cat "$BEST_REPORT")
\`\`\`

## Commits
$(git log --oneline origin/main..HEAD | head -20)

🤖 Built with Claude 4.7 Opus → handed off to Codex for loop execution
EOF
)
```

## 8. File locations

```
/mnt/c/Users/mapdr/Downloads/dalbitalba-train-data/
├── .env.local                         # credentials (gitignored)
├── .planning/calibration/
│   └── raw-vs-raw.json                 # achievability ceiling
├── .state/
│   ├── train_pod_state.json            # current pod metadata
│   └── budget_spent.json               # YOU must create/maintain, init {"usd": 0.2}
├── chain_train.sh                      # in-pod CPT→merge→SFT orchestration
├── train_cpt.py / train_sft.py         # parametric via env vars
├── scripts/
│   ├── launch_train_pod.py
│   ├── launch_eval_pod.py              # needs Qwen3 update
│   ├── merge_cpt_to_fp16.py
│   ├── phase6_eval.py                  # READY — 5-metric gate
│   ├── poll_pod.py                     # YOU build
│   ├── phase6_generate.py              # YOU build
│   ├── autonomous_loop.sh              # YOU build
│   └── create_final_pr.sh              # YOU build
├── runs/                               # per-run artifacts, pushed to branches
│   ├── train-run-<stamp>/              # training logs
│   └── eval-run-<stamp>/               # eval samples + metrics
└── HANDOFF_CODEX.md                    # this file
```

## 9. Stop conditions

End loop and ntfy alert when ANY of:
- Budget spent > $25
- cycle_counter ≥ 3
- 2 consecutive cycles with bigram_jsd delta < 0.01 (no improvement)
- Same pod failure mode 2 cycles in a row
- `.state/STOP` file exists (emergency kill switch)

On any stop condition, always push a final `runs/summary-<stamp>/` branch with cycle logs + best eval report + decision trail.

## 10. When in doubt

- Prefer ntfy alert + stop over best-guess retry
- Prefer atomic commits per cycle (not per file)
- Never disable preflight smoke test in chain_train.sh
- Never embed tokens in dockerStartCmd (use shell expansion only)
- Data is 67k raw → 92k oversampled; recipe tweaks are the correct lever, NOT dataset changes (Phase 1 is frozen)
