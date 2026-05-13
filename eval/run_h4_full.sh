#!/bin/bash
# H4 paper-grade regression gate using lm-evaluation-harness.
# Runs the canonical KoBEST + HAE-RAE task suite against a base model and an
# adapter-applied model, then enforces Δ ≤ -0.05 (5pp drop) per task.
#
# This is the heavyweight path; `eval/eval_kobest.py` + `eval/eval_haerae.py`
# remain as the lightweight sanity-dry-run path for macmini smoke.
#
# Required env:
#   H4_BASE_MODEL     — HF model id of the base (e.g. Qwen/Qwen3-8B-Base)
#   H4_ADAPTER_PATH   — path to merged or LoRA adapter
#   H4_OUTPUT_DIR     — where to write *.json reports
#
# Optional:
#   H4_TASKS          — comma list (default: kobest,haerae)
#   H4_MAX_DROP       — accuracy drop threshold (default: 0.05)
#   H4_DEVICE         — torch device (default: mps on mac, cuda elsewhere)

set -euo pipefail

H4_BASE_MODEL="${H4_BASE_MODEL:?H4_BASE_MODEL required}"
H4_ADAPTER_PATH="${H4_ADAPTER_PATH:?H4_ADAPTER_PATH required}"
H4_OUTPUT_DIR="${H4_OUTPUT_DIR:?H4_OUTPUT_DIR required}"
H4_TASKS="${H4_TASKS:-kobest,haerae}"
H4_MAX_DROP="${H4_MAX_DROP:-0.05}"
H4_DEVICE="${H4_DEVICE:-mps}"

mkdir -p "$H4_OUTPUT_DIR"
PYBIN="${PYBIN:-.venv/bin/python}"

if ! "$PYBIN" -c "import lm_eval" >/dev/null 2>&1; then
    echo "[SKIP] lm-eval-harness not installed — run: $PYBIN -m pip install lm-eval"
    exit 0
fi

echo "[H4] base=$H4_BASE_MODEL adapter=$H4_ADAPTER_PATH tasks=$H4_TASKS device=$H4_DEVICE"

"$PYBIN" -m lm_eval \
    --model hf \
    --model_args "pretrained=${H4_BASE_MODEL},dtype=float16,device_map=${H4_DEVICE}" \
    --tasks "$H4_TASKS" \
    --output_path "${H4_OUTPUT_DIR}/base.json" \
    --num_fewshot 0 2>&1 | tee "${H4_OUTPUT_DIR}/base.log"

"$PYBIN" -m lm_eval \
    --model hf \
    --model_args "pretrained=${H4_BASE_MODEL},peft=${H4_ADAPTER_PATH},dtype=float16,device_map=${H4_DEVICE}" \
    --tasks "$H4_TASKS" \
    --output_path "${H4_OUTPUT_DIR}/adapter.json" \
    --num_fewshot 0 2>&1 | tee "${H4_OUTPUT_DIR}/adapter.log"

"$PYBIN" -c "
import json, sys, glob
base = next(iter(json.load(open(glob.glob('${H4_OUTPUT_DIR}/base.json*')[0]))['results'].items()))
adapter = next(iter(json.load(open(glob.glob('${H4_OUTPUT_DIR}/adapter.json*')[0]))['results'].items()))
# Compare per-task accuracy
import json
base_res = json.load(open(glob.glob('${H4_OUTPUT_DIR}/base.json*')[0]))['results']
adp_res = json.load(open(glob.glob('${H4_OUTPUT_DIR}/adapter.json*')[0]))['results']
report = {'tasks': {}, 'max_drop_threshold': ${H4_MAX_DROP}, 'verdict': 'PASS', 'failures': []}
for task in base_res:
    b = base_res[task].get('acc,none', base_res[task].get('acc', 0))
    a = adp_res.get(task, {}).get('acc,none', adp_res.get(task, {}).get('acc', 0))
    drop = a - b
    fail = drop < -${H4_MAX_DROP}
    report['tasks'][task] = {'base': b, 'adapter': a, 'drop': round(drop, 4), 'fail': fail}
    if fail:
        report['verdict'] = 'FAIL_H4'
        report['failures'].append(f'{task}: drop={drop:.4f}')
open('${H4_OUTPUT_DIR}/h4_summary.json', 'w').write(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({'verdict': report['verdict'], 'failures': report['failures'], 'report': '${H4_OUTPUT_DIR}/h4_summary.json'}))
sys.exit(0 if report['verdict'] == 'PASS' else 2)
"
