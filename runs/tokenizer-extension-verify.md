# Tokenizer Extension Verification

Date: 2026-04-29
Branch: `budget30-pre-launch`

## Scope

- Rebuilt `v3-data/tokenizer/` from `Qwen/Qwen3-8B-Base`.
- Confirmed the structured CPT marker set now covers depths `0..5`.
- Confirmed `train_cpt.py` can load the saved tokenizer bundle and re-apply the manifest at runtime when `CPT_EXTEND_TOKENS=1`.

## Commands

```bash
python3 -m py_compile scripts/extend_tokenizer_v3.py train_cpt.py scripts/launch_train_pod.py scripts/local_verification_loop.py
source .venv/bin/activate
python3 scripts/extend_tokenizer_v3.py --model Qwen/Qwen3-8B-Base --out-dir v3-data/tokenizer
export TRAIN_CPT_JSONL=v3-data/cpt_structured_v3.jsonl
export CPT_TOKENIZER_DIR=v3-data/tokenizer
export CPT_EXTEND_TOKENS=1
export CPT_LOG_FILE=runs/train_cpt_tokenizer_import.log
python3 - <<'PY'
from transformers import AutoTokenizer
import train_cpt

train_jsonl = train_cpt.resolve_train_jsonl()
source, structured = train_cpt.resolve_tokenizer_source(train_jsonl)
runtime_tokens = train_cpt.load_runtime_special_tokens()
tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
added = tokenizer.add_special_tokens({"additional_special_tokens": runtime_tokens})
print(train_jsonl, source, structured, len(runtime_tokens), added)
PY
```

## Build Result

- Base tokenizer size: `151669`
- Extended tokenizer size: `151680`
- Added tokens: `11`
- Manifest file: `v3-data/tokenizer/token_list.json`

Added token IDs:

| Token | ID |
| --- | ---: |
| `<|post|>` | `151669` |
| `<|/post|>` | `151670` |
| `<|comment depth=0|>` | `151671` |
| `<|comment depth=1|>` | `151672` |
| `<|comment depth=2|>` | `151673` |
| `<|comment depth=3|>` | `151674` |
| `<|comment depth=4|>` | `151675` |
| `<|comment depth=5|>` | `151676` |
| `<|/comment|>` | `151677` |
| `<|thread|>` | `151678` |
| `<|/thread|>` | `151679` |

## Runtime Probe

Saved in `runs/train_cpt_tokenizer_probe.log`.

- `structured_detected=True`
- `tokenizer_source=v3-data/tokenizer`
- `runtime_token_count=11`
- `runtime_added=0`

Interpretation:

- The saved tokenizer bundle already contains the full marker set.
- The runtime manifest path is still valid, so a pod can re-apply the same 11 tokens safely when `CPT_EXTEND_TOKENS=1`.
- If the saved tokenizer directory is missing or incomplete, `train_cpt.py` now falls back to `BASE_MODEL` plus runtime token extension instead of blocking launch.

## Verdict

`PASS` — the tokenizer artifact on disk and the runtime fallback path now agree on the same 11 structured CPT tokens.
