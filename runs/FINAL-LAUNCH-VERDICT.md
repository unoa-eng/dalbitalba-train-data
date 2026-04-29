# Final Launch Verdict

Date: 2026-04-29
Branch: `budget30-pre-launch`
Verdict: `GO`

## Evidence

| Check | Result | Evidence |
| --- | --- | --- |
| Tokenizer extension script | `PASS` | `source .venv/bin/activate && python3 scripts/extend_tokenizer_v3.py --model Qwen/Qwen3-8B-Base --out-dir v3-data/tokenizer` completed successfully |
| Structured token inventory | `PASS` | Saved tokenizer now includes `<|post|>`, `<|/post|>`, `<|thread|>`, `<|/thread|>`, `<|/comment|>`, `<|comment depth=0|>` through `<|comment depth=5|>` |
| Runtime tokenizer fallback | `PASS` | `train_cpt.py` now honors `CPT_TOKENIZER_DIR` and `CPT_EXTEND_TOKENS=1`, loads `token_list.json` when present, and resizes embeddings after runtime token insertion |
| Pod v3-data staging | `PASS` | `scripts/launch_train_pod.py` startup command now copies `cp -r /workspace/repo/v3-data /workspace/data/v3-data`; `chain_train.sh` repeats the same copy inside the pod before data checks |
| Pod path resolution | `PASS` | Recipe path normalizes to `/workspace/data/v3-data/cpt_structured_v3.jsonl`; tokenizer dir normalizes to `/workspace/repo/v3-data/tokenizer` |
| Launch recipe wiring | `PASS` | `recipes/budget30_v2.env` now sets `CPT_TOKENIZER_DIR=v3-data/tokenizer` and `CPT_EXTEND_TOKENS=1` |
| Launch dry-run | `PASS` | `python3 scripts/launch_train_pod.py --dry-run` passed the budget gate and rendered a valid RunPod payload with `CPT_TOKENIZER_DIR`, `CPT_EXTEND_TOKENS`, and the new `v3-data` staging step |
| Local verification loop | `PASS` | `python3 scripts/local_verification_loop.py --profile budget30` returned `{"verdict":"PASS"}` and wrote `runs/local-verification-20260429-043209/report.md` |
| Syntax smoke | `PASS` | `python3 -m py_compile train_cpt.py scripts/launch_train_pod.py scripts/extend_tokenizer_v3.py` succeeded |

## Notes

- `recipes/budget30_v2.env` remains a plain shell-assignment file. Launch validation used `set -a` before sourcing it so the Python process inherited the recipe values in `zsh`.
- `.gitignore` now covers `.venv*/`, local HF cache paths, and `runs/**/adapter/` so the required commit can exclude local-only envs and adapter weights cleanly.

## Decision

Launch is clear from a blocker perspective.

The tokenizer artifact exists on disk, runtime extension is wired as fallback, the selected v3 CPT JSONL is staged into the pod at the path the recipe resolves to, the launch wrapper accepts the recipe, and the local verification loop is currently `PASS`.
