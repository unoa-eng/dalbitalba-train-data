# Tokenizer Extension Verification

Date: 2026-04-29

## Scope

- Extended the `Qwen/Qwen3-8B-Base` tokenizer with the structured CPT v3 markers used by `v3-data/cpt_structured_v3.jsonl`.
- Verified that each structural marker encodes as a single token with the saved tokenizer in `v3-data/tokenizer/`.
- Verified that `scripts/launch_train_pod.py --dry-run` forwards `CPT_TOKENIZER_DIR` into the pod payload.

## Commands

```bash
.venv/bin/python scripts/extend_tokenizer_v3.py
python3 -m py_compile scripts/extend_tokenizer_v3.py train_cpt.py scripts/launch_train_pod.py
set -a; source recipes/budget30_v2.env; set +a
RUNPOD_API_KEY=dummy HF_TOKEN=dummy HF_USERNAME=unoa GITHUB_TOKEN=dummy \
  .venv/bin/python scripts/launch_train_pod.py --dry-run
```

## Tokenizer Build Result

- Base tokenizer size: `151669`
- Extended tokenizer size: `151676`
- Added tokens: `7`
- Saved tokenizer dir: `v3-data/tokenizer/`

Added token IDs:

| Token | ID |
| --- | ---: |
| `<|post|>` | `151669` |
| `<|/post|>` | `151670` |
| `<|comment depth=0|>` | `151671` |
| `<|comment depth=1|>` | `151672` |
| `<|/comment|>` | `151673` |
| `<|thread|>` | `151674` |
| `<|/thread|>` | `151675` |

## Encode Verification

Each structured marker is split into multiple subword tokens by the base tokenizer, but exactly one token by the extended tokenizer:

| Token | Base encode length | Extended encode length | Extended ID |
| --- | ---: | ---: | ---: |
| `<|post|>` | `5` | `1` | `151669` |
| `<|/post|>` | `6` | `1` | `151670` |
| `<|comment depth=0|>` | `8` | `1` | `151671` |
| `<|comment depth=1|>` | `8` | `1` | `151672` |
| `<|/comment|>` | `6` | `1` | `151673` |
| `<|thread|>` | `5` | `1` | `151674` |
| `<|/thread|>` | `6` | `1` | `151675` |

Representative structured sample:

```text
<|thread|><|post|>제목: 테스트
본문<|/post|><|comment depth=0|>첫 댓글<|/comment|><|/thread|>
```

- Base tokenizer length: `44`
- Extended tokenizer length: `17`

## Train Loader Verification

- `train_cpt.dataset_contains_structured_tokens("v3-data/cpt_structured_v3.jsonl")` returned `true`.
- `train_cpt.resolve_tokenizer_source("v3-data/cpt_structured_v3.jsonl")` resolved to `v3-data/tokenizer`.
- `train_cpt.py` now raises if structured CPT markers are detected without `CPT_TOKENIZER_DIR`, preventing a silent launch with subword-split structural markers.

## Launcher Verification

`scripts/launch_train_pod.py --dry-run` output includes:

- `[gate] verifier PASS`
- `"TRAIN_CPT_JSONL": "***REDACTED***"`
- `"INPUT_JSONL": "***REDACTED***"`
- `"CPT_TOKENIZER_DIR": "***REDACTED***"`

Path normalization check:

- `normalize_workspace_repo_path("v3-data/tokenizer") -> /workspace/repo/v3-data/tokenizer`
- `normalize_workspace_repo_path("./v3-data/tokenizer") -> /workspace/repo/v3-data/tokenizer`

## Verdict

`PASS` — the v3 structured CPT markers now have dedicated tokenizer IDs, the extended tokenizer is saved in-repo at `v3-data/tokenizer/`, and the train launcher forwards the tokenizer directory into the pod environment.
