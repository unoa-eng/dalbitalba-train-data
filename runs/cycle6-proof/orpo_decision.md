# ORPO Phase-4 Activation Decision (cycle-6)

**Date**: 2026-05-13
**Repo HEAD**: main (post cycle-6 P0 bundle merged via PR #9 + #10)
**Analyzing agent**: Sonnet ORPO-readiness, independent run

---

## Readiness Matrix

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `orpo_pairs.jsonl` judged-pair quality | **NO** | `scripts/round2_build_orpo_pairs.py:8-13` — `chosen` is pure heuristic (PASS verdict OR `bigram_jsd ≤ 0.10`); all 1,472 pairs have `source_run_chosen=cpt_corpus.v3.jsonl` (100% cpt_corpus backfill); zero rows carry any `judge`/`llm_judge`/`human_label` field |
| 2 | `train_orpo.py` tokenizer_v4 + `resize_token_embeddings` | **NO** | `train_orpo.py:79,82` — loads `AutoTokenizer.from_pretrained(base_model)` (no `tokenizer_v4` path); `AutoModelForCausalLM` loads with `torch_dtype="auto"` only; `model.resize_token_embeddings()` never called |
| 3 | `train_orpo.py` 4-bit quantization (memory headroom) | **NO** (non-blocking) | `train_orpo.py:82` — `torch_dtype="auto"` (bf16/fp16), no `BitsAndBytesConfig`; `bf16=True` in `ORPOConfig:103`; on L40S 48GB with 8B + LoRA at batch=1/GAS=16 it fits but zero headroom |
| 4 | Thread-level holdout on ORPO `chosen` backfill | **PARTIAL** | `round2_integrity_check.py:75-94` `check_orpo_leak` uses exact-text match only against `val_set.v3.jsonl` + eval holdout; `round2_build_orpo_pairs.py:116,224-229` backfill also exact-text only — thread-level dedup absent, a thread containing a val comment can contribute other comments as chosen |

---

## Decision: **FIX-FIRST** (do not activate this cycle)

The recipe's own `ORPO_STATUS=deferred_until_real_judged_pairs` (`recipes/round2-cycle1.env:46`) is correct and self-consistent. **Every single ORPO chosen row is a heuristic cpt_corpus backfill with zero judgment signal** — activating now would train a preference signal that is effectively random noise dressed up as preference. Criteria 1 and 2 are hard blockers; 4 is also a hard blocker for paper-grade; 3 is a hardening step.

---

## Required fixes before activation (priority order)

### P0 — Criterion 1: real judgment signal for `orpo_pairs.jsonl`
- **File**: `scripts/round2_build_orpo_pairs.py:8-13` (heuristic comment + `chosen` logic)
- **Patch sketch**: run at minimum an LLM-judge pass (Claude `claude-sonnet-4-5-20250929` or GPT-4o `gpt-4o-2024-11-20`) scoring `chosen` vs `rejected` on "naturalness + community tone"; store `{"judge": "llm", "score_chosen": X, "score_rejected": Y}`; emit pairs only where `score_chosen − score_rejected ≥ 1.0` (on a 0-5 scale)
- **Why P0**: without judgment, ORPO learns noise

### P0 — Criterion 2: tokenizer_v4 path + vocab resize
- **File**: `train_orpo.py:79,82`
- **Patch sketch** (mirrors `train_sft.py` pattern):
  ```diff
  - tokenizer = AutoTokenizer.from_pretrained(base_model)
  + tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER_PATH", base_model))
  ```
  Insert after L82:
  ```python
  model.resize_token_embeddings(len(tokenizer))
  ```
- **Why P0**: without resize, token IDs ≥ 151643 (the 265 added-tokens range) cause silent OOB or runtime crash

### P0 — Criterion 4 (combined with 1): thread-level holdout on backfill
- **Files**: `scripts/round2_build_orpo_pairs.py:116` (backfill filter loop) + `scripts/round2_integrity_check.py:75` (`check_orpo_leak`)
- **Patch sketch**: add `--val-root-ids` arg; load `root_ids` from `val_set.v3.jsonl` + `sft_thread_conditioned.eval.jsonl`; in backfill loop (`round2_build_orpo_pairs.py:173`) add `if r.get("root_id") in val_root_ids: continue`; mirror same `root_id` guard in `check_orpo_leak`
- **Why P0**: paper-grade thread-holdout invariant must hold across all phases

### P1 — Criterion 3: optional 4-bit quantization path
- **File**: `train_orpo.py:82`
- **Patch sketch**: add `if os.environ.get("ORPO_LOAD_IN_4BIT") == "1":` branch using `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` — matches `train_cpt.py`
- **Why P1**: hardening only; current fp16+LoRA at batch=1 fits but with no headroom, deferred is fine

---

## Activation gate (for next cycle)

```bash
# Required state before flipping ORPO_NUM_EPOCHS to ≥1 in round2-cycle1.env:
test "$(jq '.[0]|select(.judge=="llm")' orpo_pairs.jsonl)" != ""   # P0-1 evidence
grep -q "resize_token_embeddings" train_orpo.py                     # P0-2 evidence
.venv/bin/python scripts/round2_integrity_check.py | grep -E "root_id.*intersect=0"  # P0-4 evidence
```

If all three checks PASS, flip the recipe knob; otherwise stay deferred.

---

## Paper-grade note

Updating H2's "ORPO reject-action" clause in `TRAINING_DESIGN_V3.md §0` to honestly reflect that ORPO is deferred this cycle (and only the CPT + thread-conditioned SFT pipeline is being evaluated) is more honest than activating with synthetic preference pairs. **Defer publicly, not silently.**
