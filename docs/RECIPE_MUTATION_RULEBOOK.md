# Recipe Mutation Rulebook (Claude-authored judgment layer)

**Purpose**: Encode the expert judgment for how to mutate the training recipe
based on eval-gate failure signals, so the autonomous loop can apply mutations
mechanically without needing an online LLM at each cycle.

**Authored**: 2026-04-24 by Claude Opus 4.7
**Consumed by**: `scripts/recipe_mutator.py` — reads `runs/eval-run-<latest>/metrics.json`
+ current recipe env, emits updated recipe env for the next pod launch.

---

## Axes of mutation (ordered by expected impact per $ spent)

| Axis | Default | Cheap range | Expensive range | Mutation cost |
|------|---------|-------------|-----------------|---------------|
| `CPT_NUM_EPOCHS` | 1 | 2 | 3+ | +80% CPT time per extra epoch |
| `LORA_R` | 64 | 128 | 256 | +VRAM only, ~same time |
| `CPT_USE_DORA` | 0 | 1 | — | +30% train time, +quality (arxiv 2402.09353) |
| `OVERSAMPLE_LG_XXL` | 2× | 3× | 4× | Data regen required (no pod cost) |
| `OVERSAMPLE_DIGIT_ENG` | 2× | 4× | 6× | Data regen required |
| `SFT_NUM_EPOCHS` | 2 | 3 | — | +40% SFT time |
| `CPT_LR` | 2e-4 | 1e-4 | 5e-4 | Same time, different convergence |
| `SFT_LR` | 5e-5 | 2e-5 / 1e-4 | — | Same time |
| `SEQ_LEN` | 1024 | 2048 | — | 2× memory, 1.5× train time |

## Rules (first matching rule wins per cycle)

**R0 — Emergency stop (check first):**
- IF regression > 20% on ANY metric vs last cycle → `ROLLBACK` to previous winning recipe, ntfy `STOP_ROLLBACK`, exit loop.
- IF `cycle_counter >= 5` → `STOP`, commit final report.
- IF `budget_spent >= 25.0 USD` → `STOP`, commit final report.

**R1 — High bigram JSD (most common failure):**
- IF `bigram_jsd > 0.15`:
  - First occurrence → `CPT_NUM_EPOCHS = 2` (if currently 1)
  - Second (still > 0.15 with 2 epochs) → `LORA_R = 128` (if currently 64)
  - Third → `LORA_R = 128` AND `CPT_USE_DORA = 1`
  - Fourth → `STOP` (bigram JSD not responsive to any cheap lever; investigate data/tokenizer)

**R2 — Length distribution mismatch:**
- IF `length_kl > 0.10`:
  - First → set `OVERSAMPLE_LG_XXL = 3`, regenerate cpt_corpus via `scripts/phase1_data_pipeline.py`
  - Second → `OVERSAMPLE_LG_XXL = 4` AND `SEQ_LEN = 2048` (trade-off: memory for long-form)

**R3 — Character class density mismatch:**
- IF `digit_density_delta > 0.03` OR `english_density_delta > 0.02`:
  - Bump `OVERSAMPLE_DIGIT_ENG` from 2 to 4
  - Never exceed 6 (otherwise synthetic overweight hurts bigram fidelity)

**R4 — MAUVE semantic gap:**
- IF `mauve_score < 0.80` AND bigram/length are PASS:
  - Enable `CPT_USE_DORA = 1` (weight decomposition helps semantic match)
  - If DoRA already on → bump `SFT_NUM_EPOCHS` from 2 to 3

**R5 — All close but no pass (boundary case):**
- IF 3 of 5 metrics are within 10% of threshold, others fail marginally:
  - Extend `CPT_NUM_EPOCHS` to 2 (cheapest global improvement)

**R6 — No signal, try parameter space expansion:**
- IF no specific metric dominates failure (spread roughly uniform over thresholds):
  - Bump `LORA_R = 128` (more capacity)

**R7 — Architecture switch (LoRA-CPT cap reached):**
- Trigger: `bigram_jsd > 0.15` AND `LORA_R >= 128` AND `CPT_USE_DORA == 1` AND `CPT_NUM_EPOCHS >= 2`
- Rationale: arXiv:2405.09673 ("LoRA Learns Less and Forgets Less", May 2024) shows LoRA-CPT is roughly 16× data-inefficient vs full fine-tune for domain adaptation. The LoRA-only escalation chain (R1 → R1b → R1c) cannot dig past this floor on a 92K Korean community corpus.
- Mutation: emit `CPT_FULL_FT=1`, `LORA_R=256`, `LORA_ALPHA=256`. Downstream `chain_train.sh` / `launch_train_pod.py` interpret `CPT_FULL_FT=1` as "switch to fp16 full fine-tune" (or, if VRAM-bound on L40S, fall back to r=256 rsLoRA).
- Stop guard: if `CPT_FULL_FT=1` is already set in recipe and `bigram_jsd > 0.15` for two more cycles, hard-stop with `R7_EXHAUSTED` (the model class itself is the wrong choice; consider Bllossom-8B or Open-Ko-8B as base).

**R8 — Data-bound stagnation (knobs exhausted):**
- Trigger: last 2 cycles applied no recipe changes AND `|Δ bigram_jsd| < 0.005`. Implementation looks at `history[-2:]` `recipe_changes`. (Distinct from R0's general stagnation: this checks specifically that the recipe was not mutated, so the stagnation is attributable to the corpus, not to a mid-mutation plateau.)
- Rationale: When the recipe knobs are truly exhausted but JSD does not move, the bottleneck is not in the recipe — it is in the corpus. The most common offenders are: cross-thread duplicate ad templates that thread-internal Jaccard dedup does not catch (FineWeb pattern, arXiv:2406.17557), incomplete NFKC + Compatibility Jamo restoration, and the 21.6% residual ad keyword rate measured directly on `cpt_corpus.v2.jsonl` at 2026-05-07.
- Mutation: emit data-regen request without changing the recipe.
  ```
  REGEN_DATA=1
  REGEN_REASON=stagnation
  REGEN_ENABLE_MINHASH=1            # global MinHash dedup (scripts/dedup_minhash.py)
  REGEN_ENABLE_ENTROPY_FILTER=1     # FineWeb char-5gram entropy gate
  REGEN_MIN_ENTROPY=3.8             # bits per char 5-gram; 0 disables
  ```
- Supervisor behaviour: `autonomous_loop.sh` should detect `REGEN_DATA=1`, run `phase1_data_pipeline.py` + `clean_ad_spam.py --min-entropy 3.8`, regenerate the corpus, then relaunch training with the same recipe. The next cycle's metrics on the cleaner data tell us whether the stagnation was data-bound (JSD drops) or architecture-bound (R7 fires next).

## Escalation — human check required

Write `.state/ESCALATE.md` with cycle context + ntfy alert "ESCALATE:" prefix when:
- Any CUDA / driver / OOM error appears 2× in a row
- Any novel error string not seen in loop's failure catalog
- Cycle completes without emitting metrics.json (eval corruption)
- HF hub push fails 3× in same cycle

## State tracking format (`.state/loop_state.json`)

```json
{
  "cycle": 3,
  "recipe": {
    "BASE_MODEL": "Qwen/Qwen3-8B-Base",
    "CPT_NUM_EPOCHS": 2,
    "SFT_NUM_EPOCHS": 2,
    "LORA_R": 128,
    "CPT_USE_DORA": 0,
    "OVERSAMPLE_LG_XXL": 3,
    "OVERSAMPLE_DIGIT_ENG": 2
  },
  "last_metrics": {
    "bigram_jsd": 0.12,
    "length_kl": 0.08,
    "digit_density_delta": 0.015,
    "english_density_delta": 0.010,
    "mauve_score": 0.78
  },
  "history": [
    { "cycle": 1, "metrics": {...}, "verdict": "FAIL", "mutation_applied": "R1: CPT_NUM_EPOCHS 1→2" },
    { "cycle": 2, "metrics": {...}, "verdict": "FAIL", "mutation_applied": "R1: LORA_R 64→128" }
  ],
  "budget_spent_usd": 12.4,
  "stop_reason": null
}
```

## Example rulebook application trace

- Cycle 1: metrics={jsd: 0.21, kl: 0.09, ...} → R1 fires → mutation "CPT_NUM_EPOCHS 1→2"
- Cycle 2: metrics={jsd: 0.14, kl: 0.09, ...} → R1 again (still >0.15? no — 0.14 < 0.15, R1 doesn't fire). Next check R2: 0.09 < 0.10, pass. Next R3, R4, R5. If all pass → PR.
- Cycle 3: only runs if cycle 2 failed.

### Trace including R7 / R8 (post-2026-05-07 hardening)

- Cycle 1: jsd=0.22 → R1 (epochs 1→2)
- Cycle 2: jsd=0.18 → R1b (LORA_R 64→128)
- Cycle 3: jsd=0.17 → R1c (CPT_USE_DORA 0→1)
- Cycle 4: jsd=0.165 → R1_EXHAUSTED soft (no recipe change, history records R1_EXHAUSTED)
- Cycle 5: jsd=0.163 → recipe unchanged for 2 cycles, |Δjsd|<0.005 → **R8** fires (regen data with MinHash + entropy gate)
- Cycle 6 (post-regen): if jsd drops to 0.10 → continue with R1/R2/etc. on cleaner data; if jsd still ≥0.15 → **R7** fires (full fine-tune escalation).

## Non-negotiable invariants

These axes are frozen; mutator must NEVER change:
- `BASE_MODEL` (Qwen/Qwen3-8B-Base)
- CONTAINER_IMAGE (pytorch:2.4.0-cuda12.4.1)
- `bf16=True`, `fp16=False`
- `paged_adamw_32bit`
- Preflight smoke test in chain_train.sh
- `HF_HUB_ENABLE_HF_TRANSFER=1`
- GPU_TYPE priority: L40S first

## Convergence heuristic (auto-stop when near ceiling)

Raw-vs-raw baseline bigram JSD = 0.019 (achievability ceiling).
IF `bigram_jsd < 0.04` AND all other gates pass:
- Declare `CONVERGED`, run final PR, stop loop.

`bigram_jsd < 0.04` means we're within 2× of the intrinsic raw data variability,
which is indistinguishable from raw on a full-spectrum comparison.
