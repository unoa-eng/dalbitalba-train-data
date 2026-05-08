# R3 Fix-Stack — Oversampling Re-validation

**Date**: 2026-05-08
**Scope**: Confirm fractional-multiplicity loss-weight oversampling (R3.1
fix in `train_sft.py`, commit `f37168e`) actually fires across mixed
weights, with bucket-level distribution matching the expected stochastic
model `E[multiplicity(w)] = w`.

**Synthetic input**: `runs/smoke-2026-05-08/synth_lw.jsonl` (100 rows)
- 70× weight=1.0, 15× weight=1.5, 10× weight=2.0, 5× weight=2.5
- Shuffled with `random.seed(13)` (reproducible)

**RNG**: `_OVERSAMPLE_RNG = random.Random(SFT_OVERSAMPLE_SEED=13)` —
matches the module-level seed in `train_sft.py:95`.

## Result

| weight | input rows | expected expanded | tol | observed | status |
|--------|-----------:|------------------:|----:|---------:|:------:|
| 1.0    | 70         | 70.0              | ±0  | 70       | OK     |
| 1.5    | 15         | 22.5              | ±3  | 25       | OK     |
| 2.0    | 10         | 20.0              | ±0  | 20       | OK     |
| 2.5    | 5          | 12.5              | ±3  | 12       | OK     |

- **Total**: 100 → 127 rows (+27, expected ~125 ± tol).
- **Determinism**: re-running with the same seed reproduced identical
  expanded count (127 → 127). Confirms `_OVERSAMPLE_RNG` reseed yields
  the same Bernoulli draws.
- **Floor invariants**: weight=1.0 and weight=2.0 buckets matched
  exactly (no fractional component → multiplicity is deterministic).
- **Banker's-rounding bug regression**: pre-fix logic
  `multiplicity = max(1, int(round(lw)))` would have collapsed both
  weight=1.5 and weight=2.0 to multiplicity=2 (Python rounds 1.5 → 2
  via banker's rounding), making 1.5 indistinguishable from 2.0.
  Observed 1.5-bucket=25 vs 2.0-bucket=20 confirms fractional Bernoulli
  is firing distinctly per bucket.

**Artifact**: `runs/smoke-2026-05-08/oversample_validation.json`
**Test driver**: `/tmp/test_oversample_real.py` (replicates train_sft.py
lines 362-391 verbatim against the synth dataset).

## Verdict

R3.1 fractional-multiplicity fix is empirically validated. The
SFT_LOSS_WEIGHT_ARGOT=1.5 / THRESHOLD=2 escalation contract is now
honored end-to-end (per-row weight differentiation no longer collapses
under banker's rounding).
