# V3 Training Readiness

Date: 2026-04-27

## Scope

- Dataset under review: `cpt_corpus.v3.jsonl`
- Launch path under review: `scripts/launch_train_pod.py` -> `chain_train.sh`
- Cost model under review: `scripts/local_verification_loop.py` defaults (`sec_per_step=18.43`, `hourly_usd=0.79`, effective batch `16`)

## Kind Distribution vs Source

`cpt_corpus.v3.jsonl` does not expose a `source` field. It exposes `source_id` and `source_field`, and `source_field` is the field that currently aligns with `kind`.

| kind | rows | share | source_field |
| --- | ---: | ---: | --- |
| `comment` | 51,180 | 83.02% | `comment` |
| `post` | 10,470 | 16.98% | `post` |
| `total` | 61,650 | 100.00% | — |

Additional note:

- `length_bucket="패치"` accounts for `14,794` rows (`24.00%` of the full CPT corpus), all under `kind=comment`.
- Any future “kind vs source” audit for `v3` should key on `source_field`, not `source`.

## Cost Check

Corrected step math:

- `ceil(61650 / 16) = 3854`
- `2 epochs = 3854 * 2 = 7708` steps
- The task note’s `7707` is off by one

At the current verifier assumptions:

| profile | steps | hours | CPT cost |
| --- | ---: | ---: | ---: |
| `1 epoch` | `3854` | `19.73` | `$15.59` |
| `2 epochs` | `7708` | `39.46` | `$31.17` |
| `2 epochs capped at $30` | `7417` | `37.97` | `$30.00` |
| `full-chain cap with 3h merge/upload buffer` | `6831` | `34.97` | `$27.63` |

Interpretation:

- `2 epochs` on `v3` is over the `$30` CPT ceiling by about `$1.17`.
- `1 epoch` is comfortably inside the ceiling.
- If the goal is “as much CPT as possible while keeping CPT itself under `$30`”, the numeric cap is `CPT_MAX_STEPS=7417` (`~1.924` epochs).
- If the goal is “fit the whole chain inside `$30` with roughly 3 hours of merge/upload headroom”, the safer cap is `CPT_MAX_STEPS=6831` (`~1.772` epochs).

## Recommendation

For the strict `budget30` path, use one of these:

1. Safest:
   - `CPT_NUM_EPOCHS=1`
   - `SKIP_SFT=1`
   - Reason: simple, auditable, no fractional-step tuning needed, leaves enough headroom for post-CPT stages.
2. Tight CPT-only cap:
   - `CPT_NUM_EPOCHS=2`
   - `CPT_MAX_STEPS=7417`
   - Reason: fits the CPT stage under `$30`, but leaves little to no room for merge/upload if the budget is truly end-to-end.
3. Tight full-chain cap:
   - `CPT_NUM_EPOCHS=2`
   - `CPT_MAX_STEPS=6831`
   - Reason: preserves roughly 3 hours of non-CPT runtime inside the same `$30` envelope.

I would keep `budget30` on option `1` unless there is a deliberate decision to spend almost the entire budget on CPT alone.

## Readiness Notes

- `runs/refinement-v3-report.md` already shows `v3` is the better CPT baseline than `v2`.
- Before this task, the launch flow still defaulted to `v2` filenames and relied on swap-based validation. The launch wrapper now auto-resolves `cpt_corpus.v3.jsonl` when it exists, so the `v3 -> v2 filename swap` workaround is no longer required on the RunPod path.
- The verifier itself still reads `v2` filenames by default, so the launch path is `v3`-ready, but the local readiness report is not yet fully `v3`-native.
