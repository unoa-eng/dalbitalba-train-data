# V3 Verifier False-Positive Fix

- Timestamp: `2026-04-29T03:54:28Z`
- Target corpus: `v3-data/cpt_structured_v3.jsonl`
- Task: remove confirmed false-positive verifier hits and re-run the local verifier under `budget30`

## Removed Rows

Removed `5` rows from `v3-data/cpt_structured_v3.jsonl`.

1. Original line `6163`
   - `source_id=1197168`
   - `kind=comment`
   - Reason: institutional phone number (`국립정신건강센터 ... 02-2204-0114`) triggered `phone_like`
2. Original line `6164`
   - `source_id=1197168`
   - `kind=thread`
   - Reason: thread duplicate of the same institutional phone number triggered `phone_like`
3. Original line `40997`
   - `source_id=1189688`
   - `kind=comment`
   - Reason: numeric percentages like `99.999999` / `0.000000001` matched the phone regex format and triggered `phone_like`
4. Original line `41002`
   - `source_id=1189688`
   - `kind=thread`
   - Reason: thread duplicate of the same numeric-format false positive triggered `phone_like`
5. Original line `40830`
   - `source_id=1189721`
   - `kind=thread`
   - Reason: `중딩`/`미성년자` proximity to `조건만남` in a rejecting/accusatory context triggered `minor_sexual_proximity`; this was a heuristic mismatch rather than exploitative content

## Before / After

- Before removal:
  - `phone_like=4`
  - `minor_sexual_proximity=1`
- After removal:
  - `phone_like=0`
  - `minor_sexual_proximity=0`
  - Remaining rows: `53,675`

## Verification

Command run:

```bash
TRAIN_CPT_JSONL=v3-data/cpt_structured_v3.jsonl INPUT_JSONL=v3-data/cpt_structured_v3.jsonl python3 scripts/local_verification_loop.py --profile budget30
```

Result:

- Verdict: `PASS`
- Report: `runs/local-verification-20260429-035428/report.md`
- Severe findings: `0`
- Warnings: `0`
- CPT rows: `53,675`
- Estimated CPT train time: `17.18h`
- Estimated CPT cost: `$13.57`

## Recipe Update

Updated `recipes/budget30_v2.env`:

```bash
TRAIN_CPT_JSONL=v3-data/cpt_structured_v3.jsonl
```

This promotes the cleaned structured v3 CPT file into the active `budget30_v2` recipe.
