# Refinement V3 Report

- Baseline diagnostic: `runs/refinement-20260427-165117/cycle-1/diagnostic.json`
- V3 diagnostic: `runs/refinement-20260427-165130/cycle-1/diagnostic.json`
- Validation command: `python3 scripts/refinement_loop.py --source-dir /Users/unoa/Downloads/crawled-data-v2 --cycle 1`
- Swap mode: `cpt_corpus.v3.jsonl` was temporarily copied onto `cpt_corpus.v2.jsonl` for the validation run, then `cpt_corpus.v2.jsonl` was restored.
- CPT rows: `46,856 -> 61,650` (`+14,794`, `+31.6%`)

## Summary

| Metric | Before (`v2`) | After (`v3`) | Change |
| --- | --- | --- | --- |
| Total gaps | `9` moderate | `1` moderate | `-8` gaps |
| Critical gaps | `0` | `0` | flat |
| Moderate gaps | `9` | `1` | `-8` |
| Training texts seen by diagnostic | `138,255` | `153,049` | `+14,794` |
| Training source IDs | `11,232` | `18,133` | `+6,901` |
| Source coverage | `99.52%` | `99.52%` | flat |

## GAP Term Comparison

| Term | Before ratio | After ratio | Train count delta | Result |
| --- | --- | --- | --- | --- |
| `TC` | `0.1048` | `0.5722` | `463 -> 2,621` | resolved |
| `밀빵` | `0.2289` | `0.6852` | `1,087 -> 3,374` | resolved |
| `케어` | `0.2564` | `0.7242` | `2,608 -> 7,637` | resolved |
| `쩜오` | `0.3979` | `0.7055` | `2,351 -> 4,322` | resolved |
| `하이퍼` | `0.3486` | `0.7420` | `87 -> 192` | resolved |
| `도파민` | `0.4992` | `0.6783` | `3,478 -> 4,900` | resolved |
| `갯수` | `0.4588` | `0.8776` | `2,000 -> 3,967` | resolved |
| `선릉` | `0.3980` | `0.5798` | `143 -> 216` | resolved |
| `ㅡㅡ` | `0.4915` | `0.4767` | `3,885 -> 3,907` | still below threshold |

## Interpretation

- The additive GAP repair patch behaved as intended for the business/location vocabulary that was previously lost to the promo filter. All eight targeted business terms that were below the `0.5` ratio cutoff in `v2` moved above the cutoff in `v3`.
- `TC` was the biggest recovery: the ratio moved from `0.1048` to `0.5722`, which indicates the additive patch repaired exactly the most damaging underrepresentation called out in the evaluation strategy.
- The only remaining gap is `ㅡㅡ` (`0.4767`). This is effectively flat versus baseline and suggests the additive promo-only repair helps domain/business tokens much more than emotive tone markers.
- Structural metrics stayed stable: there were still `0` structure gaps and source coverage stayed at `99.52%`, so the patch did not introduce a coverage regression while expanding CPT by `31.6%`.

## Evaluation Gate Changes

- `scripts/phase6_eval.py` now uses `bigram_jsd <= 0.08` instead of `<= 0.15`.
- Added `domain_keyword_alignment`, defined as the minimum generated/raw sample-presence ratio across 20 core GAP terms, with gate `>= 0.50`.
- Added `tone_distribution_match`, defined as the maximum absolute delta between generated/raw `반말` and `존댓말` ratios, with gate `<= 0.15`.
- `scripts/cycle_report.py` now renders the two new metrics, and `scripts/recipe_mutator.py` uses the tightened `bigram_jsd` threshold (`0.08`) so mutation logic stays aligned with the actual gate.

## Recommendation

- `v3` is a strong improvement over `v2` for GAP repair and should be treated as the new CPT baseline for the next training/eval pass.
- If we want to clear the final remaining dataset-side gap before training, the next patch should target short emotive sentences containing `ㅡㅡ` rather than more promo-derived business vocabulary.
