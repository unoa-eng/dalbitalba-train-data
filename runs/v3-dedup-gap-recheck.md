# V3 Dedup GAP Recheck

- Date: `2026-04-27`
- Required command run: `python3 scripts/refinement_loop.py --source-dir /Users/unoa/Downloads/crawled-data-v2 --cycle 1`
- New diagnostic: `runs/refinement-20260427-171435/cycle-1/diagnostic.json`
- Prior full-patch reference: `runs/refinement-20260427-165130/cycle-1/diagnostic.json`
- Baseline reference: `runs/refinement-20260427-165117/cycle-1/diagnostic.json`

## Method

`scripts/refinement_loop.py` is hardcoded to read `cpt_corpus.v2.jsonl`, so I used the same validation method as the earlier `v3` report:

1. Backed up `cpt_corpus.v2.jsonl`
2. Copied deduped `cpt_corpus.v3.jsonl` (`48,125` rows) onto `cpt_corpus.v2.jsonl`
3. Ran the required command
4. Restored the original `cpt_corpus.v2.jsonl`

Dataset sizes relevant to this recheck:

- `cpt_corpus.v2.jsonl`: `46,856` rows
- prior full patch `v3`: `61,650` rows (`+14,794`)
- current deduped `v3`: `48,125` rows (`+1,269`)
- `cpt_patch_gap_repair_dedup.jsonl`: `1,269` rows

## Verdict

Dedup undid most of the earlier GAP repair.

- Prior full-patch `v3` had `1` moderate gap total.
- Deduped `v3` has `7` moderate gaps total.
- Of the 8 terms previously marked `resolved`, 6 regressed below the `0.5` ratio threshold again.
- Only `도파민` and `갯수` remain resolved.
- `ㅡㅡ` is still unresolved.
- Source coverage did not change: `99.52%` before and after.

This means the earlier improvement was mostly carried by the now-removed duplicate/near-duplicate patch rows, not by a durable set of unique training examples.

## GAP Comparison

| Term | `v2` ratio | full-patch `v3` ratio | deduped `v3` ratio | `v2` count | full-patch `v3` count | deduped `v3` count | Status after dedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TC` | `0.1048` | `0.5722` | `0.1132` | `463` | `2,621` | `502` | regressed |
| `밀빵` | `0.2289` | `0.6852` | `0.2430` | `1,087` | `3,374` | `1,158` | regressed |
| `케어` | `0.2564` | `0.7242` | `0.2945` | `2,608` | `7,637` | `3,006` | regressed |
| `쩜오` | `0.3979` | `0.7055` | `0.4147` | `2,351` | `4,322` | `2,459` | regressed |
| `하이퍼` | `0.3486` | `0.7420` | `0.4671` | `87` | `192` | `117` | regressed |
| `도파민` | `0.4992` | `0.6783` | `0.5476` | `3,478` | `4,900` | `3,829` | still resolved |
| `갯수` | `0.4588` | `0.8776` | `0.5209` | `2,000` | `3,967` | `2,279` | still resolved |
| `선릉` | `0.3980` | `0.5798` | `0.4465` | `143` | `216` | `161` | regressed |
| `ㅡㅡ` | `0.4915` | `0.4767` | `0.4923` | `3,885` | `3,907` | `3,906` | still unresolved |

Key takeaways:

- `TC`, `밀빵`, `케어`, and `쩜오` are effectively back at `v2`-like coverage.
- `도파민` and `갯수` survive dedup, but with much smaller gains than the full-patch `v3`.
- `선릉` falls back below threshold, so the location-specific recovery did not survive dedup either.

## Structural Metrics

No structural gaps were flagged by `refinement_loop.py` in the deduped rerun. The lexical GAP repair regressed, but tone/text-type/sentence-shape stayed within the script's thresholds.

Important caveat: phase 1b uses `random.sample(..., 500)` with no fixed seed, so exact percentages vary between runs. The direction is still clear enough for a recheck.

| Metric | source (dedup rerun) | full-patch `v3` train | deduped `v3` train | Read |
| --- | ---: | ---: | ---: | --- |
| tone `혼합` | `60.0%` | `66.6%` | `63.4%` | dedup slightly closer to source |
| tone `반말` | `21.4%` | `23.6%` | `27.4%` | dedup slightly farther from source |
| tone `존댓말` | `18.6%` | `9.8%` | `9.2%` | still underrepresented |
| type `정보형` | `92.6%` | `85.4%` | `87.0%` | dedup slightly closer to source |
| type `질문형` | `5.8%` | `12.8%` | `11.6%` | still overrepresented, but improved |
| type `후기형` | `1.6%` | `1.8%` | `1.4%` | effectively stable |
| avg sentences / text | `3.99` | `2.88` | `3.33` | dedup improves shape alignment |
| avg words / sentence | `4.18` | `5.05` | `4.65` | dedup improves shape alignment |

Interpretation:

- Dedup helps sentence-shape alignment a bit.
- Dedup does not fix the persistent `존댓말` deficit.
- The real failure mode is domain term density, not broad tone or sentence structure.

## Coverage / Volume Effect

Coverage stayed flat because the underlying post coverage was already high in `v2`.

- source coverage: `99.52%` in `v2`, full-patch `v3`, and deduped `v3`
- train source IDs:
  - `v2`: `11,232`
  - full-patch `v3`: `18,133`
  - deduped `v3`: `12,302`
- training texts seen by the diagnostic:
  - `v2`: `138,255`
  - full-patch `v3`: `153,049`
  - deduped `v3`: `139,524`

So the deduped patch preserved only a small fraction of the exposure boost that had repaired the business/location vocabulary.

## Bottom Line

The 8-gap claim does not hold for the deduped `v3`.

- `6/8` previously resolved terms reopened.
- `2/8` stayed resolved (`도파민`, `갯수`).
- `ㅡㅡ` remained unresolved throughout.
- Structure stayed acceptable, so the regression is specifically about forum-native domain vocabulary coverage.

If this corpus is meant to preserve the original GAP repair, the dedup strategy is too aggressive for the targeted business/promo-derived comments and needs a more selective uniqueness rule.
