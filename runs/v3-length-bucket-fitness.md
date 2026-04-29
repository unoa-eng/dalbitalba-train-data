# V3 Length Bucket Fitness

- Generated at: `2026-04-27T08:02:58Z`
- Inputs: `cpt_corpus.v2.jsonl`, `cpt_patch_gap_repair.jsonl`, `cpt_corpus.v3.jsonl`
- Verdict: `ready_with_warning`

## Key Findings

- Pre-fix bug: all `14,794` repair rows carried `length_bucket="패치"`; both the patch file and merged v3 corpus are now normalized to canonical buckets.
- Patch mix is highly short-form: `xs 43.8%`, `sm 53.5%`, `md 2.2%`, `lg 0.4%`, `xl ~0.0%`, `xxl 0.0%` by rows.
- Merged v3 row mix after fix: `xs 22.4%`, `sm 48.6%`, `md 18.8%`, `lg 6.8%`, `xl 3.2%`, `xxl 0.1%`.
- Overall target alignment improves versus v2: TV distance `0.1808 -> 0.1441`.
- Long-form share remains low: rows `>=100 chars` moved `13.2% -> 10.2%` vs phase1 target `21.0%`.
- Token impact is smaller than row impact: patch adds `24.0%` of rows but only `10.8%` of characters.

## Readiness

- Warning: Rows >=100 chars remain underrepresented versus target (v3 long-share rows 10.15% vs target 21.00%).
- Warning: Repair patch is overwhelmingly short-form (97.38% xs+sm by rows); this is acceptable for gap repair but will skew row-level sampling if additional short patches are appended without rebalance.
- Note: Canonical length_bucket labels are now consistent with text length in patch and merged v3 artifacts.
- Note: Patch contributes 24.00% of v3 rows but only 10.84% of v3 characters, so token-weighted training impact is materially smaller than row share.

## Recommendation

- RunPod CPT on v3 is acceptable after the bucket-label fix, but treat the corpus as short-form heavy.
- If phase6 shows terse generations or weak long-answer retention, rebalance before the next paid cycle instead of appending more short GAP patches.
