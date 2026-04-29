# v3 Structured Build Report

- Date: `2026-04-29`
- Repo: `/Users/unoa/projects/dalbitalba-train-data`
- Raw crawl: `/Users/unoa/Downloads/crawled-data-v2`
- Worker scope: build missing v3 artifacts, evaluate structured CPT against live `cpt_context_stream.jsonl`, decide whether to promote `budget30_v2.env`, and run final local verification.

## Code changes made during integration

- `scripts/build_structured_cpt_v3.py`
  - now emits a merged CPT file: `v3-data/cpt_structured_v3.jsonl`
  - adds verifier-compatible metadata fields: `source_field`, `length_bucket`
  - hardens contact/ad filtering for dotted phone numbers and broader contact ID patterns
  - drops minor/sexual proximity rows at build time
- `scripts/build_5task_sft_v3.py`
  - applies the same stricter ad/contact and minor/sexual filtering
- `scripts/local_verification_loop.py`
  - now respects `TRAIN_CPT_JSONL` / `INPUT_JSONL` when resolving the CPT file, so sourced recipes and verifier runs audit the same corpus

## Built artifacts

### Structured CPT

- `v3-data/cpt_structured_individual.jsonl`: `45,230` rows
- `v3-data/cpt_structured_threads.jsonl`: `8,450` rows
- `v3-data/cpt_structured_v3.jsonl`: `53,680` rows

Builder summary (`v3-data/cpt_v3_summary.json`):

- raw posts scanned: `11,288`
- kept post rows: `10,248`
- raw comments scanned: `51,495`
- kept comment rows: `34,982`
- kept thread rows: `8,450`
- dropped promo comments: `10,107`
- dropped blocked comments: `1,403`
- dropped short comments: `5,003`
- dropped short posts: `1,032`
- dropped promo posts: `8`

Sample inspection:

- confirmed `<|post|>` rows are present
- confirmed `<|comment depth=...|>` rows are present
- confirmed `<|thread|>` rows are present

Merged CPT structure coverage:

- `kind=post`: `10,248`
- `kind=comment`: `34,982`
- `kind=thread`: `8,450`
- rows containing `<|post|>`: `18,698`
- rows containing `<|comment depth=...|>`: `43,432`
- rows containing `<|thread|>`: `8,450`

### 5-task SFT

- `v3-data/sft_5task_v3.jsonl`: `58,476` rows

Builder summary (`v3-data/sft_v3_summary.json`):

- `T1_title_to_body`: `9,945` (`17.01%`)
- `T2_root_comment`: `29,647` (`50.70%`)
- `T3_reply_comment`: `10,562` (`18.06%`)
- `T4_topic_to_post`: `3,383` (`5.79%`)
- `T5_continue`: `4,939` (`8.45%`)
- skipped posts: `16`

Note:

- `T2 + T3` comment-generation share = `68.76%`

## Comparison vs current CPT

Current live CPT:

- `cpt_context_stream.jsonl`: `42,137` rows
- kind mix: `10,043 post`, `20,477 comment`, `11,617 context_comment`

Candidate structured CPT:

- `v3-data/cpt_structured_v3.jsonl`: `53,680` rows
- kind mix: `10,248 post`, `34,982 comment`, `8,450 thread`

Delta:

- row count: `+11,543` (`+27.39%`)
- unique source IDs: current `11,383` vs v3 `10,248`
- structure tokens: current `0`, v3 pervasive across post/comment/thread rows

Selected domain-term coverage (`row-hit %`):

| term | current | v3 merged |
| --- | ---: | ---: |
| 하퍼 | 6.66 | 5.64 |
| 초이스 | 4.16 | 4.34 |
| 수위 | 3.53 | 3.23 |
| 업진 | 3.48 | 3.13 |
| 마담 | 1.31 | 1.39 |
| 텐카 | 1.97 | 1.83 |
| 상띠 | 1.87 | 1.69 |
| 손놈 | 1.72 | 1.74 |
| 진상 | 0.96 | 1.34 |
| 부장 | 0.85 | 1.24 |
| 실장 | 0.75 | 1.05 |
| 쩜오 | 1.36 | 1.01 |
| 밀빵 | 0.82 | 0.73 |

Takeaway:

- core slang coverage is largely preserved
- however the merged structured corpus is noticeably denser on a few moderation/adjacent terms (`진상`, `부장`, `실장`) and materially longer overall

## Evaluation results

### Merged structured CPT

Command:

```bash
python3 scripts/eval_domain_metrics_v3.py \
  --reference cpt_context_stream.jsonl \
  --generated v3-data/cpt_structured_v3.jsonl \
  --output runs/v3-structured-eval.json
```

Result:

- Bigram JSD: `0.249792` (`FAIL`, target `< 0.08`)
- Length KL: `0.042703` (`FAIL`, target `< 0.01`)
- Style match: `4/9`
- Overall verdict: `NEEDS_IMPROVEMENT`

Main style deviations:

- 초성율: `23.2 -> 37.1`
- 웃음 마커: `9.6 -> 18.2`
- 울음 마커: `6.8 -> 12.0`
- 평균 길이: `82.8 -> 136.4`
- 중간 길이: `52 -> 70`

### Individual-only structured CPT

Command:

```bash
python3 scripts/eval_domain_metrics_v3.py \
  --reference cpt_context_stream.jsonl \
  --generated v3-data/cpt_structured_individual.jsonl \
  --output runs/v3-structured-individual-eval.json
```

Result:

- Bigram JSD: `0.231791` (`FAIL`)
- Length KL: `0.037107` (`FAIL`)
- Style match: `5/9`
- Overall verdict: `NEEDS_IMPROVEMENT`

Conclusion:

- neither v3 CPT variant clears the current similarity gate
- the thread-augmented merged file is directionally closer to the design intent, but it is farther from the current live CPT distribution than the budget30 path should accept today

## Budget and verification

### Candidate v3 CPT verification

Command:

```bash
set -a
source recipes/budget30_v2.env
export TRAIN_CPT_JSONL=v3-data/cpt_structured_v3.jsonl
export INPUT_JSONL=v3-data/cpt_structured_v3.jsonl
set +a
python3 scripts/local_verification_loop.py --profile budget30
```

Result: `FAIL`

- report: `runs/local-verification-20260429-034609/report.md`
- CPT rows: `53,680`
- estimated CPT train time: `17.18h`
- estimated CPT cost: `$13.57`

Verifier blockers:

- `phone_like=4`
- `minor_sexual_proximity=1`

Notes on those residual hits:

- the large ad/contact leak was fixed (`1851 -> 4`)
- the remaining phone-like hits are down to one institutional phone number plus false positives from numeric formatting/thread duplication
- the remaining minor/sexual hit is a regex false positive on `중딩` + `성질내`
- despite those likely false positives, the verifier still returns `FAIL`, so the candidate is not launch-ready under the current gate

### Active budget30 recipe verification

Command:

```bash
set -a
source recipes/budget30_v2.env
set +a
python3 scripts/local_verification_loop.py --profile budget30
```

Result: `PASS`

- report: `runs/local-verification-20260429-034358/report.md`
- active CPT rows: `42,137`
- estimated CPT train time: `13.48h`
- estimated CPT cost: `$10.65`

## Recipe promotion decision

`recipes/budget30_v2.env` was **not** updated.

Reason:

- the structured v3 CPT does not pass the requested quality bar
- both automated eval runs return `NEEDS_IMPROVEMENT`
- the candidate verifier run still returns `FAIL`

Current `TRAIN_CPT_JSONL` therefore remains:

```bash
TRAIN_CPT_JSONL=cpt_context_stream.jsonl
```

## Final status

- v3 structured CPT build: `DONE`
- v3 5-task SFT build: `DONE`
- v3 CPT comparison and evaluation: `DONE`
- budget recalculation for candidate CPT: `DONE`
- budget30 final verification of active recipe: `DONE`
- promotion into `budget30_v2.env`: `DEFERRED`

Recommended next step:

- if promotion is still desired, the next iteration should target similarity drift directly:
  - reduce thread-row share or sample threads instead of full inclusion
  - normalize residual verifier false positives or refine the verifier regexes
  - re-balance the structured corpus to pull average/median length back toward the live CPT stream
