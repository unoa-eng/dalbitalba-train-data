# Final Data Fidelity Audit

- Repo: `/Users/unoa/projects/dalbitalba-train-data`
- Source: `/Users/unoa/Downloads/crawled-data-v2`
- Audit seed: `20260428`
- Evidence JSON: `runs/final-data-fidelity-audit.json`
- Sampling note: criteria 1-3 sampled from raw threads not present in `val_set.v2.jsonl`, because criterion 5 separately audits train/val isolation.

## Criterion 1: 50 random source posts -> `cpt_corpus.v3.jsonl`

`FAIL`

- Sample result: `47/50` exact matches.
- Expected rule: `text == "<title>\n<content>"` by `source_id`.
- Sample failures:
  - `1192555`: missing in `v3` (`.` / `감사합니다 !`)
  - `1201353`: missing in `v3` (`사라오늘돟떼초?` / `??`)
  - `1193683`: present but normalized mismatch (`…` became `...`)
- Corpus-wide on non-val raw posts:
  - `10,051 / 10,722` exact matches
  - `419` present-but-mismatched rows
  - `252` missing rows
- Failure mode is not catastrophic, but it is real: the shipped train corpus is not a lossless title+body mirror of the raw source.

## Criterion 2: 50 random source comments -> `cpt_corpus.v3.jsonl`

`FAIL`

- Sample result: `0/50` matched the stripped comment body.
- Failure breakdown:
  - `32/50` were present only as raw tagged comments, not stripped bodies.
  - `18/50` were absent from the thread in `v3`.
- Representative failures:
  - Raw: `[1-1] 어떻게 갚으셨오요?`
    - Expected in `v3`: `어떻게 갚으셨오요?`
    - Actual in `v3`: `[1-1] 어떻게 갚으셨오요?`
  - Raw: `[1-3] 비회원 [1-2]  차은우 사태보고 ...`
    - Expected in `v3`: parent refs removed
    - Actual in `v3`: full raw string with reply tags still attached
  - Raw promo comments such as `1193747` and `1200074` were not found at all in the thread.
- Corpus-wide on non-val raw comments:
  - `10 / 53,037` matched the cleaned body
  - `33,988 / 53,037` matched only the raw tagged form
  - `19,039 / 53,037` were not found
- Additional corpus signal:
  - `36,386 / 37,584` comment rows in `cpt_corpus.v3.jsonl` start with a reply tag like `[2]` or `[3-1]` (`96.8%`)
- This is the clearest fidelity failure in the audit. The thread-aware cleaner is not what the final CPT corpus actually contains.

## Criterion 3: `cpt_context_stream.jsonl` context reconstruction

`PASS`

- Sample result: `10/10` random `context_comment` rows matched source thread structure exactly.
- Verified fields:
  - `title` -> raw post title
  - `원글` -> raw post body excerpt
  - `댓글` or `부모댓글` + `답글` -> correct cleaned source comments
  - `comment_key` and `parent_comment_key` -> correct source positions
- Representative passing rows:
  - `1197102` / `comment_key=6` / `root`
  - `1199866` / `comment_key=2` / `root`
  - `1193841` / `comment_key=5` / `root`

## Criterion 4: promo contamination still present in `cpt_corpus.v3.jsonl`

`FAIL`

- Contaminated rows by heuristic union: `2,708 / 48,054` (`5.64%`)
- Breakdown:
  - phone-number rows: `0`
  - kakao/open-chat ID or link rows: `488`
  - recruiter-template rows: `2,658`
- Important nuance:
  - direct phone numbers appear to have been scrubbed to `[전화번호]`
  - that does **not** mean the promo content is gone; many recruiter templates still remain as recognizable ads
- Representative contamination:
  - `1186948`: `카카오톡-keroro25`, `문의 환영`
  - `1199993`: `*카톡*ooaa119`, `*텔레* mmamlb`, `*라인* bagel98`
  - `1194753`: `실장`, `출근부터 퇴근까지 케어`, `[전화번호]`
  - `1188013`: `T.C`, `지명비`, `편하게 연락 주세요`
- This is high enough to keep teaching recruiter/operator style, even with some PII placeholders already applied.

## Criterion 5: `val_set.v2.jsonl` overlap with `cpt_corpus.v3.jsonl`

`PASS`

- exact text overlap: `0`
- `source_id` overlap: `0`
- thread-level overlap: `0`
- exact tuple overlap (`source_id`, `source_field`, `text`): `0`

## Verdict

Overall verdict: `FAIL`

- The context-stream patch is structurally correct.
- The validation split is clean.
- The final CPT corpus still fails the main data-fidelity bar because comment rows are overwhelmingly stored in raw tagged form instead of stripped bodies, and recruiter/promo contamination remains material.
