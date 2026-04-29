# Re-Audit: Data Integrity

- Date: `2026-04-28`
- Worker: `worker-1`
- Repo: `/Users/unoa/projects/dalbitalba-train-data`
- Raw source: `/Users/unoa/Downloads/crawled-data-v2`
- Primary CPT input: `cpt_context_stream.jsonl`
- Validation set: `val_set.v2.jsonl`
- Read first: `AUTOPILOT_LOOP.md`
- Random sampling seed used for this audit: `20260428`

## Verdict

Overall verdict: `FAIL`

The train/val isolation check passes and sampled `context_comment` structure is correct against raw source, but the corpus still fails hard on two core integrity requirements:

- `comment` rows still contain leading reply tags at scale.
- promo / recruiter contamination is still present at scale, including explicit contact handles and phone placeholders.

## Criterion Results

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Count `cpt_context_stream.jsonl` rows by kind | `PASS` | `43,635` total rows = `21,558 comment`, `12,014 context_comment`, `10,063 post`. |
| 2 | Verify `0` `comment` rows still have `[N]` tags at start | `FAIL` | `2,087` `comment` rows still match the leading-tag pattern. Examples: `cpt_context_stream.jsonl:67`, `:204`, `:11680`. |
| 3 | Verify `0` rows match promo heuristics | `FAIL` | Floor: `871` rows have explicit contact markers (`[전화번호]`, Kakao/Line/Tele/open.kakao style handles). Adding template-only recruiter copy raises the total to `1,233` rows. Examples: `:94`, `:268`, `:375`, `:561`, `:821`, `:31400`. |
| 4 | Sample `30` random rows and manually inspect for truncation / encoding / nonsense | `FAIL` | `27/30` looked structurally natural and showed no mojibake, replacement chars, or NULs. `3/30` had material quality failures: recruiter pitch at `:7693`, tag leak at `:11680`, full recruiter ad thread at `:31400`. |
| 5 | Verify `val_set.v2.jsonl` has `0` overlap with `cpt_context_stream.jsonl` on text / source_id / thread | `PASS` | Exact text overlap `0`; exact `source_id` overlap `0`; canonical thread overlap `0` where thread ID is `source_id.split(':', 1)[0]`. |
| 6 | Sample `10` `context_comment` rows and verify thread structure against source | `PASS` | Reconstructed context from raw crawl matched serialized `context_comment` text `10/10` times, including both root and reply cases. |
| 7 | Check duplicate rate within `cpt_context_stream.jsonl` | `FAIL` | Exact-text duplicate rows: `208 / 43,635` (`0.48%`) across `65` duplicate groups. Exact row duplicates: `58 / 43,635` (`0.13%`) across `29` groups. The biggest duplicate clusters are junk/moderation/ad templates rather than meaningful conversational repeats. |

## Criterion 1: Row Count by Kind

| kind | rows |
| --- | ---: |
| `comment` | `21,558` |
| `context_comment` | `12,014` |
| `post` | `10,063` |
| `total` | `43,635` |

## Criterion 2: Leading `[N]` Tag Leakage in `comment` Rows

Result: `FAIL`

- Matching `comment` rows: `2,087`
- Regex used: start-of-text reply tag with optional short author prefix, matching rows like `비회원 [2-1] ...`

Examples:

- `cpt_context_stream.jsonl:67` — `비회원 [2-1] 괜히 욕심내서 성형했다가...`
- `cpt_context_stream.jsonl:204` — `비회원 [6-1] 이래서 누구와 일하는지가 중요합니다 ㅠㅠ 연락주세요`
- `cpt_context_stream.jsonl:11680` — `비회원 [1-1] ㄹㅌ콜 계속들어오고있어요~`

This is not an edge case. Tag stripping is still failing across thousands of plain `comment` rows.

## Criterion 3: Promo / Recruiter Contamination

Result: `FAIL`

I separated the result into a hard-contact floor and a softer template layer.

### Hard-contact floor

Rows with explicit contact markers: `871`

- `567` `comment`
- `297` `context_comment`
- `7` `post`

Contact markers counted here:

- numeric phone numbers
- `[전화번호]`
- Kakao / Katalk IDs
- Line IDs
- Telegram / open.kakao / open chat style handles

### Template-only recruiter copy

Additional rows with recruiter template phrasing but no explicit handle: `362`

- `252` `comment`
- `101` `context_comment`
- `9` `post`

Examples of template phrases matched:

- `연락주세요`
- `모시겠습니다`
- `당일 정산`, `당일 지급`, `당일 수금`
- `24시간 운영`
- `팁길만 걸으세요`
- `도와드리겠습니다`

### Combined heuristic count

Combined promo-heuristic matches: `1,233`

Examples:

- `cpt_context_stream.jsonl:94` — explicit phone placeholder + solicitation
- `cpt_context_stream.jsonl:268` — `[전화번호]`, `카카오 kai1922`, recruiter copy
- `cpt_context_stream.jsonl:375` — `[전화번호]`, `카톡 gana0505`, `24시간 운영`
- `cpt_context_stream.jsonl:561` — repeated `총사장` pitch with two Kakao IDs and phone placeholders
- `cpt_context_stream.jsonl:821` — context row with direct recruiter pitch ending in `연락주세요`
- `cpt_context_stream.jsonl:31400` — full recruiter ad embedded inside a `context_comment` parent comment

This corpus is still materially contaminated by recruiter-style language.

## Criterion 4: Manual Inspection of 30 Random Rows

Result: `FAIL`

Manual review summary:

- `27/30` sampled rows looked like normal forum-style Korean text.
- No sampled rows showed UTF-8 corruption, `�` replacement characters, or NUL bytes.
- No sampled rows showed obvious serialization truncation beyond the intended `context_comment` excerpt format.
- `3/30` rows still had material integrity problems.

Flagged samples:

- `cpt_context_stream.jsonl:7693`
  `comment` row containing clear recruiter copy: `A부터 Z까지 하나씩 친절하게 도와드리겠습니다 / 기회 한번 주시면...`
- `cpt_context_stream.jsonl:11680`
  `comment` row still starts with `비회원 [1-1] ...`
- `cpt_context_stream.jsonl:31400`
  `context_comment` row whose parent comment is a long recruiter ad with `[전화번호]`, Kakao handle, `총사장`, `1등팀`, `24시`, `택시비 지원`

Interpretation:

- encoding quality is acceptable in the sample
- thread serialization quality is acceptable in the sample
- contamination cleanup is not acceptable in the sample

## Criterion 5: Train / Val Overlap

Result: `PASS`

Overlap checks:

- exact text overlap: `0`
- exact `source_id` overlap: `0`
- canonical thread overlap: `0`

Thread comparison used:

- `thread_id = str(source_id).split(':', 1)[0]`

On this criterion, the current `cpt_context_stream.jsonl` is cleanly separated from `val_set.v2.jsonl`.

## Criterion 6: `context_comment` Structure Audit

Result: `PASS`

Method:

- sampled `10` random `context_comment` rows
- loaded the raw thread by `source_id` from `/Users/unoa/Downloads/crawled-data-v2`
- re-parsed comment keys from raw comments
- rebuilt the expected serialized context using the same title / post excerpt / parent / reply structure
- compared reconstructed text against the stored `context_comment` text

Result:

- exact reconstruction match: `10 / 10`

Representative matches:

- `cpt_context_stream.jsonl:3547` / `source_id=1188083` / root comment
- `cpt_context_stream.jsonl:20023` / `source_id=1197459` / reply comment
- `cpt_context_stream.jsonl:11500` / `source_id=1189412` / reply comment with raw `비회원 [1-1]` source text correctly normalized into the context row
- `cpt_context_stream.jsonl:31921` / `source_id=1193480` / reply comment

Important nuance:

- the structure is correct
- the content filtering is not

In other words, `context_comment` rows are being reconstructed faithfully from raw threads, but that also means raw recruiter comments are being faithfully preserved when they are not filtered out upstream.

## Criterion 7: Duplicate Rate

Result: `FAIL`

### Exact text duplicates

- duplicate groups: `65`
- rows participating in exact-text duplicates: `208 / 43,635` (`0.48%`)
- max duplicate count for a single text: `32`

Top duplicate texts:

- `신고에 의해 블라인드 처리 되었습니다..` × `32`
- `광고 권한이 없습니다.` × `10`
- `하퍼 도파민 / 쩜오 썸데이 마동석입니다 ... [전화번호] ... 연락주세요` × `10`
- repeated New Year recruiter-spam lines ending in `팁길만 걸으세요~` × `3` to `4`

### Exact row duplicates

- duplicate groups: `29`
- rows participating in full row duplicates: `58 / 43,635` (`0.13%`)
- max duplicate count for a single full row: `2`

Interpretation:

- the overall duplicate rate is not huge numerically
- the duplicate mass is concentrated in moderation boilerplate and recruiter spam
- that is a data-quality failure, not harmless natural repetition

## Additional Observations

- Rows with Unicode replacement char `�`: `0`
- Rows containing NUL byte: `0`
- `context_comment` reconstruction quality is good enough to keep
- contamination filtering and tag stripping are not good enough to ship

## Bottom Line

`cpt_context_stream.jsonl` is **not ready** to be treated as a clean final CPT stream.

What is fixed:

- train/val leakage on text / source / thread is clean
- `context_comment` structure is correctly serialized against raw source

What is still broken:

- `2,087` plain `comment` rows still leak leading `[N]` reply tags
- at least `871` rows still carry explicit contact markers, and `1,233` rows match contact-or-template promo heuristics
- duplicate clusters are still dominated by moderator stubs and recruiter spam
- random manual inspection still surfaces obvious contamination immediately
