# Dalbit Training Design Completeness Audit

Date: `2026-04-27`

Target question: is the current pipeline actually designed to produce forum-native posts/comments that are hard to distinguish from the raw crawl?

Scope checked:

- `AUTOPILOT_LOOP.md`
- `train_cpt.py`
- `train_sft.py`
- `scripts/build_thread_aware_datasets.py`
- `scripts/phase6_eval.py`
- `scripts/refinement_loop.py`
- `scripts/generate_samples.py`
- `scripts/remove_val_train_leak.py`
- Live artifacts: `cpt_corpus.v3.jsonl`, `cpt_corpus.v2.jsonl`, `cpt_patch_gap_repair_dedup.jsonl`, `val_set.v2.jsonl`, `sft_pairs.v2.jsonl`, `sft_pairs_v2.jsonl`
- Raw source: `/Users/unoa/Downloads/crawled-data-v2`

Important note before the audit:

- The repo still contains several stale reports that describe the pre-dedup `v3` corpus as `61,650` rows.
- The live artifact on disk is different:
  - `cpt_corpus.v2.jsonl`: `46,856`
  - `cpt_patch_gap_repair_dedup.jsonl`: `1,269`
  - `cpt_corpus.v3.jsonl`: `48,125`
- This audit uses the live files, not the stale `61,650`-row reports.

## Verdict

Partially, but not enough to call it forum-native by design.

The current pipeline can learn domain slang and short-form surface style, but it is still weak on the exact things the operating doc says matter most:

- comment context and reply flow
- kind-stratified evaluation
- title-aware post generation
- thread-aware leakage control
- ad/operator-noise avoidance

The biggest design problem is not `seq_len`; it is that the active training path still treats most text as isolated rows, while the stated target requires thread-conditioned realism.

## 1. AUTOPILOT goals vs current v3 pipeline

I treated the explicit `목표` bullets in `AUTOPILOT_LOOP.md:5-12` and the directly goal-defining bullets in `현재 핵심 원칙` (`AUTOPILOT_LOOP.md:25-28`) as the operative success criteria.

| Stated goal | Status in current pipeline | Evidence |
| --- | --- | --- |
| Generated text should be hard to distinguish from real raw posts/comments | `PARTIAL` | The active training path still uses mostly standalone text rows for CPT (`train_cpt.py:147-175`) and minimal `post + comment` pairs for SFT (`train_sft.py:198-211`). That is enough for style transfer, but not enough for reply-flow realism. |
| Blind eval should make human/AI discrimination hard | `PARTIAL` | The deterministic evaluator is not a blind eval at all; it is flat text similarity (`scripts/phase6_eval.py:3-28`). There is a separate blind-eval path in the doc, but the checked scripts here do not implement it. Also, blind-sample prompts come from `sft_pairs_v2.jsonl` (`scripts/generate_samples.py:25-27`), while the active SFT trainer consumes `sft_pairs.v2.jsonl` (`train_sft.py:66-68`), so prompt format and training format are misaligned. |
| Raw-vs-generated distributions should match for length, punctuation, 초성, 은어, 감정톤, 댓글 구조 | `PARTIAL` | Current eval covers length, domain terms, and coarse tone, but not punctuation ratios, 초성 rates, emoji/emotive markers, or reply structure (`scripts/phase6_eval.py:12-28`, `scripts/refinement_loop.py:322-492`). Live data also shows a real mismatch: source comments are much longer than train comments (`source p90=256 / p95=349 chars` vs `train p90=96 / p95=146`). |
| Ad/operator noise should not be learned as human style | `PARTIAL/WEAK` | The thread-aware builder tries to filter promo comments (`scripts/build_thread_aware_datasets.py:204-237`), but the active CPT corpus still contains a meaningful amount of promo-ish/operator language. Using the same keyword family as the repo’s filters, `cpt_corpus.v3.jsonl` still has `2,963 / 48,125` rows (`6.16%`) that look promo-like. Phone numbers are mostly scrubbed, but operator tone remains. |
| It is not enough to resemble posts; comment context and reply flow must also resemble raw | `NO` | The active SFT trainer uses only `post + comment` pairs (`train_sft.py:198-211`). Parent-comment context exists in `scripts/build_thread_aware_datasets.py:242-283`, but that output is not what `train_sft.py` or `chain_train.sh` train on. |
| Match short-turn dialogue, 질문/반문, 생략부호, 감정기호, not just slang terms | `PARTIAL` | `phase6_eval.py` has no direct punctuation or emotive-marker metric. `refinement_loop.py` has only a coarse `질문형/후기형/정보형` classifier and tone regexes (`scripts/refinement_loop.py:345-391`). It does not directly score `?`, `!`, `...`, `ㅋㅋ`, `ㅎㅎ`, `ㅠ`, `ㅜ`, or reply depth. |
| Eval must be stratified by `kind` (post/comment) | `NO` | `phase6_eval.py` loads only flat `text` (`scripts/phase6_eval.py:47-61`), and `refinement_loop.py` mixes posts/comments/SFT/val into one training blob (`scripts/refinement_loop.py:128-190`). This directly conflicts with `AUTOPILOT_LOOP.md:28`. |

## 2. Do `phase6_eval.py` and `refinement_loop.py` measure what matters?

Short answer: no. They are useful coarse diagnostics, but they are not acceptance tests for the stated goal.

### `scripts/phase6_eval.py`

What it does measure well:

- character bigram similarity
- flat text length distribution
- digit/English density
- sample-level presence of selected domain keywords
- coarse tone balance
- optional MAUVE

What it does not measure:

- `post` vs `comment` separately
- thread coherence
- root comment vs reply comment behavior
- title fidelity
- punctuation ratios (`?`, `!`, `...`)
- 초성/emoji/emotive marker rates as first-class metrics
- ad/operator-noise rate

Additional implementation issue:

- The docstring says `length_kl` is a token-count histogram, but the code uses `len(text)` character length (`scripts/phase6_eval.py:97-121`).

Conclusion:

- Good as a lightweight flat-text sanity check.
- Not sufficient to decide whether comments feel like real in-thread replies.

### `scripts/refinement_loop.py`

What it does measure:

- domain-term frequency gaps
- coarse tone / text-type / sentence-count structure
- coarse coverage by source post id

Why it is not a faithful design evaluator:

1. It is hardcoded to old files.
   - `TRAIN_FILES` still points to `cpt_corpus.v2.jsonl`, `sft_pairs.v2.jsonl`, and `val_set.v2.jsonl` (`scripts/refinement_loop.py:29-33`).

2. It contaminates “train” with validation data.
   - `load_train_concat()` explicitly reads `val_set.v2.jsonl` and appends it into the text blob used for diagnostics (`scripts/refinement_loop.py:172-190`).
   - That means the diagnostic is not a pure train-vs-source comparison.

3. It mixes incompatible modalities.
   - It merges CPT text rows, SFT post fields, SFT comment fields, and val rows into one concatenated corpus (`scripts/refinement_loop.py:128-190`).
   - That inflates lexical coverage and hides formatting differences.

4. Its “coverage” is really source-post coverage, not comment/thread coverage.
   - Raw comments inherit the post id as `sid` in `iter_source_texts()` (`scripts/refinement_loop.py:99-108`).
   - `phase1c_coverage()` then reports overlap on those ids as “Source posts … Covered in train” (`scripts/refinement_loop.py:435-489`).

5. It is not kind-stratified.
   - The structure sampler draws mixed texts only (`scripts/refinement_loop.py:322-398`).

Conclusion:

- Good as a gap-mining heuristic.
- Not trustworthy as the main design-completeness signal.

## 3. Is the CPT format correct? Should comments include context, or is raw text enough?

Current CPT format:

- `train_cpt.py` loads only `text` and throws away everything else (`train_cpt.py:147-163`).
- Tokenization is “one row = one standalone continuation text” (`train_cpt.py:166-175`).
- The live `cpt_corpus.v3.jsonl` rows contain `text`, `kind`, `source_id`, `source_field`, `length_bucket`, but no title in the post rows and no parent context in comment rows.

For the stated goal, raw text alone is not enough.

Why:

- A raw standalone comment row can teach style, slang, and surface rhythm.
- It cannot teach why a reply is sharp, deferential, sarcastic, corrective, or helpful in that exact place in a thread.
- This is especially true when the surface form includes tags like `[1-3]` but the parent text is absent; the tag remains as style noise, but the actual discourse relation is gone.

What “correct enough” would look like:

1. Keep standalone raw rows.
   - They are still useful for unconditional style modeling.

2. Add a context-serialized CPT stream for comments.
   - Example:
   - `제목: ...`
   - `원글: ...`
   - `부모댓글: ...`
   - `답글: ...`

3. Keep post/comment identity explicit.
   - Either separate corpora, or a visible kind marker.

Practical recommendation:

- Keep roughly a mixed CPT recipe:
  - standalone posts/comments for style
  - thread-serialized comment examples for discourse realism

Without that, the active design remains “forum-style flat text,” not “forum-native replies.”

## 4. Is `seq_len=1024` appropriate for the real text lengths?

Yes as a truncation ceiling. No as the main issue to optimize.

Measured on the live raw source and live `v3`:

- Source contentful post body lengths:
  - mean `67.92`
  - p95 `219`
  - p99 `412`
  - max `1846`
- Source post length with title included:
  - mean `82.91`
  - p95 `238`
  - p99 `427`
  - max `1874`
- Source comment lengths:
  - mean `83.43`
  - p95 `349`
  - p99 `491`
  - max `1209`
- Current `cpt_corpus.v3.jsonl`:
  - `66.49%` of rows are `<50` chars
  - `24.77%` are `<20` chars
  - only `10 / 48,125` rows (`0.02%`) are `>=1024` chars
- Source overall:
  - only `20 / 67,129` non-empty raw texts (`0.03%`) are `>=1024` chars

Conclusion:

- `1024` is already more than enough for almost every real text row in the current data.
- Raising it to `1536` or `2048` will not solve realism.
- The real design problem is that CPT currently trains on huge numbers of very short isolated rows without packing or thread serialization.

Implication:

- If you keep the current row format, `1024` is fine.
- If you move to thread-aware serialized rows, `1024` is still probably fine for `title + post excerpt + parent + reply`.
- If you later pack multiple short rows or add longer multi-turn context, then revisiting the ceiling becomes worth it.

## 5. Should posts include title in CPT? Current pipeline uses content only.

Yes, titles should be included.

Measured on the raw source:

- `11,280 / 11,280` contentful posts have a non-empty title (`100%`)
- title mean length: `13.98` chars
- `97.49%` of titles are not a literal substring of the body
- `96.39%` of titles contain at least one token not present in the body

That means the title is not redundant decoration. It usually carries topic framing the body does not repeat verbatim.

Current behavior:

- The thread-aware builder stores `title` on CPT post rows, but still writes only `content` into `text` (`scripts/build_thread_aware_datasets.py:137-149`).
- The active CPT trainer then ignores even the stored `title` field and consumes only `text` (`train_cpt.py:147-163`).

Recommendation:

- Serialize post CPT as `제목: {title}\n본문: {content}`.
- If you want the model to also handle titleless body-only continuations, include a smaller secondary body-only view, not the other way around.

## 6. Are comment reference tags like `[1]`, `[2-1]` preserved or stripped? Should they be?

What the live files do:

- Raw source comments tagged at start: `100%`
- `val_set.v2.jsonl` tagged comments: `100%`
- `cpt_corpus.v3.jsonl` tagged comments: `96.63%`
- `sft_pairs.v2.jsonl` active training comment outputs: `100%`
- `cpt_patch_gap_repair_dedup.jsonl` patch rows: `0%` tagged
- `sft_pairs_v2.jsonl` thread-aware prompt file: `0%` tagged

Why the inconsistency exists:

- `build_thread_aware_datasets.py` strips the visible tag from comment text and stores it separately as `comment_key` / `parent_comment_key` (`scripts/build_thread_aware_datasets.py:84-96`, `218-283`).
- The active SFT trainer does not use that file; it uses `sft_pairs.v2.jsonl`, which keeps the rendered tags (`train_sft.py:66-68`, `198-211`).
- The gap-repair patch rows are extracted sentences from promo comments and are written without tags (`scripts/sanitize_context.py`, live file check).

Should tags be preserved?

- For raw realism and export-style generation: yes.
  - They are visible surface form in the source crawl, and they correlate with reply depth.
- For product-style conditioned reply generation: optional.
  - In that case you can hide the tag at inference time, but you still need explicit structure fields so the model learns depth and parentage.

Best design:

- Preserve both views:
  - rendered-tag text for realism eval / raw-style generation
  - normalized no-tag text plus explicit `comment_key`, `parent_comment_key`, and `depth` for structured reply generation

What is wrong today is not “preserve vs strip”; it is that different parts of the stack do both, inconsistently.

## 7. Is the train/val split truly non-leaking after the v3 patch?

No.

The current situation:

- Exact text overlap between live `cpt_corpus.v3.jsonl` and live `val_set.v2.jsonl`: `0 / 2,451`
- Exact `source_id` overlap: `0 / 2,451`

That sounds clean, but it is not the whole story.

Thread-level leakage check:

- `252 / 2,451` val rows (`10.28%`) belong to threads that also appear in `cpt_patch_gap_repair_dedup.jsonl`
- That affects `44 / 566` unique val threads (`7.77%`)
- Breakdown of leaked val rows:
  - posts: `44`
  - comments: `208`

Why this happened:

- `remove_val_train_leak.py` only checks exact text duplicates against `cpt_corpus.v2.jsonl` (`scripts/remove_val_train_leak.py:9-40`).
- The patch rows are generated later from the full raw crawl, not from the train-only partition.
- Those patch rows use synthetic ids like `postid:comment:n`, so exact `source_id` matching does not catch them, but the post/thread prefix still matches the held-out val threads.

So the live split is only “exact-text non-leaking.” It is not “thread non-leaking.”

For a task whose goal is reply realism, thread leakage matters more than exact-text leakage.

## Additional design drifts worth calling out

### A. The richer thread-aware dataset exists, but the active trainer does not use it

- `scripts/build_thread_aware_datasets.py` creates:
  - title-conditioned post tasks
  - root-comment tasks with post context
  - reply-comment tasks with parent context
- But `train_sft.py` trains on `sft_pairs.v2.jsonl`, not on `sft_pairs.thread_aware.jsonl` or `sft_pairs_v2.jsonl`.

This is the clearest “design intent exists, active pipeline does not consume it” gap.

### B. The SFT trainer evaluates on raw val text only

- `train_sft.py` builds eval examples from `val_set.v2.jsonl` using `build_raw_example()` only (`train_sft.py:225-229`).

That means SFT validation loss cannot directly tell you whether comment conditioning or reply coherence improved.

### C. The live corpus still contains meaningful promo/operator tone

Using the repo’s own keyword family as a coarse heuristic:

- `cpt_corpus.v3.jsonl`: `2,963 / 48,125` rows (`6.16%`) still look promo-like
- `cpt_corpus.v2.jsonl`: `2,898 / 46,856` rows (`6.18%`)

So the current `v3` patch improved domain-term recovery, but it did not materially reduce promo/operator stylistic contamination.

## Recommended changes, in order

1. Rebuild `v3` from the train split only.
   - Generate any future patch rows after val threads are excluded, not from the full raw directory.

2. Make the active SFT path actually thread-aware.
   - Either train directly on `instruction/input/output` rows from `scripts/build_thread_aware_datasets.py`, or extend `train_sft.py` to consume `parent_comment_key` / parent text.

3. Put title into CPT post text.
   - `제목 + 본문` should be the default post serialization.

4. Add a context-serialized CPT stream for comments.
   - Keep standalone rows, but add a material percentage of `원글 + 부모댓글 + 답글` rows.

5. Unify comment-tag handling.
   - Either keep rendered tags everywhere relevant to realism, or generate both rendered and normalized views explicitly.

6. Rewrite evaluation around the actual target.
   - Stratify by `kind`
   - evaluate posts and comments separately
   - add a thread-coherence eval slice
   - measure punctuation / 초성 / emotive markers directly
   - add a promo/operator contamination metric

7. Refresh stale `v3` reports.
   - A `61,650`-row pre-dedup report should not be used as acceptance evidence for the live `48,125`-row artifact.

## Bottom line

Current `v3` is a better flat-text slang/style corpus than `v2`, but the active pipeline is still not fully designed for the thing the operating doc asks for: comments that feel native in thread context, not just local in vocabulary.

If the target remains “indistinguishable from real forum posts/comments,” the next gain will come from thread-aware training and leak-safe evaluation, not from increasing `seq_len` or adding more standalone patch rows.
