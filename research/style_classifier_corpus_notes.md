# Korean Style Classifier — Corpus Notes

**Date:** 2026-05-07  
**Task:** P1-B binary style classifier (community vs formal Korean)

## Positive Corpus (label=1 — community/informal)

**Source:** `cpt_corpus.v2.jsonl`  
**Filter:** `len(text) >= 30` AND regex match on `ㅈㄴ|ㅇㅈ|ㄹㅇ|ㅋㅋ|ㅎㅎ|쩜오|텐카|초이스|손놈|업진|밀빵`  
**Stratification:** by `length_bucket` field, capped at 1,500 per bucket  

Available matching rows before cap:

| Bucket | Count |
|--------|-------|
| sm     | 2,922 |
| md     | 3,577 |
| lg     | 1,575 |
| xl     |   939 |
| xxl    |    54 |
| **total** | **9,067** |

After per-bucket cap and shuffle: **5,000 samples** selected (TARGET_POSITIVE).

Note: `xxl` bucket only contributes 54 rows (well under cap of 1,500). This is expected
— very long informal posts are rare in the crawl data. No mitigation needed for binary
classification since the other buckets are well-represented.

The Jamo normalization fix applied in `feat/p0-data-hardening` (commit `ba4056f`) was
critical — without restoring `ㅋㅋ` / `ㅎㅎ` / `ㅇㅈ` from NFKC Choseong form, the
keyword filter would match near-zero rows.

## Negative Corpus (label=0 — formal Korean)

**Attempted sources (in priority order):**

1. `wikimedia/wikipedia` (20231101.ko subset via HuggingFace datasets streaming)
   - First paragraph of each article, truncated to 512 chars
   - Target: 5,000 samples
   - Status: attempted at runtime; falls back if network unavailable

2. `HAERAE-HUB/KMMLU` (question stems)
   - Academic exam questions in formal Korean
   - Status: second fallback

3. Bundled fallback (~50 sentences, repeated to reach 5,000)
   - Government/academic/news-style Korean sentences
   - Covers: government ministry announcements, legal text, science reports, history
   - **Limitation:** Only 50 unique sentences; diversity is limited. The repetition
     means the model may overfit to specific lexical patterns of these sentences.
     If bundled fallback is used, expect slightly lower test AUC than with Wikipedia.
   - Mitigation: sentences span diverse domains (constitutional law, ecology, education,
     economics, medicine) to reduce lexical bias.

## Split

80/10/10 stratified train/val/test split using `sklearn.model_selection.train_test_split`
with `stratify=labels` and `random_state=42`.

## Why AUC < 0.65 = "indistinguishable"

The phase6 gate uses AUC as a proxy for style fidelity. When the model generates
text that is sufficiently informal/community-style, the classifier cannot distinguish
AI output from real community posts — hence AUC < 0.65 on (AI=community, raw=formal)
comparison. AUC > 0.95 on the test split confirms the classifier itself is accurate.

## Known Limitations

- CPU-only training (~30-60 min for 3 epochs on 10,000 samples)
- klue/roberta-base is 110M params — adequate for binary style classification
- The community keywords are domain-specific (유흥업 terminology included); classifier
  may not generalize to other informal Korean domains
- Bundled fallback corpus lacks the lexical diversity of Wikipedia; if used, treat
  AUC estimates as slightly optimistic
