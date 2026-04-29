# Data Strategy Proposal: Additive Gap-Repair & Contextual Sanitization

**Worker:** worker-1 (Data Strategy)
**Date:** 2026-04-27

## 1. Core Findings from 1st Refinement Loop
The "Hybrid Comment Separation" attempt in the 1st loop provided valuable data but failed to improve net metrics due to two factors:
- **Aggressive Filtering:** The recovery logic (`is_valid_user_part`) discarded any user part containing `TC` or `밀빵` because these terms are also "promo keywords". This effectively blinded the model to the very terms we were trying to fix.
- **Global Distribution Shift:** The "Full Rebuild" approach changed the token ratios of existing "OK" terms (e.g., `ㅅㅂ`, `ㅡㅡ`), causing new regressions.

## 2. Strategic Direction

### A. Additive Merge (Patching) vs. Full Rebuild
**Recommendation: Additive Patching.**
- We should treat `cpt_corpus.v2.jsonl` (46k rows) as a **Stable Base**.
- Instead of rebuilding the entire corpus, we will generate a **"Gap-Repair Patch"** dataset and merge it into the base.
- **Benefits:** Prevents "Whack-a-Mole" where fixing one term breaks another. It allows for surgical oversampling of specific terms without diluting the entire style profile.

### B. Contextual Ad Sanitization (Term-Centric Recovery)
The dilemma is that 95%+ of domain terms live in ads. Discarding ads = Discarding domain knowledge.
**Recommendation: Shift from "Filter Out Ads" to "Sanitize Context from Ads".**
1.  **Relax Filter for Hybrid:** If a `user_part` contains a target GAP term (`TC`, `밀빵`, `케어`), it should bypass the `still_promo` length and keyword check. A user asking "TC how much?" is high-value context.
2.  **Sentence Extraction from Pure Ads:** Instead of ignoring 12k pure ad comments:
    - Strip PII (Phone numbers, Kakao/Line IDs, URLs).
    - Extract sentences containing GAP terms.
    - Example: `"[광고] 강남 1등 도파민입니다. TC 당일지급, 밀빵 확실히 해드립니다."` 
      -> `[Sanitized]: "강남 1등 도파민입니다. TC 당일지급, 밀빵 확실히 해드립니다."`
    - This preserves the **domain-specific semantic link** between terms (e.g., TC + 당일지급) without the noise of contact info.

### C. The Role of Term Exposure in CPT
**Observation:** Qwen3-8B-Base knows "Korean", but it doesn't know the **dalbitalba dialect**.
- In this domain, `TC` is not "Total Carbon", it's "Table Charge".
- CPT is not just for style; it's for **semantic re-mapping**.
- If the model's exposure to `TC` in its specific sense (hourly pay context) is near zero, it will fail to generate it or respond to it correctly. Style is the "delivery", but vocabulary is the "content". We need both.

## 3. Implementation Plan

1.  **Update `scripts/split_hybrid_comments.py`:**
    - Implement `TERM_AWARE_BYPASS`: If a line contains `TC`, `밀빵`, `케어`, `쩜오`, `하이퍼`, it is preserved even if it looks "promo-ish".
2.  **Create `scripts/sanitize_context.py`:**
    - Logic: `Raw Ad` -> `PII Scrubbing` -> `Sentence Tokenization` -> `Keep if has GAP term`.
3.  **Generate `cpt_patch_v2_gap_repair.jsonl`:**
    - Combine relaxed hybrid recovery + sanitized ad sentences.
4.  **Strategic Oversampling:**
    - Duplicate the patch rows to reach a target ratio (e.g., 0.5-0.8 of raw crawl frequency) before merging with `cpt_corpus.v2.jsonl`.

## 4. Expected Outcome
- **Zero Regressions:** Stable base remains untouched.
- **TC/밀빵/케어 Recovery:** Direct injection of context-rich, noise-free sentences.
- **Budget Efficiency:** Adding ~5k-10k high-quality rows is more efficient for `budget30` than re-processing 60k rows with questionable quality.
