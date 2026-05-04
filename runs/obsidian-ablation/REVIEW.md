# Obsidian Auxiliary Training Review

## Scope

Question: should the Obsidian export be included more directly in training?

Answer adopted for this repo: only through an explicit metadata-only curation step, and only as an opt-in auxiliary signal.

## Competitive Review

Three parallel reviewers were used to attack and defend the design.

- `Parfit`: accepted an Obsidian path only if it stayed metadata-only, preserved the existing inference prompt format, and remained opt-in.
- `Volta`: rejected the earlier direct prompt/profile injection idea because it would create train/inference schema drift, overweight a small sample, and blur provenance boundaries.
- `Nash`: accepted the path only with operational guardrails: no raw vault body text, explicit provenance, conservative sampling, and measurable audit outputs.

Consensus: the repo should not train on raw Obsidian page text. The safe path is to use Obsidian frontmatter only, map it by `source_id`, and limit its effect to a mild sampling bias in SFT.

## Accepted Policy

- No raw Obsidian markdown body, title text, or comment text enters training.
- No prompt mutation is allowed for the Obsidian experiment path.
- No target mutation is allowed for the Obsidian experiment path.
- The only auxiliary signal is membership of a training row in the curated Obsidian sample, matched by `source_id` or `root_id`.
- The path is opt-in only through `SFT_OBSIDIAN_ENABLE=1`.
- The canonical `budget30_v2` path remains unchanged.

## Implementation

- `scripts/build_obsidian_style_map.py` extracts frontmatter metadata only.
- `scripts/build_obsidian_sft_variant.py` builds a conservative SFT variant by oversampling matched rows.
- `recipes/budget30_v2_obsidian.env` enables the opt-in profile for RunPod training.
- `chain_train.sh` and `chain_train_round2.sh` build the variant at runtime when the feature flag is enabled.

## Local Validation

Input dataset: `v3-data/sft_5task_v3.jsonl`

- Base rows: `58,476`
- Obsidian-matched rows: `3,332`
- Base matched ratio: `0.056981`
- Target matched ratio: `0.08`
- Duplicates added: `1,464`
- Variant rows: `59,940`
- Variant matched ratio: `0.080013`

Matched rows by task:

- `T1_title_to_body`: `466`
- `T2_root_comment`: `1,684`
- `T3_reply_comment`: `740`
- `T4_topic_to_post`: `136`
- `T5_continue`: `306`

Duplicated rows by task:

- `T1_title_to_body`: `212`
- `T2_root_comment`: `744`
- `T3_reply_comment`: `328`
- `T4_topic_to_post`: `53`
- `T5_continue`: `127`

Policy checks:

- `prompt_mutation=false`
- `target_mutation=false`
- `membership_signal_only=true`

## Conclusion

The safe conclusion from the parallel review is not "train on Obsidian more aggressively."

The safe conclusion is:

1. keep raw source DB / crawl data as the primary training truth,
2. keep Obsidian as a secondary curated signal,
3. enforce metadata-only inclusion,
4. preserve prompt compatibility,
5. treat the Obsidian path as an auditable opt-in experiment until downstream eval proves benefit.
