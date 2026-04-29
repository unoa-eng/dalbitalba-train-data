# Ablation Results

Date: 2026-04-29
Model: `Qwen/Qwen3-0.6B` with MLX LoRA
Prompt set: `runs/ablation/prompts.jsonl`

## Main Results

`jondaet%` and `banmal-like%` are sample-level rates from generated outputs. `coherence` is a 1-5 readability score from sample review aided by structural and repetition heuristics. For `A1`, the samples live in `A1-flat-v2` while the training curve was written to `A1-flat-cpt-v2`; the final val loss below uses that saved curve.

| ID | Condition | Final val loss | Structure fidelity | jondaet% | banmal-like% | Coherence | TC | 밀빵 | 케어 | ㅋㅋ | ㅠㅠ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | Base model, no FT | N/A | 0% | 35.0 | 25.0 | 3.25 | 9 | 0 | 7 | 0 | 0 |
| A1 | Flat CPT v2 | 4.531 | 0% | 15.0 | 50.0 | 3.30 | 0 | 0 | 0 | 3 | 55 |
| A2 | Structured CPT v3 | 3.455 | 95% | 10.0 | 45.0 | 3.60 | 0 | 0 | 0 | 19 | 1 |
| A3 | Context stream | 4.271 | 0% | 30.0 | 80.0 | 3.50 | 0 | 0 | 0 | 109 | 104 |
| A4 | Flat CPT v2 + title | 4.604 | 0% | 5.0 | 50.0 | 3.30 | 0 | 0 | 0 | 226 | 83 |
| B1 | Structured, LR 1e-4, 8 layers | 3.469 | 90% | 5.0 | 45.0 | 3.70 | 0 | 0 | 0 | 4 | 162 |
| B2 | Structured, LR 2e-4, 8 layers | 3.898 | 100% | 15.0 | 45.0 | 3.90 | 0 | 0 | 0 | 15 | 4 |
| B3 | Structured, LR 5e-5, 8 layers | 3.372 | 100% | 10.0 | 40.0 | 3.85 | 0 | 0 | 0 | 7 | 3 |
| B4 | Structured, LR 1e-4, 16 layers | 3.579 | 100% | 5.0 | 40.0 | 3.90 | 0 | 0 | 0 | 15 | 9 |

## Findings

1. Structure tokens are the clearest win. `A2` drops val loss from `4.531` in `A1` to `3.455` and moves structure fidelity from `0%` to `95%`. `A3` confirms that extra context without explicit structure does not solve the formatting problem.
2. The base model is unusable for this domain without fine-tuning. `B0` repeatedly copies `제목:` scaffolds, emits unrelated metadata, and never closes the structured format correctly.
3. Within the structured runs, `LR=5e-5` is the strongest default on 0.6B. `B3` has the best final val loss (`3.372`), perfect structural closure, and the closest banmal-like rate to the structured reference set (`40.0%` vs reference `41.6%`).
4. `LR=2e-4` is too aggressive at this scale and budget. `B2` keeps clean tags, but its loss regresses sharply relative to `B3` and `A2`, which is consistent with overshooting on a small model at only 200 steps.
5. Going from 8 LoRA layers to 16 layers is not justified here. `B4` is cleaner than `B1` on structure, but it is still worse than `B3` on loss and does not produce a meaningful lexical gain.
6. Flat-data variants overproduce surface markers without semantic control. `A3` and `A4` spray `ㅋㅋ` and `ㅠㅠ`, but the outputs are less grounded and structurally broken. Marker counts alone are not a quality metric.

## Best Configuration

Recommendation: use `B3` as the local winner for the RunPod 8B starting point.

Why:
- Best final val loss in the matrix: `3.372`
- `100%` structure fidelity on the prompt set
- Tone is closest to the structured source distribution
- Sample quality is still noisy, but it avoids the worst repetition collapse seen in `A3`, `A4`, and `B1`

Practical ranking for structured runs:
1. `B3` (`5e-5`, 8 layers)
2. `A2` / `B1` as near-baseline structured references
3. `B4` if higher capacity adapters are revisited later
4. `B2` should not be the default

## What Transfers To 8B

Likely to transfer:
- Keep explicit structure tokens. This is the most stable conclusion in the entire matrix.
- Start the real run with a lower LR than `2e-4`. The 0.6B ranking strongly favors the conservative side.
- Treat 8 LoRA layers as the default until a larger-model sweep proves 16 helps.
- Use local MLX runs for ranking data format and coarse hyperparameters before spending GPU time.

Mostly 0.6B artifacts:
- Garbled semantics inside otherwise well-formed tags
- Extreme repetition of `ㅋㅋ`, `ㅠㅠ`, or short filler completions
- Weak recovery of rare domain lexicon such as `밀빵`
- Over-compression into one-line answers even when prompts imply richer posts/comments

Interpretation: the 0.6B model is good enough to rank structure and optimization choices, but not good enough to judge final realism. The 8B run should inherit the structure and LR lessons, not the absolute sample ceiling.

## Updated Training Design Recommendations

Recommended updates to `TRAINING_DESIGN_V3.md` based on this evidence:

1. Replace the current default local LoRA LR recommendation of `2e-4` with `5e-5` for the 0.6B MLX sanity-check track.
2. Make structured CPT the default corpus format for all further local ablations and for the first 8B CPT attempt.
3. Keep 8 LoRA layers as the default sweep anchor. Only revisit 16 layers if the 8B run shows underfitting after the LR is stabilized.
4. Explicitly separate two goals in the document:
   - local 0.6B experiments for ranking format and hyperparameters
   - 8B training for actual realism and lexical fidelity
5. Expand evaluation notes so domain-term counts are interpreted together with structure and readability. Raw marker frequency can be misleading because bad runs also overproduce slang tokens.

## Notes

- `A4` sample generation finished with an empty output file from the original worker process; the final `runs/ablation/A4-flat-v2-title/samples.jsonl` was regenerated from the saved adapter using `scripts/ablation_generate_mlx.py`.
- `A1` artifacts were split across two directories by the worker process:
  - samples: `runs/ablation/A1-flat-v2/samples.jsonl`
  - val curve / train log: `runs/ablation/A1-flat-cpt-v2/`
- Consolidated objective metrics are saved in:
  - `runs/ablation/matrix_metrics.json`
  - `runs/ablation/matrix_review.md`
