# Hyperparameter Comparison: Qwen3-0.6B MLX LoRA

Date: 2026-04-29
Worker: `worker-2`

## Setup

- Base model: `Qwen/Qwen3-0.6B`
- Shared data split: `runs/ablation/_shared/structured_v3_450_25_25_seed42`
- Source corpus: `v3-data/cpt_structured_v3.jsonl`
- Split sizes: `450 train / 25 valid / 25 test`
- Sample prompts: `runs/ablation/_shared/sample_prompts_v1.jsonl`
- All runs used `200` iterations, batch size `1`, seed `42`, and saved adapters at `runs/ablation/B*/adapter`

## Main Results

| Run | Config | Val loss curve (`1 -> 50 -> 100 -> 150 -> 200`) | Final val | Test loss | Structure | Sample read |
|---|---|---|---:|---:|---:|---|
| `B1` | `lr=1e-4`, `num_layers=8` | `5.008 -> 3.917 -> 3.536 -> 3.448 -> 3.469` | `3.469` | `3.637` | `0.90` | Reasonable baseline. Short comments land often, but 2/20 samples break closing-tag balance and one comment run rambled to the max-length cap. |
| `B2` | `lr=2e-4`, `num_layers=8` | `5.008 -> 4.454 -> 4.105 -> 4.007 -> 3.898` | `3.898` | `4.391` | `1.00` | Clearly worst. Loss stays high throughout and many outputs collapse into short noisy fragments or odd question-like endings. |
| `B3` | `lr=5e-5`, `num_layers=8` | `5.008 -> 3.537 -> 3.439 -> 3.359 -> 3.372` | `3.372` | `3.502` | `1.00` | Best overall. Lowest validation and test loss, fully balanced tags, and slightly longer/more topical post continuations than the other runs. |
| `B4` | `lr=1e-4`, `num_layers=16` | `5.008 -> 3.988 -> 3.664 -> 3.555 -> 3.579` | `3.579` | `3.902` | `1.00` | More LoRA layers did not help. Samples preserve tags but comments get noisier (`ㅠㅠㅠㅠ도방...`, `ㅇㄴㄴㄹㅇㅈㅎㅎ`) without a loss benefit. |

Structure column is the balanced-tag ratio from the concurrently generated `eval.json` files in each run directory.

## Loss-Curve Read

- `B3` is the only run that stays ahead of the field from step `50` onward and finishes with the best final validation loss.
- `B1` looks competitive through step `150`, then rebounds slightly by step `200`.
- `B2` never recovers from the aggressive learning rate. It is uniformly worse at every evaluation checkpoint.
- `B4` improves over `B2` but still trails the 8-layer `B1`/`B3` runs. On this `0.6B` local setup, additional LoRA depth appears to add capacity without improving optimization.

## Sample Quality Notes

- `B1` baseline examples are short and safe, but occasionally malformed on comments. Example prompt 19 ended as `ㅎ남슴인데`, and the run produced only `0.90` balanced structure overall.
- `B2` preserves wrappers but quality drops visibly. Example prompt 13 became `ㄷ / 말에 건인다며서속아님<|/comment|>`, which is both noisier and less readable than the lower-LR runs.
- `B3` is still imperfect, but it most often keeps the structured wrapper and emits a plausible domain-adjacent continuation. Example prompt 7 produced `게디만 이은가 셔츠가 낫다는 이야기도 있죠...<|/post|>`, which is still messy Korean but is closer to the input topic than the competing runs.
- `B4` does not justify the extra layers. Example prompt 11 became `ㅠㅠㅠㅠ도방에서 분동 동아리 많음(ㅠㅠㅠㅠㅠㄱㄱㄷㅇ<|/comment|>`, which is structurally closed but semantically worse than `B1`/`B3`.

## Recommendation

Use `B3` as the local MLX default for structured-v3 sanity checks:

- `learning-rate=5e-5`
- `num-layers=8`

Why:

- Best final validation loss: `3.372`
- Best test loss: `3.502`
- No structure-balance failures
- Better sample behavior than `B2` and `B4`, while avoiding the late-step rebound seen in `B1`

What not to carry forward:

- Do not use `2e-4` on this setup. It is consistently too hot.
- Do not increase to `16` LoRA layers for the local `0.6B` ablation path unless there is a separate reason to test larger capacity. It adds cost and degraded this controlled comparison.

## Cross-Check Against A2

The concurrently produced `A2-structured-v3` run currently shows `3.455` final validation loss and `3.042` test loss. That is useful as a rough sanity check that an 8-layer structured run remains the right comparison target, but it is not a strict apples-to-apples result unless the exact same split and prompt set were used. For the controlled worker-2 matrix above, `B3` remains the best recommendation.

## Artifacts

- `runs/ablation/B1/`
- `runs/ablation/B2/`
- `runs/ablation/B3/`
- `runs/ablation/B4/`
- Shared split and prompt set: `runs/ablation/_shared/`
