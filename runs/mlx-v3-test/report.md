# MLX v3 local LoRA test

Date: 2026-04-29
Repo: `/Users/unoa/projects/dalbitalba-train-data`
Worker: `worker-1`

## Environment

- Requested install check:
  - `pip3 list | grep mlx` showed `mlx 0.31.1`
  - `mlx-lm` was not available in the default `python3`
- Default Homebrew `python3` is `3.14.4`, but its `pip3` is broken with a `pyexpat` / `libexpat` import error.
- Workaround used:
  - Created local venv: `.venv-mlx312`
  - Installed `mlx 0.31.2` and `mlx-lm 0.31.3` under `python3.12`
- Model cache found at:
  - `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`

## Data prep

- Source file: `v3-data/cpt_structured_v3.jsonl`
- Source rows available: `53,675`
- Requested sample file created:
  - `/tmp/mlx-train-sample.jsonl`
  - 500 rows, each normalized to `{"text": ...}`
- MLX CLI adjustment:
  - `mlx_lm lora` does not accept a single JSONL file for `--data`
  - It requires a directory with `train.jsonl`, `valid.jsonl`, `test.jsonl`
- Split created:
  - `/tmp/mlx-train-sample-data/train.jsonl` = 450
  - `/tmp/mlx-train-sample-data/valid.jsonl` = 25
  - `/tmp/mlx-train-sample-data/test.jsonl` = 25

## Training command

Requested command had two incompatibilities with current `mlx_lm`:
- `python3 -m mlx_lm.lora` is deprecated
- `--lora-layers 8` is now `--num-layers 8`

Run command:

```bash
.venv-mlx312/bin/python -m mlx_lm lora \
  --model Qwen/Qwen3-0.6B \
  --train \
  --data /tmp/mlx-train-sample-data \
  --iters 200 \
  --batch-size 1 \
  --num-layers 8 \
  --adapter-path runs/mlx-v3-test/adapter \
  --steps-per-report 10 \
  --steps-per-eval 20 \
  --save-every 100 \
  --test
```

Artifacts:
- Training log: `runs/mlx-v3-test/train.log`
- Adapter dir: `runs/mlx-v3-test/adapter/`
- Val curve CSV: `runs/mlx-v3-test/val_loss_curve.csv`
- Generated samples: `runs/mlx-v3-test/generated_samples.jsonl`

## Metrics

- Trainable parameters: `1.442M / 596.050M` (`0.242%`)
- Peak memory: `3.069 GB`
- Final test loss: `3.735`
- Final test perplexity: `41.902`

Validation loss curve:

| Iter | Val loss |
|---|---:|
| 1 | 5.046 |
| 20 | 3.623 |
| 40 | 3.448 |
| 60 | 3.395 |
| 80 | 3.349 |
| 100 | 3.283 |
| 120 | 3.271 |
| 140 | 3.261 |
| 160 | 3.252 |
| 180 | 3.240 |
| 200 | 3.229 |

Observations:
- Loss improved monotonically through all recorded eval points.
- Unlike the earlier v2 note in `TRAINING_DESIGN_V3.md` (`val_loss_best=4.166 @200iter`), this 500-row v3 sample did **not** show a rebound by 200 iters.
- This is **not** an apples-to-apples comparison with the earlier v2 run because:
  - v2 used `42,170 / 4,686` train/val rows
  - this run used only `450 / 25` train/val rows
  - this run also used `8` LoRA layers instead of the earlier `16`-layer setup documented in the design note

## Sample quality review

20 samples were generated from the trained adapter:
- 10 post-style prompts
- 10 comment-style prompts

Quick structural checks:
- Post outputs containing closing `</post>` tag: `9/10`
- Comment outputs containing closing `</comment>` tag: `9/10`
- Outputs with obvious date-like leakage (`2023년` / `2025년`): `2/20`

Quality summary:

### What improved

- The adapter clearly learned the structured wrapper tokens:
  - `<|post|> ... <|/post|>`
  - `<|comment depth=0|> ... <|/comment|>`
- Short comment-style outputs are the strongest result.
- Comment generations carry some target-domain surface style:
  - laughter / short reactions: `ㅋㅋ`, `ㅠ`, `ㅎ`
  - compact one-line responses
  - informal tone

Representative comment outputs:
- `[12]` `초이스 안 잡히면 멘탈 나가죠 ㅠ`
- `[13]` `TC 그 정도면 너무 짠데 ㅋㅋ`
- `[16]` `손놈 진상은 진짜 운빨도 큼 ㅋㅋㅋㅋㅋㅋ`

### What is still weak

- Post-level coherence is poor in most samples.
- Several post generations drift into metadata-like or unrelated text.
- One post broke structure by opening a nested comment tag before closing the post.
- Some outputs contain obvious garbage or repeated template leakage.

Representative failures:
- `[01]` incoherent noun chaining: `모바일 애당나리적인 활동 추천만 추천하는지`
- `[04]` repeated date leakage: `지금 2025년 7월 10일` repeated, then structure break into comment tag
- `[07]` metadata-like leakage: `테이블: 500개 / 언어: 영어`

### Overall read

- Comment style transfer: **partially successful**
- Post style transfer: **weak**
- Semantic coherence: **still poor at 0.6B / 500-row sample scale**

This matches the general pattern already noted in `TRAINING_DESIGN_V3.md`: local MLX on `Qwen3-0.6B` is useful as a sanity check for structure and tone direction, but not enough to establish high-quality post generation.

## Bottom line

- The v3 structured file is trainable in MLX after adapting to the current CLI format.
- Training behavior is stable and the validation curve is better than the prior v2 note on a raw-loss basis, but the dataset sizes are too different to treat that as a win by itself.
- The most credible signal from this run is:
  - structured tags are learnable
  - short reply/comment tone transfers
  - coherent long-form post generation remains weak
