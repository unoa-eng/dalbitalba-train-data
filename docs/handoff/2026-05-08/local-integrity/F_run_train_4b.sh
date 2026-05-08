#!/bin/bash
# Stage F: MLX SFT on Qwen3-4B-Base, 1000 iter
set -e
cd /Users/unoa/dalbitalba-train-data

START=$(date +%s)
echo "=== Stage F started at $(date) ===" | tee runs/local-integrity-2026-05-08/F_train_4b.log

/Users/unoa/projects/dalbitalba-train-data/.venv-mlx312/bin/mlx_lm.lora \
  --model Qwen/Qwen3-4B-Base \
  --train --data runs/local-smoke-2026-05-08-comprehensive/ \
  --iters 1000 --batch-size 1 --learning-rate 5e-5 \
  --num-layers 24 \
  --adapter-path runs/local-integrity-2026-05-08/F_adapters_4b \
  --val-batches 20 --steps-per-eval 100 --steps-per-report 50 \
  --max-seq-length 1024 \
  2>&1 | tee -a runs/local-integrity-2026-05-08/F_train_4b.log

END=$(date +%s)
echo "=== Stage F finished at $(date) (elapsed $((END-START))s) ===" | tee -a runs/local-integrity-2026-05-08/F_train_4b.log
