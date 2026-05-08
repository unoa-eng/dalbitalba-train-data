#!/bin/bash
# Stage F.v2: MLX SFT on Qwen3-4B-Base with reduced layers + grad-checkpoint
set -e
cd /Users/unoa/dalbitalba-train-data
START=$(date +%s)
echo "=== Stage F.v2 (4B reduced) started at $(date) ===" | tee runs/local-integrity-2026-05-08/F_train_4b_v2.log
/Users/unoa/projects/dalbitalba-train-data/.venv-mlx312/bin/mlx_lm.lora \
  --model Qwen/Qwen3-4B-Base \
  --train --data runs/local-smoke-2026-05-08-comprehensive/ \
  --iters 1000 --batch-size 1 --learning-rate 5e-5 \
  --num-layers 8 \
  --adapter-path runs/local-integrity-2026-05-08/F_adapters_4b_v2 \
  --val-batches 10 --steps-per-eval 200 --steps-per-report 50 \
  --max-seq-length 768 \
  --grad-checkpoint \
  2>&1 | tee -a runs/local-integrity-2026-05-08/F_train_4b_v2.log
END=$(date +%s)
echo "=== Stage F.v2 finished at $(date) (elapsed $((END-START))s) ===" | tee -a runs/local-integrity-2026-05-08/F_train_4b_v2.log
