#!/bin/bash
# Mac mini local 1-step LoRA simul on quantized Qwen3-8B-Base.
# Mirrors the wiring used by chain_train_round2.sh + train_cpt.py
# (CPT_LR, CPT_WARMUP_RATIO, WANDB_ENTITY) without RunPod, BnB or CUDA.
#
# Why: prove the env → optimizer LR → wandb dalbit-ai run path runs end-to-end
# on Apple Silicon, removing the RunPod host-assign dependency that blocked
# cycle-6 US-C610.
#
# Required env (.env.local sources WANDB_*; recipe sources CPT_*):
#   set -a; source .env.local; source recipes/round2-cycle1.env; set +a
# Then:
#   bash scripts/macmini_local_train_simul.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MLX_MODEL="${MLX_MODEL:-runs/cycle7-mac-simul/qwen3-8b-mlx-4bit}"
DATA_DIR="${DATA_DIR:-runs/cycle7-mac-simul/data}"
ADAPTER_DIR="${ADAPTER_DIR:-runs/cycle7-mac-simul/adapter}"
LOG_FILE="${LOG_FILE:-runs/cycle7-mac-simul/simul.log}"

mkdir -p "$ADAPTER_DIR" "$(dirname "$LOG_FILE")"

if [ ! -d "$MLX_MODEL" ] || ! .venv/bin/python -c "import mlx_lm" >/dev/null 2>&1; then
    echo "[SKIP] mlx_lm or MLX model dir missing — simul skipped (not paper-grade blocker)"
    echo "       To enable: .venv/bin/pip install mlx-lm && .venv/bin/python -m mlx_lm convert --hf-path Qwen/Qwen3-8B-Base -q --q-bits 4 --mlx-path $MLX_MODEL"
    exit 0
fi

CPT_LR="${CPT_LR:-2e-4}"
LORA_R="${LORA_R:-8}"

echo "[simul] model=$MLX_MODEL lr=$CPT_LR lora_r=$LORA_R wandb_entity=${WANDB_ENTITY:-<UNSET>} project=${WANDB_PROJECT:-<UNSET>}"

# WANDB_NAME makes the run identifiable in the dashboard.
export WANDB_NAME="${WANDB_NAME:-macmini-simul-$(date -u +%Y%m%dT%H%M%SZ)}"

.venv/bin/python -m mlx_lm.lora \
    --model "$MLX_MODEL" \
    --train \
    --data "$DATA_DIR" \
    --fine-tune-type lora \
    --num-layers 1 \
    --batch-size 1 \
    --iters 1 \
    --learning-rate "$CPT_LR" \
    --steps-per-report 1 \
    --steps-per-eval 100 \
    --grad-accumulation-steps 1 \
    --adapter-path "$ADAPTER_DIR" \
    --save-every 1 \
    --max-seq-length 512 \
    --report-to wandb \
    --project-name "${WANDB_PROJECT:-dalbitalba-round2}" \
    --seed 42 2>&1 | tee "$LOG_FILE"
