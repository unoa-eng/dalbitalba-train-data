#!/bin/bash
# Phase 2 production: cycle9 ORPO adapter 63k expansion.
# Calls scripts/cycle9_generation_audit.py in chunks against Phase 1 seed-derived
# prompt pool. Resumable: each chunk has its own --prompt-offset.
#
# Validated sampling (from agent smoke 2026-06-10): produces natural Korean
# domain replies. Adapter: phase3-pref-targeted-artifact-v4-smoke200/0000200.
#
# Throughput estimate (single MLX worker on Mac mini):
#   ~6 s/sample × 63274 = ~105 hours wall-clock.
# Use --chunk-size for resumable batching.

set -euo pipefail
cd /Users/unoa/dalbitalba-train-data

POOL=runs/cycle10-phase1-claude-direct/phase2_prompt_pool.jsonl
OUT_ROOT=runs/cycle10-phase2-orpo-expansion
ADAPTER=runs/cycle9/checkpoint_views/pref_targeted_artifact_v4_0000200
MODEL=runs/cycle7-mac-simul/qwen3-8b-mlx-4bit
CHUNK=${CHUNK:-200}
TOTAL=$(wc -l < "$POOL")
START_OFFSET=${1:-0}

mkdir -p "$OUT_ROOT"
echo "[phase2] total=$TOTAL chunk=$CHUNK start_offset=$START_OFFSET"

offset=$START_OFFSET
while [ "$offset" -lt "$TOTAL" ]; do
  chunk_dir="$OUT_ROOT/chunk_$(printf '%07d' $offset)"
  if [ -f "$chunk_dir/summary.json" ]; then
    echo "[skip] $chunk_dir already complete"
    offset=$((offset + CHUNK))
    continue
  fi
  echo "[chunk] offset=$offset -> $chunk_dir"
  rm -rf "$chunk_dir"
  .venv/bin/python scripts/cycle9_generation_audit.py \
    --adapter-path "$ADAPTER" \
    --model-path "$MODEL" \
    --source-jsonl "$POOL" \
    --output-dir "$chunk_dir" \
    --suite valid-prefix \
    --prompt-count $CHUNK \
    --prompt-offset $offset \
    --prompt-template raw \
    --stop-next-index \
    --max-tokens 200 \
    --temp 0.35 \
    --top-p 0.9 \
    --seed 67 \
    --repetition-penalty 1.3 \
    --repetition-context-size 128 \
    --min-new-tokens 60 \
    --ban-control-tokens \
    --text-bias '[1-3]=-5' \
    --text-bias '비회원=-2' \
    --text-bias '저도=-1' \
    --text-bias '개=-1.5' \
    --text-bias '2026=-5' \
    --text-bias '병오년=-5' \
    --text-bias '팁길=-5' \
    --text-bias '텐카=-3.5' \
    --text-bias 'ㅈㄴ=-2.5' \
    --text-bias 'ㅇㅈ=-2.5' \
    --text-bias 'ㅇㅈㄹ=-2.0' \
    --text-bias 'ㅎㅍ=-1.5' \
    --text-bias '쩜오=+3.0' \
    --text-bias '도파민=+2.0' \
    --text-bias '초이스=+1.5' \
    --text-bias '사라=+0.5' \
    2>&1 | tail -5 || echo "[err] chunk $offset failed"
  offset=$((offset + CHUNK))
done
echo "[done] all chunks processed"
