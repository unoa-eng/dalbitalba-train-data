#!/bin/bash
# Phase 2 full run: regenerate all 10,008 seed comments with boost-4 setup.
# Chunked (500/chunk) so a crash resumes from the last incomplete chunk.
# Usage: bash dalbit_phase2_full_launch.sh [start_chunk]
set -e
cd /Users/unoa/dalbitalba-train-data

SRC=/tmp/dalbit_phase2_src_bulk.jsonl
OUTROOT=/tmp/dalbit_phase2_full
CHUNK=500
TOTAL=9453
NCHUNKS=$(( (TOTAL + CHUNK - 1) / CHUNK ))
START=${1:-0}

mkdir -p "$OUTROOT"
for (( i=START; i<NCHUNKS; i++ )); do
  OFFSET=$(( i * CHUNK ))
  COUNT=$CHUNK
  if (( OFFSET + COUNT > TOTAL )); then COUNT=$(( TOTAL - OFFSET )); fi
  DIR="$OUTROOT/chunk_$(printf '%03d' $i)"
  if [[ -f "$DIR/samples.jsonl" ]]; then
    echo "chunk $i already complete, skipping"
    continue
  fi
  rm -rf "$DIR"
  echo "=== chunk $i: offset=$OFFSET count=$COUNT $(date '+%F %T') ==="
  .venv/bin/python scripts/cycle9_generation_audit.py \
    --adapter-path runs/cycle9/checkpoint_views/pref_targeted_artifact_v4_0000200 \
    --model-path runs/cycle7-mac-simul/qwen3-8b-mlx-4bit \
    --source-jsonl "$SRC" \
    --output-dir "$DIR" \
    --suite valid-prefix \
    --prompt-count $COUNT \
    --prompt-offset $OFFSET \
    --prompt-template raw \
    --stop-next-index \
    --max-tokens 200 \
    --temp 0.35 \
    --top-p 0.9 \
    --seed $(( 67 + i )) \
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
    --text-bias '사라=+0.5' || { echo "[err] chunk $i failed, continuing"; }
  echo "=== chunk $i done $(date '+%F %T') ==="
done
echo "ALL CHUNKS COMPLETE $(date '+%F %T')"
