#!/usr/bin/env bash
# Parallel Codex generation over all batches. Restart-safe (gen_one skips finished).
# Usage: run_gen.sh [PARALLEL]   default 6
BASE="/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen"
P="${1:-6}"
mkdir -p "$BASE/batch_out"
ls "$BASE/batch_in" | sed 's/\.json$//' | sort | \
  xargs -P "$P" -I{} bash "$BASE/scripts/gen_one.sh" {}
echo "=== generation pass complete ==="
echo "produced: $(ls "$BASE/batch_out"/*.json 2>/dev/null | wc -l) / 276 batches"
