#!/bin/bash
# Autonomous Phase 2 full 63k run.
# Sequence:
#   1. Wait for pilot N=50 to finish (chunk_0000000)
#   2. Quick quality sanity check on first 5 samples (Hangul ratio ≥ 0.5)
#   3. If pass: continue chunked generation from offset 50 to 63274
#   4. Periodic ETA log every chunk
#
# Decision rule (autonomous, no user gate):
#   PASS = (Hangul ratio of first 5 samples avg ≥ 0.5)
#   FAIL = abort with diagnostic to runs/cycle10-phase2-orpo-expansion/ABORT.txt
#
# Throughput reality: ~5-8 min/sample on Mac mini single-worker.
# 63k full = 5-30 days wall-clock. Pilot proves the pipeline; full extends it.

set -uo pipefail
cd /Users/unoa/dalbitalba-train-data

OUT=runs/cycle10-phase2-orpo-expansion
DRIVER=runs/cycle10-phase1-claude-direct/phase2_production_driver.sh
PILOT_DIR="$OUT/chunk_0000000"

echo "[auto] waiting for pilot ($PILOT_DIR) to complete..."

# Wait until pilot's 50th sample exists (or 24h max)
deadline=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  done=$(ls "$PILOT_DIR"/raw/sample_*.text.txt 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done" = "50" ]; then
    echo "[auto] pilot complete (50 samples)"
    break
  fi
  printf "\r[auto] pilot progress: %s/50" "$done"
  sleep 60
done
echo

# Sanity check first 5 samples
echo "[auto] quality check on first 5 samples..."
.venv/bin/python - <<'PYEOF' || { echo "[auto] sanity check failed; aborting" > "$OUT/ABORT.txt"; exit 1; }
import json, re, glob
samples = sorted(glob.glob("runs/cycle10-phase2-orpo-expansion/chunk_0000000/clean/sample_*.text.txt"))[:5]
if not samples:
    print("no clean samples"); raise SystemExit(2)
ratios = []
for p in samples:
    with open(p) as f:
        txt = f.read().strip()
    total = max(1, len(txt))
    hangul = sum(1 for c in txt if '가' <= c <= '힯')
    ratio = hangul / total
    ratios.append(ratio)
    print(f"  {p}: hangul_ratio={ratio:.2f} len={len(txt)} sample={txt[:120]!r}")
avg = sum(ratios) / len(ratios)
print(f"avg hangul_ratio={avg:.3f}")
if avg < 0.5:
    print(f"FAIL: avg hangul_ratio {avg:.3f} < 0.5"); raise SystemExit(3)
print("PASS")
PYEOF

echo "[auto] starting full chunked generation from offset 50 → 63274"
CHUNK=200 bash "$DRIVER" 50 2>&1 | tee -a "$OUT/full_run.log"
echo "[auto] full run finished"
date >> "$OUT/full_run.log"
