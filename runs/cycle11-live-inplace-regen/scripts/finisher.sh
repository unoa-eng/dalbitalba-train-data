#!/usr/bin/env bash
# Autonomous tail orchestrator (runs off Claude budget). Launch in background.
B="/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen"
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$B/finisher.log"; }

log "finisher start"
# 1) while initial generation runs, incrementally load completed good posts
while pgrep -f "run_gen.sh" >/dev/null 2>&1; do
  python3 "$B/scripts/load_batches.py" >> "$B/load.log" 2>&1 || true
  sleep 300
done
log "initial generation finished; gen json=$(ls "$B"/batch_out/*.json 2>/dev/null | wc -l | tr -d ' ')/276"

# 2) second generation pass to fill any batches that failed JSON extraction
bash "$B/scripts/run_gen.sh" 8 >> "$B/run_gen.log" 2>&1
log "fill pass done; gen json=$(ls "$B"/batch_out/*.json 2>/dev/null | wc -l | tr -d ' ')/276"

# 3) final load of all good posts
python3 "$B/scripts/load_batches.py" >> "$B/load.log" 2>&1 || true
log "final load done"

# 4) regenerate the requeue tail (per-post failures + structural batches), twice for residual
python3 "$B/scripts/regen_requeue.py" >> "$B/requeue.log" 2>&1 || true
python3 "$B/scripts/regen_requeue.py" >> "$B/requeue.log" 2>&1 || true
log "requeue regen done; residual=$(python3 -c "import json;print(len(json.load(open('$B/requeue_posts.json'))))" 2>/dev/null)"
log "FINISHER DONE"
