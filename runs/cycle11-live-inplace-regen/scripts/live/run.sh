#!/usr/bin/env bash
# Launch the live ops pipeline: gap backfill (once) then the real-time daemon.
# Single-instance guarded. Restart-safe (daemon persists state).
HERE="/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen/scripts/live"
cd "$HERE" || exit 1

if pgrep -f "live/daemon.py" >/dev/null 2>&1; then
  echo "daemon already running (pid $(pgrep -f 'live/daemon.py' | tr '\n' ' '))"; exit 0
fi

# 1) one-time gap backfill (marker-guarded inside)
python3 "$HERE/gapfill.py" >> "$HERE/gapfill.log" 2>&1

# 2) real-time daemon forever
exec python3 "$HERE/daemon.py"
