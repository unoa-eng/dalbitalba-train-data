#!/usr/bin/env bash
# Single-instance launcher for the live ops daemon (gap backfill is folded INTO daemon.py).
# Used directly or by the launchd KeepAlive agent (auto-restart on crash/login).
HERE="/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen/scripts/live"
cd "$HERE" || exit 1

# refresh prod env (service key) if a refresher exists; tolerate absence
[ -f /tmp/dalbit.env.prod ] || { echo "WARN: /tmp/dalbit.env.prod missing" >> "$HERE/run.log"; }

# count only the actual python daemon process (avoid matching monitoring shells)
if [ "$(ps -Ao command | grep 'live/daemon.py' | grep -i Python | grep -v grep | wc -l | tr -d ' ')" -ge 1 ]; then
  echo "[$(date '+%F %T')] daemon already running" >> "$HERE/run.log"
  exit 0
fi

exec python3 "$HERE/daemon.py"
