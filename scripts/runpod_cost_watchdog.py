#!/usr/bin/env python3
"""RunPod cost-cap watchdog.

Polls the RunPod REST API for `<pod_id>` and accumulates
`costPerHr * uptimeSeconds / 3600` as the live spend estimate. When the
estimate crosses 95% of the cap a single ntfy WARN fires; when it crosses
100% the watchdog calls `stop_pod` and writes an abort marker that
chain_train_round2.sh's graceful_abort path can detect.

Designed to be spawned by chain_train_round2.sh as a background process:

    python3 scripts/runpod_cost_watchdog.py \
        --pod-id "$RUNPOD_POD_ID" \
        --cap "${BUDGET_CAP_USD:-60}" \
        --state-dir ".state/round2" &

The chain trap kills this PID at end-of-run. Standalone, the script also
respects SIGTERM/SIGINT and exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import urllib.request
from pathlib import Path

RUNPOD_REST = "https://rest.runpod.io/v1/pods"


def runpod_get(api_key: str, pod_id: str) -> dict:
    req = urllib.request.Request(
        f"{RUNPOD_REST}/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def runpod_stop(api_key: str, pod_id: str) -> bool:
    req = urllib.request.Request(
        f"{RUNPOD_REST}/{pod_id}/stop",
        headers={"Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        print(f"[watchdog] stop_pod failed: {exc}", file=sys.stderr)
        return False


def ntfy(topic: str, msg: str) -> None:
    if not topic:
        return
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=msg.encode("utf-8"),
                method="POST",
            ),
            timeout=10,
        )
    except Exception:
        pass


def load_runpod_api_key() -> str:
    """Pick from env or ~/.runpod/config.toml."""
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        m = re.search(r"apikey\s*=\s*'([^']+)'", cfg.read_text())
        if m:
            return m.group(1)
    raise SystemExit("[watchdog] RUNPOD_API_KEY missing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", required=True)
    ap.add_argument("--cap", type=float, required=True, help="USD cap (e.g. 60)")
    ap.add_argument("--state-dir", default=".state/round2")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--warn-frac", type=float, default=0.95)
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    abort_marker = state_dir / "COST_CAP_HIT"
    spend_log = state_dir / "watchdog_spend.json"
    ntfy_topic = os.environ.get("NTFY_TOPIC", "").strip()

    api_key = load_runpod_api_key()
    warned = False
    running = True

    def _term(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    print(f"[watchdog] pod={args.pod_id} cap=${args.cap:.2f} interval={args.interval}s")
    while running:
        try:
            pod = runpod_get(api_key, args.pod_id)
        except Exception as exc:
            print(f"[watchdog] poll error: {exc}", file=sys.stderr)
            time.sleep(args.interval)
            continue

        uptime = pod.get("uptimeSeconds") or 0
        rate = pod.get("costPerHr") or 0.0
        spend = rate * uptime / 3600.0
        spend_log.write_text(json.dumps(
            {"pod_id": args.pod_id, "uptime_s": uptime, "rate_per_hr": rate,
             "spend_usd": round(spend, 4), "cap_usd": args.cap, "ts": int(time.time())}
        ))

        if spend >= args.cap:
            print(f"[watchdog] CAP HIT spend=${spend:.2f} >= ${args.cap:.2f}; stop_pod + abort")
            abort_marker.write_text(f"spend={spend:.2f}\ncap={args.cap:.2f}\n")
            runpod_stop(api_key, args.pod_id)
            ntfy(ntfy_topic, f"dalbit runpod CAP HIT spend=${spend:.2f} >= ${args.cap:.2f} pod={args.pod_id}")
            return 0
        if not warned and spend >= args.cap * args.warn_frac:
            warned = True
            print(f"[watchdog] WARN spend=${spend:.2f} >= {int(args.warn_frac*100)}% of ${args.cap:.2f}")
            ntfy(ntfy_topic, f"dalbit runpod 95% cap warn spend=${spend:.2f} cap=${args.cap:.2f} pod={args.pod_id}")

        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
