#!/usr/bin/env python3
"""Stop RunPod pods that have been running longer than --max-age-hours.

Default behavior: list RUNNING pods, compute age from createdAt, stop any
that exceed the threshold AND are not referenced by .state/train_pod_state.json.

Usage:
    python3 scripts/abandoned_pod_sweeper.py --dry-run
    python3 scripts/abandoned_pod_sweeper.py --max-age-hours 36
    python3 scripts/abandoned_pod_sweeper.py --max-age-hours 24 --verbose
    python3 scripts/abandoned_pod_sweeper.py --include-stopped --dry-run
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

API_BASE = "https://rest.runpod.io/v1/pods"
STATE_FILE = Path(__file__).resolve().parent.parent / ".state" / "train_pod_state.json"

STALE_STATUSES = {"RUNNING", "STARTING"}
STOPPED_STATUSES = {"EXITED", "STOPPED", "TERMINATED"}


def list_pods(api_key: str) -> list[dict]:
    """Fetch all pods for the account via RunPod REST API."""
    req = urllib.request.Request(
        API_BASE,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def is_active(pod_id: str) -> bool:
    """Return True if pod_id is referenced in the local state file (active training pod)."""
    if not STATE_FILE.exists():
        return False
    try:
        st = json.loads(STATE_FILE.read_text())
        active_ids: list = st.get("active_pod_ids") or []
        primary_id = st.get("pod_id")
        if primary_id:
            active_ids = active_ids + [primary_id]
        return pod_id in active_ids
    except Exception:
        return False


def stop_pod(api_key: str, pod_id: str) -> bool:
    """Send stop request to RunPod REST API. Returns True on success."""
    url = f"{API_BASE}/{pod_id}/stop"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status < 300
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} stopping {pod_id}: {e.reason}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"  URL error stopping {pod_id}: {e.reason}", file=sys.stderr)
        return False


def notify(msg: str) -> None:
    """Send ntfy notification if NTFY_TOPIC is set."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=msg.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "Title": "abandoned-pod-sweeper",
                    "Priority": "high",
                },
            ),
            timeout=10,
        )
    except Exception as exc:
        # Notification failure is non-fatal
        print(f"[ntfy] send failed: {exc}", file=sys.stderr)


def parse_created_at(raw: str) -> datetime | None:
    """Parse ISO-8601 / RFC-3339 timestamp from RunPod API response."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stop RunPod pods older than --max-age-hours."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stale candidates but do not stop them.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=36.0,
        metavar="N",
        help="Age threshold in hours (default: 36).",
    )
    parser.add_argument(
        "--include-stopped",
        action="store_true",
        help="Also report stopped/exited pods beyond the threshold (informational only; never re-stopped).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable is not set.", file=sys.stderr)
        return 2

    if args.verbose:
        print(f"[sweeper] fetching pods from {API_BASE} ...")

    try:
        pods = list_pods(api_key)
    except urllib.error.HTTPError as e:
        print(f"ERROR: RunPod API returned HTTP {e.code}: {e.reason}", file=sys.stderr)
        return 2
    except urllib.error.URLError as e:
        print(f"ERROR: RunPod API unreachable: {e.reason}", file=sys.stderr)
        return 2

    if args.verbose:
        print(f"[sweeper] {len(pods)} pod(s) retrieved.")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=args.max_age_hours)

    stale: list[dict] = []
    informational: list[dict] = []

    for pod in pods:
        pod_id: str = pod.get("id", "")
        status: str = (pod.get("status") or "").upper()
        name: str = pod.get("name") or ""
        cost_per_hr: float = pod.get("costPerHr") or 0.0
        created_raw: str = pod.get("createdAt") or ""

        created = parse_created_at(created_raw)
        if created is None:
            if args.verbose:
                print(f"  [skip] {pod_id} ({name}): cannot parse createdAt={created_raw!r}")
            continue

        age_h = (now - created).total_seconds() / 3600

        if status in STALE_STATUSES:
            if created > cutoff:
                if args.verbose:
                    print(f"  [ok]   {pod_id} ({name}) status={status} age={age_h:.1f}h < threshold")
                continue
            if is_active(pod_id):
                if args.verbose:
                    print(f"  [active] {pod_id} ({name}) referenced in state file — skipping")
                continue
            stale.append({
                "id": pod_id,
                "name": name,
                "status": status,
                "age_h": round(age_h, 1),
                "cost_per_hr": cost_per_hr,
                "created_at": created_raw,
            })
        elif args.include_stopped and status in STOPPED_STATUSES and created < cutoff:
            informational.append({
                "id": pod_id,
                "name": name,
                "status": status,
                "age_h": round(age_h, 1),
            })

    # Report informational stopped pods
    if args.include_stopped and informational:
        print(f"[info] {len(informational)} stopped/exited pod(s) older than {args.max_age_hours}h:")
        for p in informational:
            print(f"  - {p['id']} name={p['name']} status={p['status']} age={p['age_h']}h")

    if not stale:
        if args.verbose:
            print(f"[sweeper] no stale RUNNING pods found (threshold={args.max_age_hours}h).")
        return 0

    # Announce stale candidates
    summary = (
        f"abandoned-pod-sweeper: {len(stale)} stale pod(s) running >{args.max_age_hours}h"
    )
    print(summary)
    for s in stale:
        cost_note = f"  cost=${s['cost_per_hr']:.3f}/hr" if s["cost_per_hr"] else ""
        print(f"  - {s['id']} name={s['name']} age={s['age_h']}h{cost_note}")

    # Send ntfy alert
    ntfy_lines = [summary] + [
        f"  - {s['id']} age={s['age_h']}h cost=${s['cost_per_hr']:.3f}/hr"
        for s in stale
    ]
    notify("\n".join(ntfy_lines))

    if args.dry_run:
        print("[dry-run] no action taken.")
        return 0

    # Stop stale pods
    errors = 0
    for s in stale:
        ok = stop_pod(api_key, s["id"])
        status_str = "OK" if ok else "FAIL"
        print(f"stop {s['id']} ({s['name']}): {status_str}")
        if not ok:
            errors += 1

    if errors:
        print(f"[sweeper] {errors}/{len(stale)} stop(s) failed.", file=sys.stderr)
        return 2

    print(f"[sweeper] {len(stale)} pod(s) stopped successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
