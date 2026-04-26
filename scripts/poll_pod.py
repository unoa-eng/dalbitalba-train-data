#!/usr/bin/env python3
"""Poll a RunPod pod, output status JSON; exit 0 running, 1 exited, 2 error."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_env() -> None:
    for candidate in (Path(".env.local"), Path.home() / ".env.local"):
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key, value.strip().strip('"').strip("'"))
        return


def main() -> int:
    load_env()
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: poll_pod.py <pod_id>"}))
        return 2

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print(json.dumps({"error": "missing RUNPOD_API_KEY"}))
        return 2

    pod_id = sys.argv[1]
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print(json.dumps({"pod_id": pod_id, "error": f"HTTP {exc.code}"}))
        return 2
    except Exception as exc:
        print(json.dumps({"pod_id": pod_id, "error": type(exc).__name__}))
        return 2

    status = data.get("desiredStatus")
    print(
        json.dumps(
            {
                "pod_id": pod_id,
                "status": status,
                "last_change": data.get("lastStatusChange"),
                "name": data.get("name"),
            }
        )
    )
    return 0 if status == "RUNNING" else 1


if __name__ == "__main__":
    sys.exit(main())
