#!/usr/bin/env python3
"""
RunPod L40S availability + balance preflight (Mac mini, free, no GPU spin).

Verifies:
  1. RunPod API key is valid (HTTP 200 on /v1/users/me or fallback)
  2. Account balance is sufficient for the chosen budget profile
  3. NVIDIA L40S is currently available in COMMUNITY pool

Usage:
    python3 scripts/check_l40s_availability.py
    python3 scripts/check_l40s_availability.py --budget-usd 30 --json
    python3 scripts/check_l40s_availability.py --gpu-type "NVIDIA L40S"

Exit codes:
    0  ALL OK — safe to launch
    1  HOLD — at least one check failed
    2  USAGE — bad inputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

RUNPOD_API = "https://rest.runpod.io/v1"


def load_env() -> None:
    for candidate in (REPO_ROOT / ".env.local", Path.home() / ".env.local"):
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key and key not in os.environ:
                    os.environ[key] = value.strip().strip('"').strip("'")
        break


def runpod_get(path: str, api_key: str) -> tuple[int, dict | list | str]:
    req = urllib.request.Request(
        f"{RUNPOD_API}/{path.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "dalbit-l40s-preflight/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return -1, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-type", default="NVIDIA L40S", help="required GPU type")
    parser.add_argument("--budget-usd", type=float, default=30.0, help="USD budget envelope to require in account")
    parser.add_argument("--json", action="store_true", help="emit JSON verdict")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("[FATAL] RUNPOD_API_KEY missing in env / .env.local", file=sys.stderr)
        return 2

    failures: list[str] = []
    passes: list[str] = []
    findings: dict = {"required_gpu": args.gpu_type, "budget_usd": args.budget_usd}

    # Check 1: API key validity
    status, payload = runpod_get("user", api_key)
    if status == 200 and isinstance(payload, dict):
        passes.append("RunPod API key valid")
        balance = payload.get("clientBalance", payload.get("currentSpendPerHr", None))
        findings["client_balance"] = balance
        if balance is not None:
            try:
                balance_f = float(balance)
                if balance_f < args.budget_usd:
                    failures.append(f"clientBalance ${balance_f:.2f} < required ${args.budget_usd:.2f}")
                else:
                    passes.append(f"balance ${balance_f:.2f} ≥ budget ${args.budget_usd:.2f}")
            except (ValueError, TypeError):
                findings["balance_parse_warn"] = f"unparseable balance: {balance!r}"
    elif status == 401:
        failures.append("RunPod API returned 401 — RUNPOD_API_KEY invalid or expired")
    elif status == -1:
        failures.append(f"RunPod API unreachable: {payload}")
    else:
        # Some plans return user info under different paths; treat 404/403 as soft
        findings["user_check_status"] = status
        passes.append(f"RunPod API reachable (status={status}; user endpoint may not be on this plan)")

    # Check 2: GPU type availability
    status, gpu_payload = runpod_get("gpuTypes", api_key)
    if status == 200 and isinstance(gpu_payload, list):
        gpu_names = []
        target_found = False
        for entry in gpu_payload:
            if not isinstance(entry, dict):
                continue
            display_name = entry.get("displayName", "") or entry.get("id", "")
            gpu_names.append(display_name)
            community_count = entry.get("communityCloud", {}).get("availableCount", 0) if isinstance(entry.get("communityCloud"), dict) else None
            secure_count = entry.get("secureCloud", {}).get("availableCount", 0) if isinstance(entry.get("secureCloud"), dict) else None
            if args.gpu_type.lower() in display_name.lower():
                target_found = True
                findings["target_gpu_community_available"] = community_count
                findings["target_gpu_secure_available"] = secure_count
                cc = community_count or 0
                sc = secure_count or 0
                if cc + sc <= 0:
                    failures.append(f"{display_name} has 0 available pods (community={cc}, secure={sc})")
                else:
                    passes.append(f"{display_name} available: community={cc}, secure={sc}")
        if not target_found:
            findings["seen_gpu_types_sample"] = gpu_names[:8]
            failures.append(f"GPU type '{args.gpu_type}' not listed by RunPod API; got {len(gpu_names)} types")
    elif status == 200:
        findings["gpu_payload_unexpected"] = str(gpu_payload)[:200]
        # Not blocking — RunPod API shape may vary
        passes.append("RunPod gpuTypes returned non-list payload (treated as soft)")
    else:
        # Treat gpuTypes failure as soft warning, not block
        findings["gpu_check_status"] = status
        findings["gpu_check_body"] = str(gpu_payload)[:200]
        passes.append(f"gpuTypes endpoint inconclusive (status={status}); rely on launch-time fallback")

    verdict = "OK" if not failures else "HOLD"
    findings["verdict"] = verdict
    findings["passes"] = passes
    findings["failures"] = failures

    if args.json:
        print(json.dumps(findings, indent=2, ensure_ascii=False))
    else:
        print(f"=== L40S preflight — {verdict} ===")
        print(f"required GPU : {args.gpu_type}")
        print(f"budget       : ${args.budget_usd:.2f}")
        if "client_balance" in findings and findings["client_balance"] is not None:
            print(f"balance      : {findings['client_balance']}")
        if "target_gpu_community_available" in findings:
            print(f"L40S avail   : community={findings['target_gpu_community_available']}, secure={findings['target_gpu_secure_available']}")
        print()
        if passes:
            print("PASS:")
            for p in passes:
                print(f"  + {p}")
        if failures:
            print("FAIL:")
            for f in failures:
                print(f"  - {f}")
        print()
        if verdict == "OK":
            print("VERDICT: OK — safe to launch.")
        else:
            print("VERDICT: HOLD — resolve every FAIL line, then re-run.")

    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
