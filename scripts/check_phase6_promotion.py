#!/usr/bin/env python3
"""
Phase6 eval -> promotion gate.

Reads runs/latest-eval.json plus runs/<branch>/metrics.json by default, or
accepts explicit CLI overrides, then enforces:
  - phase6 gate verdict == PASS
  - korean_retention_ppl is present and <= threshold
  - adapter repo contains adapter weights

Usage:
    python3 scripts/check_phase6_promotion.py
    python3 scripts/check_phase6_promotion.py --json
    python3 scripts/check_phase6_promotion.py --metrics /tmp/metrics.json --adapter-repo UNOA/repo

Exit codes:
    0  PROMOTE  (safe to promote after eval)
    1  HOLD     (do not promote; see printed reasons)
    2  USAGE    (missing inputs)
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
LATEST_EVAL = REPO_ROOT / "runs" / "latest-eval.json"

REQUIRED_HF_FILES = (
    "sft-lora/adapter_model.safetensors",
    "cpt-lora/adapter_model.safetensors",
    "adapter_model.safetensors",
)


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


def hf_list_files(repo_id: str, token: str | None) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true"
    headers = {"User-Agent": "dalbit-phase6-promotion/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read())
        return [item.get("path", "") for item in payload if isinstance(item, dict)]
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return []


def load_json(path: Path, label: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        print(f"[FATAL] {label} not found: {path}", file=sys.stderr)
        raise SystemExit(2) from exc
    except json.JSONDecodeError as exc:
        print(f"[FATAL] cannot parse {label} {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest-eval",
        default=str(LATEST_EVAL),
        help="Path to runs/latest-eval.json (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Override metrics.json path (else derive from latest-eval branch)",
    )
    parser.add_argument(
        "--adapter-repo",
        default="",
        help="Override adapter repo to verify (else read from latest-eval.json)",
    )
    parser.add_argument(
        "--max-korean-retention-ppl",
        type=float,
        default=1.5,
        help="Maximum allowed korean_retention_ppl ratio (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON verdict")
    args = parser.parse_args()

    load_env()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    latest_path = Path(args.latest_eval)
    latest = load_json(latest_path, "latest-eval pointer") if latest_path.exists() else None
    if latest is None and not (args.metrics and args.adapter_repo):
        print(f"[FATAL] latest-eval pointer not found: {latest_path}", file=sys.stderr)
        print(
            "  -> either provide runs/latest-eval.json, or pass both --metrics and --adapter-repo",
            file=sys.stderr,
        )
        return 2

    latest = latest or {}
    branch = str(latest.get("branch", "") or "").strip()
    status = str(latest.get("status", "") or "").strip()
    adapter_repo = (args.adapter_repo or latest.get("hf_adapter_repo", "") or "").strip()
    timestamp = str(latest.get("timestamp", "") or "").strip()
    metrics_path = Path(args.metrics) if args.metrics else (REPO_ROOT / "runs" / branch / "metrics.json" if branch else None)

    reasons: list[str] = []
    ok_checks: list[str] = []
    gate_verdict = ""
    gate_violations: list[str] = []
    retention_value: float | None = None

    if status:
        if status == "done_ok":
            ok_checks.append(f"latest-eval status=done_ok ({branch or 'branch:unknown'})")
        else:
            reasons.append(f"latest-eval status='{status}' (expected done_ok); branch={branch or 'unknown'}")
    else:
        ok_checks.append("latest-eval status check skipped (metrics override mode)")

    if metrics_path is None:
        print("[FATAL] cannot resolve metrics path; pass --metrics explicitly", file=sys.stderr)
        return 2
    if not metrics_path.exists():
        reasons.append(f"metrics.json missing: {metrics_path}")
        report = {}
    else:
        report = load_json(metrics_path, "metrics report")
        gate_verdict = str(report.get("gate", {}).get("verdict", "") or "").upper()
        gate_violations = [
            str(item)
            for item in report.get("gate", {}).get("violations", [])
            if str(item).strip()
        ]
        if gate_verdict == "PASS":
            ok_checks.append(f"phase6 gate PASS ({metrics_path})")
        else:
            if gate_violations:
                reasons.append(
                    "phase6 gate verdict="
                    f"{gate_verdict or 'UNKNOWN'} ({'; '.join(gate_violations)})"
                )
            else:
                reasons.append(f"phase6 gate verdict={gate_verdict or 'UNKNOWN'}")

        metrics = report.get("metrics", {})
        details = report.get("details", {})
        raw_retention = metrics.get("korean_retention_ppl")
        retention_details = details.get("korean_retention_ppl", {})
        if isinstance(raw_retention, (int, float)):
            retention_value = float(raw_retention)
            if retention_value <= args.max_korean_retention_ppl:
                ok_checks.append(
                    "korean_retention_ppl="
                    f"{retention_value:.4f} <= {args.max_korean_retention_ppl:.2f}"
                )
            else:
                reasons.append(
                    "korean_retention_ppl="
                    f"{retention_value:.4f} > {args.max_korean_retention_ppl:.2f}"
                )
        else:
            detail_status = retention_details.get("status", "missing")
            detail_reason = retention_details.get("reason", "")
            suffix = f" reason={detail_reason}" if detail_reason else ""
            reasons.append(
                f"korean_retention_ppl unavailable (status={detail_status}{suffix})"
            )

    if not adapter_repo:
        reasons.append("adapter repo missing — pass --adapter-repo or populate runs/latest-eval.json")
    else:
        files = hf_list_files(adapter_repo, hf_token)
        if not files:
            reasons.append(
                f"adapter repo {adapter_repo} -> file list empty (auth, 404, or no HF_TOKEN)"
            )
        else:
            adapter_present = any(
                needle in path
                for path in files
                for needle in REQUIRED_HF_FILES
            )
            if adapter_present:
                ok_checks.append(f"adapter weights present at {adapter_repo}")
            else:
                reasons.append(
                    f"adapter repo {adapter_repo} has {len(files)} files but no adapter_model.safetensors"
                )

    verdict = "PROMOTE" if not reasons else "HOLD"
    payload = {
        "verdict": verdict,
        "branch": branch,
        "status": status,
        "adapter_repo": adapter_repo,
        "timestamp": timestamp,
        "metrics_path": str(metrics_path),
        "phase6_verdict": gate_verdict,
        "phase6_violations": gate_violations,
        "korean_retention_ppl": retention_value,
        "checks_passed": ok_checks,
        "checks_failed": reasons,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"=== phase6 promotion gate — {verdict} ===")
        print(f"branch                 : {branch or '(none)'}")
        print(f"status                 : {status or '(none)'}")
        print(f"adapter_repo           : {adapter_repo or '(none)'}")
        print(f"metrics_path           : {metrics_path}")
        print(f"timestamp              : {timestamp or '(none)'}")
        print(f"phase6 verdict         : {gate_verdict or '(unknown)'}")
        if retention_value is None:
            print("korean_retention_ppl   : (unavailable)")
        else:
            print(f"korean_retention_ppl   : {retention_value:.4f}")
        print()
        if ok_checks:
            print("PASS:")
            for check in ok_checks:
                print(f"  + {check}")
        if reasons:
            print("FAIL:")
            for reason in reasons:
                print(f"  - {reason}")
        print()
        if verdict == "PROMOTE":
            print("VERDICT: PROMOTE — eval gate passed and adapter exists.")
        else:
            print("VERDICT: HOLD — do not promote beyond phase6.")

    return 0 if verdict == "PROMOTE" else 1


if __name__ == "__main__":
    sys.exit(main())
