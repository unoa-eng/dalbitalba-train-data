#!/usr/bin/env python3
"""
Smoke -> budget30 promotion gate.

Reads runs/latest-train.json (written by chain_train.sh persist_run_artifacts),
or accepts equivalent CLI overrides, verifies the HF CPT adapter exists, and
prints PROMOTE or HOLD with reasons.

Usage:
    python3 scripts/check_smoke_promotion.py             # default: read latest-train.json
    python3 scripts/check_smoke_promotion.py --json      # machine-readable
    python3 scripts/check_smoke_promotion.py --train-status done_ok --hf-cpt-repo UNOA/repo

Exit codes:
    0  PROMOTE  (safe to launch the next stage)
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
LATEST_TRAIN = REPO_ROOT / "runs" / "latest-train.json"

REQUIRED_HF_FILES = (
    "cpt-lora/adapter_model.safetensors",
    "adapter_model.safetensors",
)

TRAINER_STATE_CANDIDATES = (
    "trainer_state.json",
    "cpt-lora/trainer_state.json",
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
    """List files in a HF repo via public API. Returns [] on any error."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true"
    headers = {"User-Agent": "dalbit-promotion-check/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read())
        return [item.get("path", "") for item in payload if isinstance(item, dict)]
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return []


def hf_fetch_json(repo_id: str, filename: str, token: str | None) -> dict | None:
    """Download a JSON file from a HF repo. Returns None on any error."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"User-Agent": "dalbit-promotion-check/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


def load_latest_payload(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[FATAL] cannot parse {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest-train",
        default=str(LATEST_TRAIN),
        help="Path to runs/latest-train.json (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-cpt-repo",
        default="",
        help="Override HF CPT repo (else read from latest-train.json)",
    )
    parser.add_argument(
        "--hf-sft-repo",
        default="",
        help="Override HF SFT repo (else read from latest-train.json)",
    )
    parser.add_argument(
        "--train-status",
        default="",
        help="Override train status (else read from latest-train.json)",
    )
    parser.add_argument(
        "--branch",
        default="",
        help="Override run branch for local artifact checks",
    )
    parser.add_argument(
        "--require-sft",
        action="store_true",
        help="Also require a non-empty hf_repo_sft (smoke profile uses SFT; budget30 does not)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON verdict")
    args = parser.parse_args()

    load_env()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    reasons: list[str] = []
    ok_checks: list[str] = []

    latest_path = Path(args.latest_train)
    latest = load_latest_payload(latest_path) if latest_path.exists() else None
    if latest is None and not (args.train_status and args.hf_cpt_repo):
        print(f"[FATAL] latest-train pointer not found: {latest_path}", file=sys.stderr)
        print(
            "  -> either run a training pod first, or pass --train-status and --hf-cpt-repo",
            file=sys.stderr,
        )
        return 2

    latest = latest or {}
    branch = (args.branch or latest.get("branch", "") or "").strip()
    status = (args.train_status or latest.get("status", "") or "").strip()
    hf_cpt = (args.hf_cpt_repo or latest.get("hf_repo_cpt", "") or "").strip()
    hf_sft = (args.hf_sft_repo or latest.get("hf_repo_sft", "") or "").strip()
    timestamp = latest.get("timestamp", "") if latest else ""

    # Check 1: status must be done_ok
    if status == "done_ok":
        source = str(latest_path) if latest_path.exists() else "cli"
        ok_checks.append(f"train status=done_ok ({branch or 'branch:unknown'}, source={source})")
    else:
        reasons.append(f"train status='{status}' (expected done_ok); branch={branch or 'unknown'}")

    # Check 2: HF CPT adapter must exist
    if not hf_cpt:
        reasons.append("hf_repo_cpt missing in latest-train.json — supply --hf-cpt-repo to override")
    else:
        files = hf_list_files(hf_cpt, hf_token)
        if not files:
            reasons.append(f"hf_repo_cpt={hf_cpt} -> file list empty (auth or 404). check HF_TOKEN")
        else:
            adapter_present = any(needle in path for path in files for needle in REQUIRED_HF_FILES)
            if adapter_present:
                ok_checks.append(f"hf_cpt adapter present at {hf_cpt}")
            else:
                reasons.append(f"hf_cpt={hf_cpt} has {len(files)} files but no adapter_model.safetensors")

    # Check 2.5: trainer_state.json must show global_step >= max_steps
    # This is the mechanical defense against the 0618 partial-CPT trap (step 2700/5775
    # promoted as if it were complete).
    if hf_cpt:
        trainer_state = None
        resolved_trainer = None
        for cand in TRAINER_STATE_CANDIDATES:
            trainer_state = hf_fetch_json(hf_cpt, cand, hf_token)
            if trainer_state is not None:
                resolved_trainer = cand
                break
        if trainer_state is None:
            reasons.append(
                f"hf_cpt={hf_cpt} has no trainer_state.json — cannot verify training completed"
            )
        else:
            global_step = trainer_state.get("global_step", 0)
            max_steps = trainer_state.get("max_steps", 0)
            epoch = trainer_state.get("epoch", 0)
            if max_steps > 0 and global_step < max_steps:
                reasons.append(
                    f"trainer_state.json @ {resolved_trainer}: global_step={global_step} "
                    f"< max_steps={max_steps} (only {global_step / max_steps:.1%} complete) — "
                    f"refuse to promote partial training"
                )
            elif global_step <= 0:
                reasons.append(f"trainer_state.json: global_step={global_step} (training never started)")
            else:
                ok_checks.append(
                    f"trainer_state.json: global_step={global_step}, max_steps={max_steps}, epoch={epoch:.2f}"
                )

    # Check 3 (optional): SFT adapter must exist for smoke
    if args.require_sft:
        if not hf_sft:
            reasons.append("hf_repo_sft empty — smoke profile must produce an SFT adapter")
        else:
            sft_files = hf_list_files(hf_sft, hf_token)
            if not sft_files:
                reasons.append(f"hf_repo_sft={hf_sft} -> file list empty (auth or 404)")
            else:
                sft_adapter = any("adapter_model.safetensors" in p for p in sft_files)
                if sft_adapter:
                    ok_checks.append(f"hf_sft adapter present at {hf_sft}")
                else:
                    reasons.append(f"hf_sft={hf_sft} has {len(sft_files)} files but no adapter_model.safetensors")

    # Check 4: branch artifacts (DONE.txt + manifest)
    if branch:
        branch_dir = REPO_ROOT / "runs" / branch
        if branch_dir.exists():
            done_file = branch_dir / "DONE.txt"
            manifest_file = branch_dir / "manifest.json"
            if done_file.exists():
                done_txt = done_file.read_text(encoding="utf-8", errors="replace").strip()
                if "done_ok" in done_txt:
                    ok_checks.append(f"DONE.txt = done_ok ({branch_dir})")
                else:
                    reasons.append(f"DONE.txt content does not include done_ok: {done_txt[:80]}")
            else:
                reasons.append(f"DONE.txt missing in {branch_dir}")
            if manifest_file.exists():
                ok_checks.append(f"manifest.json present in {branch_dir}")
            else:
                reasons.append(f"manifest.json missing in {branch_dir}")
        else:
            reasons.append(
                f"runs/{branch}/ directory not present locally — `git fetch origin {branch}` "
                "and `git checkout {branch} -- runs/{branch}` to inspect"
            )
    else:
        ok_checks.append("local branch artifact check skipped (no branch provided)")

    promote = len(reasons) == 0
    verdict = "PROMOTE" if promote else "HOLD"

    if args.json:
        print(json.dumps(
            {
                "verdict": verdict,
                "branch": branch,
                "status": status,
                "hf_cpt": hf_cpt,
                "hf_sft": hf_sft,
                "timestamp": timestamp,
                "checks_passed": ok_checks,
                "checks_failed": reasons,
            },
            indent=2,
            ensure_ascii=False,
        ))
    else:
        print(f"=== smoke promotion check — {verdict} ===")
        print(f"branch    : {branch}")
        print(f"status    : {status}")
        print(f"hf_cpt    : {hf_cpt or '(none)'}")
        print(f"hf_sft    : {hf_sft or '(none)'}")
        print(f"timestamp : {timestamp}")
        print()
        if ok_checks:
            print("PASS:")
            for c in ok_checks:
                print(f"  + {c}")
        if reasons:
            print("FAIL:")
            for r in reasons:
                print(f"  - {r}")
        print()
        if promote:
            print("VERDICT: PROMOTE — safe to launch the next stage (budget30 or eval).")
        else:
            print("VERDICT: HOLD — do not launch the next paid stage.")
            print("Resolve every FAIL line above. Re-run this script after fixing.")

    return 0 if promote else 1


if __name__ == "__main__":
    sys.exit(main())
