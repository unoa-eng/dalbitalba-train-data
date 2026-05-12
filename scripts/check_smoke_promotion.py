#!/usr/bin/env python3
"""
Smoke -> budget30 promotion gate.

Reads the latest train pointer written by the active chain, verifies the
required HF artifacts exist, and prints PROMOTE or HOLD with reasons.

Usage:
    python3 scripts/check_smoke_promotion.py             # auto-detect classic/round2 pointer
    python3 scripts/check_smoke_promotion.py --json      # machine-readable

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
LATEST_CLASSIC = REPO_ROOT / "runs" / "latest-train.json"
LATEST_ROUND2 = REPO_ROOT / "runs" / "latest-round2-train.json"

CLASSIC_REQUIRED_HF_FILES = (
    "cpt-lora/adapter_model.safetensors",
    "adapter_model.safetensors",
)

CLASSIC_TRAINER_STATE_CANDIDATES = (
    "trainer_state.json",
    "cpt-lora/trainer_state.json",
)

ROUND2_CPT_REQUIRED_HF_FILES = (
    "round2-phase1-cpt-lora/adapter_model.safetensors",
    "round2-phase2-cpt-lora/adapter_model.safetensors",
)

ROUND2_SFT_REQUIRED_HF_FILES = (
    "round2-phase3-sft-lora/adapter_model.safetensors",
)

ROUND2_CPT_TRAINER_STATE_FILES = (
    "round2-phase1-cpt-lora/trainer_state.json",
    "round2-phase2-cpt-lora/trainer_state.json",
)

ROUND2_SFT_TRAINER_STATE_FILES = (
    "round2-phase3-sft-lora/trainer_state.json",
)

ROUND2_EVAL_FILES = (
    "eval/phase5-eval-v2.json",
    "DONE.txt",
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


def any_path_contains(files: list[str], needles: tuple[str, ...]) -> bool:
    return any(needle in path for path in files for needle in needles)


def missing_exact(files: list[str], required: tuple[str, ...]) -> list[str]:
    available = set(files)
    return [path for path in required if path not in available]


def validate_trainer_state(
    repo_id: str,
    path: str,
    token: str | None,
) -> tuple[bool, str]:
    trainer_state = hf_fetch_json(repo_id, path, token)
    if trainer_state is None:
        return False, f"{repo_id} missing trainer_state.json at {path}"
    try:
        global_step = int(trainer_state.get("global_step") or 0)
        max_steps = int(trainer_state.get("max_steps") or 0)
    except (TypeError, ValueError):
        return False, f"trainer_state.json @ {path}: invalid global_step/max_steps"
    epoch = trainer_state.get("epoch", 0)
    if max_steps <= 0:
        return False, f"trainer_state.json @ {path}: max_steps={max_steps} (cannot verify completion)"
    if global_step < max_steps:
        return (
            False,
            f"trainer_state.json @ {path}: global_step={global_step} < max_steps={max_steps} "
            f"(only {global_step / max_steps:.1%} complete)",
        )
    return True, f"trainer_state.json @ {path}: global_step={global_step}, max_steps={max_steps}, epoch={epoch}"


def resolve_default_latest(mode: str) -> Path:
    if mode == "classic":
        return LATEST_CLASSIC
    if mode == "round2":
        return LATEST_ROUND2
    candidates = [path for path in (LATEST_ROUND2, LATEST_CLASSIC) if path.exists()]
    if not candidates:
        # Prefer the active round2 pointer in the error path. Returning the
        # classic path here makes a missing-smoke failure look like the wrong
        # training chain was selected.
        return LATEST_ROUND2
    return max(candidates, key=lambda path: path.stat().st_mtime)


def infer_mode(latest_path: Path, latest_payload: dict[str, object], requested_mode: str) -> str:
    if requested_mode in {"classic", "round2"}:
        return requested_mode
    if latest_path.name == "latest-round2-train.json" or "hf_repo_round2" in latest_payload:
        return "round2"
    return "classic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest-train",
        default="",
        help="Path to latest train pointer (default: auto-detect classic/round2)",
    )
    parser.add_argument(
        "--hf-cpt-repo",
        default="",
        help="Override HF primary repo (classic: CPT repo, round2: round2 repo)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "classic", "round2"),
        default="auto",
        help="Promotion gate mode (default: auto-detect from latest pointer)",
    )
    parser.add_argument(
        "--require-sft",
        action="store_true",
        help="Also require SFT artifacts (classic: hf_repo_sft, round2: phase3 SFT adapter)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON verdict")
    args = parser.parse_args()

    load_env()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    reasons: list[str] = []
    ok_checks: list[str] = []

    latest_path = Path(args.latest_train) if args.latest_train else resolve_default_latest(args.mode)
    if not latest_path.exists():
        print(f"[FATAL] latest-train pointer not found: {latest_path}", file=sys.stderr)
        if not args.latest_train and args.mode == "auto":
            print(
                f"  -> checked: {LATEST_ROUND2} and {LATEST_CLASSIC}",
                file=sys.stderr,
            )
        print("  -> run a training pod first; the pointer is created by the active chain script", file=sys.stderr)
        return 2

    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[FATAL] cannot parse {latest_path}: {exc}", file=sys.stderr)
        return 2

    mode = infer_mode(latest_path, latest, args.mode)
    branch = latest.get("branch", "") or ""
    status = latest.get("status", "") or ""
    if mode == "round2":
        hf_primary = (args.hf_cpt_repo or latest.get("hf_repo_round2", "") or "").strip()
        hf_secondary = ""
        required_hf_files = ROUND2_CPT_REQUIRED_HF_FILES + (
            ROUND2_SFT_REQUIRED_HF_FILES if args.require_sft else ()
        ) + ROUND2_EVAL_FILES
        required_trainer_state_files = ROUND2_CPT_TRAINER_STATE_FILES + (
            ROUND2_SFT_TRAINER_STATE_FILES if args.require_sft else ()
        )
    else:
        hf_primary = (args.hf_cpt_repo or latest.get("hf_repo_cpt", "") or "").strip()
        hf_secondary = (latest.get("hf_repo_sft", "") or "").strip()
        required_hf_files = CLASSIC_REQUIRED_HF_FILES
        required_trainer_state_files = ()
    timestamp = latest.get("timestamp", "")

    # Check 1: status must be done_ok
    if status == "done_ok":
        ok_checks.append(f"{mode} latest-train status=done_ok ({branch})")
    else:
        reasons.append(f"{mode} latest-train status='{status}' (expected done_ok); branch={branch}")

    # Check 2: primary HF repo must exist and contain adapters.
    if not hf_primary:
        missing_key = "hf_repo_round2" if mode == "round2" else "hf_repo_cpt"
        reasons.append(f"{missing_key} missing in {latest_path.name} — supply --hf-cpt-repo to override")
    else:
        files = hf_list_files(hf_primary, hf_token)
        if not files:
            reasons.append(f"primary repo {hf_primary} -> file list empty (auth or 404). check HF_TOKEN")
        else:
            if mode == "round2":
                missing = missing_exact(files, required_hf_files)
            else:
                missing = [] if any_path_contains(files, required_hf_files) else ["/".join(required_hf_files)]
            if not missing:
                ok_checks.append(f"{mode} required artifacts present at {hf_primary}")
            else:
                reasons.append(f"{hf_primary} missing required {mode} artifacts: {missing}")

    # Check 2.5: trainer_state.json must show global_step >= max_steps
    # This is the mechanical defense against the 0618 partial-CPT trap (step 2700/5775
    # promoted as if it were complete).
    if hf_primary:
        if mode == "round2":
            for path in required_trainer_state_files:
                ok_state, message = validate_trainer_state(hf_primary, path, hf_token)
                if ok_state:
                    ok_checks.append(message)
                else:
                    reasons.append(message)
        else:
            trainer_state = None
            resolved_trainer = None
            for cand in CLASSIC_TRAINER_STATE_CANDIDATES:
                trainer_state = hf_fetch_json(hf_primary, cand, hf_token)
                if trainer_state is not None:
                    resolved_trainer = cand
                    break
            if trainer_state is None:
                reasons.append(
                    f"{hf_primary} has no expected trainer_state.json for {mode} — cannot verify training completed"
                )
            else:
                ok_state, message = validate_trainer_state(hf_primary, resolved_trainer or "", hf_token)
                if ok_state:
                    ok_checks.append(message)
                else:
                    reasons.append(message)

    # Check 3 (classic only): separate SFT adapter must exist for smoke
    if args.require_sft and mode == "classic":
        if not hf_secondary:
            reasons.append("hf_repo_sft empty — smoke profile must produce an SFT adapter")
        else:
            sft_files = hf_list_files(hf_secondary, hf_token)
            if not sft_files:
                reasons.append(f"hf_repo_sft={hf_secondary} -> file list empty (auth or 404)")
            else:
                sft_adapter = any("adapter_model.safetensors" in p for p in sft_files)
                if sft_adapter:
                    ok_checks.append(f"hf_sft adapter present at {hf_secondary}")
                else:
                    reasons.append(f"hf_repo_sft={hf_secondary} has {len(sft_files)} files but no adapter_model.safetensors")

    # Check 4: branch artifacts (DONE.txt + manifest)
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

    promote = len(reasons) == 0
    verdict = "PROMOTE" if promote else "HOLD"

    if args.json:
        print(json.dumps(
            {
                "verdict": verdict,
                "mode": mode,
                "latest_pointer": str(latest_path),
                "branch": branch,
                "status": status,
                "hf_primary": hf_primary,
                "hf_secondary": hf_secondary,
                "timestamp": timestamp,
                "checks_passed": ok_checks,
                "checks_failed": reasons,
            },
            indent=2,
            ensure_ascii=False,
        ))
    else:
        print(f"=== smoke promotion check — {verdict} ===")
        print(f"mode      : {mode}")
        print(f"latest    : {latest_path}")
        print(f"branch    : {branch}")
        print(f"status    : {status}")
        if mode == "round2":
            print(f"hf_round2 : {hf_primary or '(none)'}")
        else:
            print(f"hf_cpt    : {hf_primary or '(none)'}")
            print(f"hf_sft    : {hf_secondary or '(none)'}")
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
