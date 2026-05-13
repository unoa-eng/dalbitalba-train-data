#!/usr/bin/env python3
"""Mac mini control-plane smoke loop.

This script is intentionally local-only: it does not launch RunPod and does not
load the 8B base model. It records reproducible artifacts for the paid-run gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
LOCAL_PYTHONWARNINGS = "ignore:urllib3 v2 only supports OpenSSL 1.1.1+"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _resolve_training_python() -> str:
    """Prefer a venv interpreter with transformers installed.

    macmini control-plane runs with system python3.14 which intentionally
    does not have transformers; transformers-dependent smoke steps must
    delegate to .venv. Falls back to sys.executable if no venv has it.
    """
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv-mlx" / "bin" / "python",
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            r = subprocess.run(
                [str(cand), "-c", "import transformers"],
                capture_output=True,
                timeout=10,
            )
            if r.returncode == 0:
                return str(cand)
        except Exception:
            continue
    return sys.executable


TRAINING_PYTHON = _resolve_training_python()


def run_check(name: str, cmd: list[str], run_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    existing_warnings = env.get("PYTHONWARNINGS", "").strip()
    env["PYTHONWARNINGS"] = (
        f"{LOCAL_PYTHONWARNINGS},{existing_warnings}"
        if existing_warnings
        else LOCAL_PYTHONWARNINGS
    )
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    stdout_path = run_dir / f"{name}.stdout.log"
    stderr_path = run_dir / f"{name}.stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_log": str(stdout_path.relative_to(REPO_ROOT)),
        "stderr_log": str(stderr_path.relative_to(REPO_ROOT)),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def parse_json_stdout(check: dict[str, Any]) -> dict[str, Any] | None:
    stdout = check.get("stdout_tail") or ""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Mac mini Smoke Loop",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Profile: `{payload['profile']}`",
        f"- Run dir: `{payload['run_dir']}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Return Code | Log |",
        "| --- | --- | ---: | --- |",
    ]
    for check in payload["checks"]:
        status = "PASS" if check["ok"] else "FAIL"
        lines.append(
            f"| {check['name']} | `{status}` | {check['returncode']} | `{check['stdout_log']}` |"
        )
    if payload.get("local_verification"):
        local = payload["local_verification"]
        lines.extend(
            [
                "",
                "## Local Verification",
                "",
                f"- Verdict: `{local.get('verdict')}`",
                f"- Report: `{local.get('report')}`",
            ]
        )
    if payload.get("train_eval_process"):
        process = payload["train_eval_process"]
        lines.extend(
            [
                "",
                "## Train/Eval Process",
                "",
                f"- State stdout log: `{process.get('stdout_log')}`",
                f"- Return code: `{process.get('returncode')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("paper8b", "budget30", "smoke"),
        default="paper8b",
        help="profile to verify; paper8b is the full no-feature-loss paid path",
    )
    args = parser.parse_args()
    # C5-3: propagate --profile so child scripts that read BUDGET_PROFILE pick it up.
    os.environ["BUDGET_PROFILE"] = args.profile

    run_dir = RUNS_DIR / f"macmini-smoke-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        ("bash_n_chain_train_round2", ["bash", "-n", "chain_train_round2.sh"]),
        ("bash_n_chain_train", ["bash", "-n", "chain_train.sh"]),
        (
            "py_compile",
            [
                sys.executable,
                "-m",
                "py_compile",
                "scripts/launch_train_pod.py",
                "scripts/train_eval_process.py",
                "scripts/local_verification_loop.py",
                "scripts/phase6_eval_v2.py",
                "scripts/phase6_generate.py",
                "scripts/round2_integrity_check.py",
                "scripts/sft_format_smoke_test.py",
                "scripts/prelaunch_research_check.py",
                "train_cpt.py",
                "train_sft.py",
                "train_orpo.py",
            ],
        ),
        ("tokenizer_added_tokens", [TRAINING_PYTHON, "scripts/check_adapter_integrity.py", "--tokenizer-only"]),
        ("round2_integrity", [sys.executable, "scripts/round2_integrity_check.py"]),
        ("prelaunch_research", [TRAINING_PYTHON, "scripts/prelaunch_research_check.py"]),
        ("sft_format_smoke", [TRAINING_PYTHON, "scripts/sft_format_smoke_test.py"]),
        (
            f"local_verification_{args.profile}",
            [TRAINING_PYTHON, "scripts/local_verification_loop.py", "--strict", "--profile", args.profile],
        ),
        (
            "train_eval_process_dry_run",
            [
                TRAINING_PYTHON,
                "scripts/train_eval_process.py",
                "--dry-run",
                "--sample-rows",
                "40",
                "--profile",
                args.profile,
            ],
        ),
        (
            "check_data_paths",
            [
                sys.executable,
                "scripts/check_data_paths.py",
                "--profile",
                args.profile,
                "--strict",
            ],
        ),
        ("macmini_train_simul", ["bash", "scripts/macmini_local_train_simul.sh"]),
    ]

    results = [run_check(name, cmd, run_dir) for name, cmd in checks]
    verifier = next((item for item in results if item["name"] == f"local_verification_{args.profile}"), None)
    train_eval = next((item for item in results if item["name"] == "train_eval_process_dry_run"), None)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "profile": args.profile,
        "verdict": "PASS" if all(item["ok"] for item in results) else "FAIL",
        "checks": results,
        "local_verification": parse_json_stdout(verifier or {}) if verifier else None,
        "train_eval_process": train_eval,
    }

    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_report(payload), encoding="utf-8")
    (RUNS_DIR / "latest-macmini-smoke.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "timestamp": payload["timestamp"],
                "verdict": payload["verdict"],
                "profile": payload["profile"],
                "report": str((run_dir / "report.md").relative_to(REPO_ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": payload["verdict"], "report": str(run_dir / "report.md")}, ensure_ascii=False))
    return 0 if payload["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
