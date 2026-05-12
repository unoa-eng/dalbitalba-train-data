#!/usr/bin/env python3
"""Round-2 training/evaluation process runner.

This is a local control-plane script. It does not contain secrets and it does
not reimplement training. It verifies the repo, records launch readiness, runs
a deterministic evaluation smoke test, and launches the existing RunPod trainer
only when the required credentials are present.
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
STATE_DIR = REPO_ROOT / ".state" / "train-eval-process"
REQUIRED_LAUNCH_ENV = ("RUNPOD_API_KEY", "HF_TOKEN", "HF_USERNAME", "GITHUB_TOKEN", "WANDB_API_KEY")
REQUIRED_DATA = (
    "cpt_enriched.jsonl",
    "cpt_corpus.v3.jsonl",
    "sft_thread_conditioned.jsonl",
    "sft_thread_conditioned.eval.jsonl",
    "orpo_pairs.jsonl",
    "recipes/round2-cycle1.env",
    "runs/round2-obsidian-synthesis/persona-30-extracted.json",
    "tokenizer_v4/tokenizer.json",
    "chain_train_round2.sh",
)
REQUIRED_REMOTE_FILES = REQUIRED_DATA + (
    "scripts/phase6_eval_v2.py",
    "scripts/phase6_generate.py",
    "scripts/merge_cpt_to_fp16.py",
    "scripts/merge_sft_to_fp16.py",
    "scripts/round2_build_orpo_pairs.py",
    "scripts/round2_build_tc_sft.py",
    "scripts/round2_integrity_check.py",
    "scripts/prelaunch_research_check.py",
    "scripts/round2_mutator.py",
    "train_cpt.py",
    "train_sft.py",
    "train_orpo.py",
)
ACTIVE_EVAL_SOURCE = (
    REPO_ROOT / "sft_thread_conditioned.eval.jsonl"
    if (REPO_ROOT / "sft_thread_conditioned.eval.jsonl").exists()
    else REPO_ROOT / "val_set.v3.jsonl"
)
ACTIVE_PERSONA_LIST = REPO_ROOT / "runs" / "round2-obsidian-synthesis" / "persona-30-extracted.json"
LOCAL_PYTHONWARNINGS = "ignore:urllib3 v2 only supports OpenSSL 1.1.1+"
LAUNCH_CRITICAL_PREFIXES = (
    ".github/workflows/",
    "recipes/",
    "scripts/",
    "tokenizer_v4/",
    "runs/round2-obsidian-synthesis/",
)
LAUNCH_CRITICAL_FILES = {
    "chain_train.sh",
    "chain_train_round2.sh",
    "train_cpt.py",
    "train_sft.py",
    "train_orpo.py",
}


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def run(
    cmd: list[str],
    *,
    check: bool = False,
    env: dict[str, str] | None = None,
    log_dir: Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    merged_env = {**os.environ, **(env or {})}
    existing_warnings = merged_env.get("PYTHONWARNINGS", "").strip()
    merged_env["PYTHONWARNINGS"] = (
        f"{LOCAL_PYTHONWARNINGS},{existing_warnings}"
        if existing_warnings
        else LOCAL_PYTHONWARNINGS
    )
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
    )
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if log_dir and label:
        stdout_path = log_dir / f"{label}.stdout.log"
        stderr_path = log_dir / f"{label}.stderr.log"
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        result["stdout_log"] = str(stdout_path.relative_to(REPO_ROOT))
        result["stderr_log"] = str(stderr_path.relative_to(REPO_ROOT))
    if check and proc.returncode != 0:
        raise SystemExit(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_json_text(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def line_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def inspect_inputs() -> dict[str, Any]:
    files: dict[str, Any] = {}
    missing: list[str] = []
    for rel in REQUIRED_DATA:
        path = REPO_ROOT / rel
        exists = path.exists()
        if not exists:
            missing.append(rel)
        files[rel] = {
            "exists": exists,
            "bytes": path.stat().st_size if exists and path.is_file() else None,
            "lines": line_count(path) if exists and path.suffix == ".jsonl" else None,
        }
    return {"files": files, "missing": missing}


def recipe_path_for_profile(profile: str) -> Path:
    mapping = {
        "paper8b": REPO_ROOT / "recipes" / "round2-cycle1.env",
        "default": REPO_ROOT / "recipes" / "round2-cycle1.env",
        "budget30": REPO_ROOT / "recipes" / "budget30.env",
        "smoke": REPO_ROOT / "recipes" / "smoke.env",
    }
    return mapping.get(profile, mapping["paper8b"])


def launch_env_status(profile: str) -> dict[str, Any]:
    launch_recipe_path = recipe_path_for_profile(profile)
    launch_recipe = parse_env_file(launch_recipe_path)
    merged = {**launch_recipe, **os.environ}
    keys = {key: bool(merged.get(key)) for key in REQUIRED_LAUNCH_ENV}
    return {
        "required": keys,
        "ready": all(keys.values()),
        "github_repo": merged.get("GITHUB_REPO", "unoa-eng/dalbitalba-train-data"),
        "train_github_ref": merged.get("TRAIN_GITHUB_REF") or merged.get("GITHUB_REF_NAME") or "current branch",
        "launch_recipe": str(launch_recipe_path.relative_to(REPO_ROOT)),
        "profile": profile,
    }


def resolve_launch_ref() -> str:
    return (
        os.environ.get("TRAIN_GITHUB_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or run(["git", "branch", "--show-current"])["stdout"].strip()
        or "main"
    )


def inspect_remote_ref_files(ref: str) -> dict[str, Any]:
    """Verify the exact Git ref RunPod will clone contains all round2 assets."""
    fetch = run(["git", "fetch", "--quiet", "origin", ref])
    remote_ref = f"origin/{ref}"
    if fetch["returncode"] != 0:
        # TRAIN_GITHUB_REF may be a commit SHA or already fully qualified.
        remote_ref = ref
    existing: set[str] = set()
    for rel in REQUIRED_REMOTE_FILES:
        result = run(["git", "cat-file", "-e", f"{remote_ref}:{rel}"])
        if result["returncode"] == 0:
            existing.add(rel)
    missing = [rel for rel in REQUIRED_REMOTE_FILES if rel not in existing]
    status = run(["git", "status", "--short"])
    dirty_lines = [line for line in status["stdout"].splitlines() if line.strip()]
    launch_critical_dirty: list[str] = []
    for line in dirty_lines:
        path = line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[-1].strip()
        if (
            path in LAUNCH_CRITICAL_FILES
            or path.endswith(".jsonl")
            or any(path.startswith(prefix) for prefix in LAUNCH_CRITICAL_PREFIXES)
        ):
            launch_critical_dirty.append(line)
    return {
        "ref": ref,
        "resolved_ref": remote_ref,
        "ready": not missing,
        "missing": missing,
        "fetch_returncode": fetch["returncode"],
        "fetch_stderr_tail": fetch["stderr"][-1000:],
        "dirty_worktree": bool(dirty_lines),
        "launch_critical_dirty": launch_critical_dirty,
    }


def write_jsonl_sample(src: Path, dst: Path, limit: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8", errors="replace") as handle, dst.open(
        "w", encoding="utf-8"
    ) as out:
        for line in handle:
            if not line.strip():
                continue
            out.write(line)
            written += 1
            if written >= limit:
                break
    if written == 0:
        raise RuntimeError(f"no sample rows written from {src}")


def run_eval_smoke(run_dir: Path, sample_rows: int) -> dict[str, Any]:
    sample = run_dir / "phase6_smoke_same.jsonl"
    report = run_dir / "phase6_v2_smoke.json"
    write_jsonl_sample(ACTIVE_EVAL_SOURCE, sample, sample_rows)
    cmd = [
        sys.executable,
        "scripts/phase6_eval_v2.py",
        "--ai",
        str(sample),
        "--raw",
        str(sample),
        "--out",
        str(report),
        "--skip-mauve",
    ]
    if ACTIVE_PERSONA_LIST.exists() and ACTIVE_EVAL_SOURCE.name == "sft_thread_conditioned.eval.jsonl":
        cmd.extend(["--persona-list", str(ACTIVE_PERSONA_LIST)])
    result = run(cmd, log_dir=run_dir, label="phase6_eval_v2_smoke")
    payload = None
    if report.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
    return {
        "returncode": result["returncode"],
        "report": str(report.relative_to(REPO_ROOT)),
        "purpose": "metric_identity_smoke_only_not_quality_gate",
        "overall": (payload or {}).get("overall"),
        "stderr_tail": result["stderr"][-2000:],
    }


def load_preflight_details(stdout: str) -> dict[str, Any] | None:
    meta = parse_json_text(stdout.strip())
    if not meta:
        return None
    report_ref = meta.get("report")
    if not isinstance(report_ref, str):
        return meta
    report_path = Path(report_ref)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_ref
    report_json = report_path.with_name("report.json")
    if report_json.exists():
        payload = parse_json_text(report_json.read_text(encoding="utf-8"))
        if payload:
            meta["report_json"] = str(report_json.relative_to(REPO_ROOT))
            meta["details"] = payload
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Launch RunPod if required env is present")
    parser.add_argument("--dry-run", action="store_true", help="Render launch payload without creating a pod")
    parser.add_argument("--sample-rows", type=int, default=80)
    parser.add_argument(
        "--profile",
        choices=("paper8b", "budget30", "smoke"),
        default=os.environ.get("BUDGET_PROFILE", "paper8b"),
        help="RunPod recipe/verifier profile to validate",
    )
    args = parser.parse_args()
    # C5-3: propagate --profile so child scripts that read BUDGET_PROFILE pick it up.
    os.environ["BUDGET_PROFILE"] = args.profile

    run_dir = STATE_DIR / utc_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    static_checks = {
        "round2_shell_syntax": run(
            ["bash", "-n", "chain_train_round2.sh"],
            log_dir=run_dir,
            label="bash_n_chain_train_round2",
        ),
        "python_compile": run(
            [
                sys.executable,
                "-m",
                "py_compile",
                "scripts/launch_train_pod.py",
                "scripts/macmini_smoke_loop.py",
                "scripts/train_eval_process.py",
                "scripts/phase6_eval_v2.py",
                "scripts/phase6_generate.py",
                "scripts/round2_integrity_check.py",
                "scripts/sft_format_smoke_test.py",
                "scripts/prelaunch_research_check.py",
                "scripts/clean_round2_launch_data.py",
                "scripts/split_round2_sft_eval.py",
                "train_sft.py",
                "train_orpo.py",
            ],
            log_dir=run_dir,
            label="py_compile",
        ),
        "round2_integrity": run(
            [sys.executable, "scripts/round2_integrity_check.py"],
            log_dir=run_dir,
            label="round2_integrity_check",
        ),
        "prelaunch_research": run(
            # Note: prelaunch_research_check.py removed its
            # ALLOW_MISSING_RUNTIME_SECRETS bypass in commit 6f66ee7. The env
            # injection here is dead; kept the call without it for clarity.
            [sys.executable, "scripts/prelaunch_research_check.py"],
            log_dir=run_dir,
            label="prelaunch_research_check",
        ),
    }
    static_ok = all(item["returncode"] == 0 for item in static_checks.values())
    preflight = run(
        [sys.executable, "scripts/local_verification_loop.py", "--strict", "--profile", args.profile],
        log_dir=run_dir,
        label="local_verification_loop",
    )
    preflight_details = load_preflight_details(preflight["stdout"])
    inputs = inspect_inputs()
    env_status = launch_env_status(args.profile)
    launch_recipe_env = parse_env_file(recipe_path_for_profile(args.profile))
    remote_files = inspect_remote_ref_files(resolve_launch_ref())
    eval_smoke = run_eval_smoke(run_dir, args.sample_rows) if static_ok and not inputs["missing"] else None
    preflight_ok = preflight["returncode"] == 0
    inputs_ok = not inputs["missing"]
    eval_smoke_ok = eval_smoke is not None and eval_smoke["returncode"] == 0

    launch_result: dict[str, Any] | None = None
    if args.launch or args.dry_run:
        if not static_ok:
            launch_result = {
                "returncode": 2,
                "blocked": "static checks failed",
                "failed": [k for k, v in static_checks.items() if v["returncode"] != 0],
            }
        elif not preflight_ok:
            launch_result = {
                "returncode": preflight["returncode"],
                "blocked": "local verification preflight failed",
                "report": (preflight_details or {}).get("report"),
            }
        elif not inputs_ok:
            launch_result = {
                "returncode": 3,
                "blocked": "required inputs are missing",
                "missing": inputs["missing"],
            }
        elif not eval_smoke_ok:
            launch_result = {
                "returncode": (eval_smoke or {}).get("returncode", 3),
                "blocked": "phase6 identity eval smoke failed or did not run",
                "eval_smoke": eval_smoke,
            }
        elif not remote_files["ready"]:
            launch_result = {
                "returncode": 4,
                "blocked": "selected Git ref is missing files RunPod must clone",
                "ref": remote_files["ref"],
                "missing": remote_files["missing"],
            }
        elif args.launch and remote_files["launch_critical_dirty"]:
            launch_result = {
                "returncode": 5,
                "blocked": "launch-critical local changes are not committed/pushed",
                "dirty": remote_files["launch_critical_dirty"],
            }
        elif env_status["ready"] or args.dry_run:
            cmd = [sys.executable, "scripts/launch_train_pod.py", "--chain", "round2"]
            if args.dry_run:
                cmd.append("--dry-run")
            launch_result = run(cmd, env=launch_recipe_env, log_dir=run_dir, label="launch_train_pod")
        else:
            launch_result = {
                "returncode": 2,
                "blocked": "missing required launch env",
                "missing": [k for k, ok in env_status["required"].items() if not ok],
            }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "static_checks": {
            key: {
                "returncode": value["returncode"],
                "stdout_log": value.get("stdout_log"),
                "stderr_log": value.get("stderr_log"),
                "stderr_tail": value["stderr"][-2000:],
            }
            for key, value in static_checks.items()
        },
        "preflight": {
            "returncode": preflight["returncode"],
            "stdout": preflight["stdout"].strip(),
            "stderr_tail": preflight["stderr"][-2000:],
            "stdout_log": preflight.get("stdout_log"),
            "stderr_log": preflight.get("stderr_log"),
            "details": preflight_details,
        },
        "inputs": inputs,
        "remote_files": remote_files,
        "launch_env": env_status,
        "launch_recipe_env": launch_recipe_env,
        "eval_smoke": eval_smoke,
        "launch": launch_result,
    }
    (run_dir / "state.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (STATE_DIR / "latest.json").write_text(
        json.dumps({"run_dir": str(run_dir.relative_to(REPO_ROOT)), **payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not static_ok:
        return 2
    if preflight["returncode"] != 0:
        return preflight["returncode"]
    if inputs["missing"]:
        return 3
    if not remote_files["ready"]:
        return 4
    if args.launch and remote_files["launch_critical_dirty"]:
        return 5
    if eval_smoke and eval_smoke["returncode"] != 0:
        return eval_smoke["returncode"]
    if launch_result and launch_result.get("returncode", 0) != 0:
        return int(launch_result["returncode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
