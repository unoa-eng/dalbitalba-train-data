#!/usr/bin/env python3
"""
Launch a RunPod training pod that clones this repo and runs the selected
training chain.

TRAIN_CHAIN=classic runs chain_train.sh.
TRAIN_CHAIN=round2 runs chain_train_round2.sh, which executes the round-2
5-phase CPT/CPT-DoRA/TC-SFT/ORPO/eval-v2 process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = REPO_ROOT / ".state"
STATE_FILE = STATE_DIR / "train_pod_state.json"
RUNPOD_REST = "https://rest.runpod.io/v1/pods"
LOCAL_VERIFICATION_LATEST = REPO_ROOT / "runs" / "latest-local-verification.json"
VERIFIER_GATED_PROFILES = {"paper8b", "budget30", "smoke"}
# B2 — review 2026-05-12: GPU lock for cost-bound profiles
STRICT_L40S_PROFILES = {"paper8b", "budget30"}
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


def assert_verifier_pass_for_profile() -> None:
    """Refuse gated paid profiles unless the latest verifier is a clean PASS.

    Override with FORCE_LAUNCH=1 only for explicit experiments.
    """
    expected_profile = os.environ.get("BUDGET_PROFILE", "").strip()
    if expected_profile not in VERIFIER_GATED_PROFILES:
        return
    if os.environ.get("FORCE_LAUNCH", "0") == "1":
        print(f"[WARN] FORCE_LAUNCH=1 — skipping verifier gate for {expected_profile}")
        return
    if not LOCAL_VERIFICATION_LATEST.exists():
        raise SystemExit(
            f"[FATAL] BUDGET_PROFILE={expected_profile} requires a fresh local verification report.\n"
            f"        Run: python3 scripts/local_verification_loop.py --strict --profile {expected_profile}\n"
            f"        Expected file: {LOCAL_VERIFICATION_LATEST}"
        )
    try:
        report = json.loads(LOCAL_VERIFICATION_LATEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[FATAL] cannot parse {LOCAL_VERIFICATION_LATEST}: {exc}")
    verdict = report.get("verdict", "")
    severe_count = int(report.get("severe_count") or 0)
    warning_count = int(report.get("warning_count") or 0)
    report_profile = report.get("profile")
    if severe_count or warning_count or verdict != "PASS":
        raise SystemExit(
            f"[FATAL] latest local verification verdict={verdict!r} severe_count={severe_count} warning_count={warning_count}.\n"
            f"        Refuse to launch {expected_profile} unless local verification is PASS with zero warnings.\n"
            f"        Inspect: {report.get('report', LOCAL_VERIFICATION_LATEST)}\n"
            f"        Override only with FORCE_LAUNCH=1 for explicit experiments."
        )
    if report_profile and report_profile != expected_profile:
        raise SystemExit(
            f"[FATAL] latest local verification profile={report_profile!r} (expected {expected_profile!r}).\n"
            f"        Run: python3 scripts/local_verification_loop.py --strict --profile {expected_profile}"
        )
    print(f"[gate] verifier {verdict} severe_count=0 @ {report.get('timestamp', '?')}")


def _enforce_gpu_lock(profile: str, gpu_type: str) -> None:
    """B2 — review 2026-05-12: GPU lock for cost-bound profiles.

    paper8b and budget30 are strictly L40S-only to prevent cost-cap overrun
    when A100 ($1.19/h) or RTX-6000-Ada are selected as fallback.
    """
    if not profile or profile not in STRICT_L40S_PROFILES:
        return
    allowed = {"NVIDIA L40S"}
    requested = {g.strip() for g in gpu_type.split(",") if g.strip()}
    if not requested.issubset(allowed):
        raise SystemExit(
            f"GPU lock violation: profile={profile} allows only L40S, "
            f"got {sorted(requested)}. Set GPU_TYPE='NVIDIA L40S' or change profile."
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


def require_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise SystemExit(f"[ERROR] missing env: {key}")
    return value


def require_or_placeholder(key: str, dry_run: bool) -> str:
    value = os.environ.get(key, "").strip()
    if value:
        return value
    if dry_run:
        return f"__DRY_RUN_{key}__"
    raise SystemExit(f"[ERROR] missing env: {key}")


def git_dirty_launch_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    dirty: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[-1].strip()
        if (
            path in LAUNCH_CRITICAL_FILES
            or path.endswith(".jsonl")
            or any(path.startswith(prefix) for prefix in LAUNCH_CRITICAL_PREFIXES)
        ):
            dirty.append(line)
    return dirty


def assert_launch_ref_clean(dry_run: bool) -> None:
    if dry_run or os.environ.get("ALLOW_DIRTY_LAUNCH", "0") == "1":
        return
    dirty = git_dirty_launch_files()
    if dirty:
        sample = "\n".join(f"        {line}" for line in dirty[:30])
        extra = "" if len(dirty) <= 30 else f"\n        ... {len(dirty) - 30} more"
        raise SystemExit(
            "[FATAL] launch-critical local changes are not committed/pushed.\n"
            "        RunPod clones the selected Git ref, so these local files would not run in the pod.\n"
            f"{sample}{extra}\n"
            "        Commit and push the intended ref, or set ALLOW_DIRTY_LAUNCH=1 for an explicit experiment."
        )


def runpod_request(api_key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        RUNPOD_REST,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "dalbitalba-train-data/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod API HTTP {exc.code}: {body[:600]}") from None


def parse_gpu_types(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or ["NVIDIA A100 80GB PCIe"]


def normalize_workspace_data_path(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return value
    path = Path(value)
    if path.is_absolute() or value.startswith("/workspace/"):
        return value

    relative = value[2:] if value.startswith("./") else value
    if relative.startswith("data/"):
        relative = relative[len("data/") :]
    return f"/workspace/data/{relative}"


def resolve_workspace_data_path(
    env_keys: tuple[str, ...],
    candidate_names: tuple[str, ...],
) -> str:
    for key in env_keys:
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_workspace_data_path(value)
    for name in candidate_names:
        if (REPO_ROOT / name).exists():
            return f"/workspace/data/{name}"
    return f"/workspace/data/{candidate_names[-1]}"


def detect_git_ref() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        ref = result.stdout.strip()
        if ref and ref != "HEAD":
            return ref
    except Exception:
        pass
    return "main"


def resolve_git_ref() -> str:
    raw = (
        os.environ.get("TRAIN_GITHUB_REF", "").strip()
        or os.environ.get("GITHUB_REF_NAME", "").strip()
        or os.environ.get("GITHUB_REF", "").strip()
    )
    if raw.startswith("refs/heads/"):
        raw = raw.removeprefix("refs/heads/")
    elif raw.startswith("refs/tags/"):
        raw = raw.removeprefix("refs/tags/")
    elif raw.startswith("refs/pull/"):
        raw = ""
    return raw or detect_git_ref()


def save_state(pod_id: str, metadata: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {
                "pod_id": pod_id,
                "launched_at": datetime.now(timezone.utc).isoformat(),
                **metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def redact_string(value: str) -> str:
    if "x-access-token:" not in value:
        return value
    prefix, _, rest = value.partition("x-access-token:")
    _, sep, tail = rest.partition("@github.com/")
    if not sep:
        return value
    return f"{prefix}x-access-token:***REDACTED***@github.com/{tail}"


def redact_payload(payload: dict) -> dict:
    redacted = json.loads(json.dumps(payload))
    for key in list(redacted.get("env", {}).keys()):
        redacted["env"][key] = "***REDACTED***"

    start_cmd = redacted.get("dockerStartCmd")
    if isinstance(start_cmd, list):
        redacted["dockerStartCmd"] = [redact_string(item) if isinstance(item, str) else item for item in start_cmd]
    elif isinstance(start_cmd, str):
        redacted["dockerStartCmd"] = redact_string(start_cmd)
    return redacted


def main() -> None:
    # load_env MUST run before argparse defaults resolve, otherwise the
    # .env.local values for GPU_TYPE / CONTAINER_IMAGE are silently ignored.
    load_env()

    parser = argparse.ArgumentParser(description="Launch dalbitalba-train-data RunPod training pod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-type",
        default=os.environ.get(
            "GPU_TYPE",
            "NVIDIA L40S,NVIDIA A100 80GB PCIe,NVIDIA RTX 6000 Ada Generation",
        ),
        help="Comma-separated GPU preference list (L40S first for cost)",
    )
    parser.add_argument(
        "--container-image",
        default=os.environ.get(
            "CONTAINER_IMAGE",
            # torch 2.4 + CUDA 12.4 matches the L40S driver version (12040)
            # that pod4 preflight surfaced. Earlier 2.2.0-cuda12.1.1 image was
            # internally updated to a newer CUDA build than the pod drivers.
            "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        ),
    )
    parser.add_argument(
        "--chain",
        choices=("classic", "round2"),
        default=os.environ.get("TRAIN_CHAIN", "round2").strip() or "round2",
        help="Training chain to execute inside RunPod",
    )
    args = parser.parse_args()

    # B2 — GPU lock: paper8b/budget30 must not slip to A100/RTX-6000-Ada
    _enforce_gpu_lock(os.environ.get("BUDGET_PROFILE", "").strip(), args.gpu_type)

    # Mechanical defense — refuse gated paid profiles without a fresh PASS
    # verifier report. Bypassable only with FORCE_LAUNCH=1.
    assert_verifier_pass_for_profile()
    assert_launch_ref_clean(args.dry_run)

    api_key = require_or_placeholder("RUNPOD_API_KEY", args.dry_run)
    hf_token = require_or_placeholder("HF_TOKEN", args.dry_run)
    hf_username = require_or_placeholder("HF_USERNAME", args.dry_run)
    github_token = require_or_placeholder("GITHUB_TOKEN", args.dry_run)
    github_repo = os.environ.get("GITHUB_REPO", "unoa-eng/dalbitalba-train-data").strip()
    github_ref = resolve_git_ref()
    ntfy_topic = os.environ.get("NTFY_TOPIC", "").strip()
    cloud_type = os.environ.get("RUNPOD_CLOUD_TYPE", "COMMUNITY").strip().upper() or "COMMUNITY"
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base").strip() or "Qwen/Qwen3-8B-Base"
    train_cpt_jsonl = resolve_workspace_data_path(
        ("TRAIN_CPT_JSONL", "INPUT_JSONL"),
        ("cpt_corpus.v3.jsonl", "cpt_corpus.v2.jsonl"),
    )
    train_sft_pair_jsonl = resolve_workspace_data_path(
        ("TRAIN_SFT_PAIR_JSONL", "SFT_PAIR_JSONL"),
        ("sft_pairs.v2.jsonl",),
    )
    train_val_jsonl = resolve_workspace_data_path(
        ("TRAIN_VAL_JSONL", "CPT_VAL_JSONL"),
        ("val_set.v2.jsonl",),
    )
    cpt_num_epochs = os.environ.get("CPT_NUM_EPOCHS", "1").strip() or "1"
    sft_num_epochs = os.environ.get("SFT_NUM_EPOCHS", "2").strip() or "2"
    cpt_lr = os.environ.get("CPT_LR", "2e-4").strip() or "2e-4"
    sft_lr = os.environ.get("SFT_LR", "5e-5").strip() or "5e-5"
    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if args.chain == "round2":
        wandb_api_key = require_or_placeholder("WANDB_API_KEY", args.dry_run)
    wandb_project = os.environ.get("WANDB_PROJECT", "dalbitalba-v2").strip() or "dalbitalba-v2"
    sentry_dsn = os.environ.get("SENTRY_DSN", "").strip()
    gpu_types = parse_gpu_types(args.gpu_type)

    chain_script = "chain_train_round2.sh" if args.chain == "round2" else "chain_train.sh"

    # SECURITY: never embed the PAT in dockerStartCmd — RunPod stores the pod
    # spec (including this command) and exposes it in pod detail API responses.
    # The token is injected via env.GITHUB_TOKEN below; we reference it only
    # through shell variable expansion, which the RunPod API does not persist.
    #
    # Also: logs tee'd to /workspace/logs so they survive pod EXIT because
    # /workspace is backed by the Pod volume disk in this spec. Keep pipefail
    # here: otherwise a failed chain hidden behind tee can look successful to
    # RunPod's process supervisor.
    startup_cmd = (
        "set -o pipefail && "
        "mkdir -p /workspace/logs /workspace/data /workspace/scripts "
        "/workspace/out /workspace/hf_cache /workspace/recipes && "
        "export HF_HOME=/workspace/hf_cache && "
        "export TOKENIZERS_PARALLELISM=false && "
        "rm -rf /workspace/repo && "
        f"git clone --branch {github_ref} --single-branch "
        f"\"https://x-access-token:${{GITHUB_TOKEN}}@github.com/{github_repo}.git\" "
        "/workspace/repo && "
        # copy repo datasets, recipes, train scripts, and helper scripts
        "(cp /workspace/repo/*.jsonl /workspace/data/ 2>/dev/null || true) && "
        "(cp /workspace/repo/recipes/*.env /workspace/recipes/ 2>/dev/null || true) && "
        "mkdir -p /workspace/runs && "
        "(cp -a /workspace/repo/runs/round2-obsidian-synthesis /workspace/runs/ 2>/dev/null || true) && "
        "(cp -a /workspace/repo/tokenizer_v4 /workspace/ 2>/dev/null || true) && "
        "(cp -a /workspace/repo/source_db_cache /workspace/data/ 2>/dev/null || true) && "
        "(cp /workspace/repo/train_*.py /workspace/ 2>/dev/null || true) && "
        "mkdir -p /workspace/scripts && "
        "(cp /workspace/repo/scripts/*.py /workspace/scripts/ 2>/dev/null || true) && "
        f"cp /workspace/repo/{chain_script} /workspace/{chain_script} && "
        f"chmod +x /workspace/{chain_script} && "
        # tee all logs to network volume — survives pod EXIT
        f"bash /workspace/{chain_script} 2>&1 | tee -a "
        "/workspace/logs/chain_$(date -u '+%Y%m%dT%H%M%SZ').log; "
        "rc=${PIPESTATUS[0]}; "
        "echo \"$(date -u '+%Y-%m-%dT%H:%M:%SZ') chain exit rc=${rc}\" "
        ">> /workspace/logs/launcher.log; "
        "exit ${rc}"
    )

    env = {
        "HF_TOKEN": hf_token,
        "HF_USERNAME": hf_username,
        "GITHUB_TOKEN": github_token,
        "GITHUB_REPO": github_repo,
        "RUNPOD_API_KEY": api_key,
        "BASE_MODEL": base_model,
        "TRAIN_CPT_JSONL": train_cpt_jsonl,
        "INPUT_JSONL": train_cpt_jsonl,
        "TRAIN_SFT_PAIR_JSONL": train_sft_pair_jsonl,
        "SFT_PAIR_JSONL": train_sft_pair_jsonl,
        "TRAIN_VAL_JSONL": train_val_jsonl,
        "CPT_VAL_JSONL": train_val_jsonl,
        "CPT_NUM_EPOCHS": cpt_num_epochs,
        "SFT_NUM_EPOCHS": sft_num_epochs,
        "CPT_LR": cpt_lr,
        "SFT_LR": sft_lr,
        "WANDB_PROJECT": wandb_project,
        "TRAIN_CHAIN": args.chain,
    }
    for optional_key in (
        "CPT_MAX_STEPS",
        "SFT_MAX_STEPS",
        "CPT_LIMIT_ROWS",
        "CPT_VAL_LIMIT_ROWS",
        "SFT_RAW_LIMIT_ROWS",
        "SFT_PAIR_LIMIT_ROWS",
        "SFT_VAL_LIMIT_ROWS",
        "SFT_RAW_RATIO",
        "CPT_SAVE_STEPS",
        "SFT_SAVE_STEPS",
        "CPT_EVAL_STEPS",
        "SFT_EVAL_STEPS",
        "CPT_LOGGING_STEPS",
        "SFT_LOGGING_STEPS",
        "SKIP_SFT",
        "CPT_PHASE_1_DATA",
        "CPT_PHASE_2_DATA",
        "CPT_TIMEOUT_HOURS",
        "MERGE_TIMEOUT_HOURS",
        "SFT_TIMEOUT_HOURS",
        "HF_UPLOAD_TIMEOUT_HOURS",
        "ORPO_TIMEOUT_HOURS",
        "EVAL_TIMEOUT_HOURS",
        "ROUND2_RECIPE",
        "ROUND2_STOP_POD",
        "ROUND2_SKIP_HF_UPLOAD",
        "HF_REPO_ROUND2",
        "CPT_BATCH_SIZE",
        "CPT_GRAD_ACCUM",
        "SFT_BATCH_SIZE",
        "SFT_GRAD_ACCUM",
        "BUDGET_PROFILE",
        "BUDGET_CAP_USD",
        "EVAL_MODE",
        "EVAL_MAX_ROWS",
        "EVAL_GATE",
        "MAUVE_DISABLED",
        "EVAL_PERSONA_LIST",
        "GENERATION_TEMP",
        "GENERATION_TOP_P",
        "GENERATION_MAX_NEW_TOKENS",
        "ORPO_NUM_EPOCHS",
        "ORPO_BETA",
    ):
        value = os.environ.get(optional_key, "").strip()
        if value:
            env[optional_key] = value
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
        env["TRAIN_REPORT_TO"] = "wandb"
        env["WANDB_PROJECT"] = wandb_project
        env["WANDB_RUN_GROUP"] = os.environ.get("WANDB_RUN_GROUP", f"{args.chain}-{github_ref}").strip()
        env["WANDB_TAGS"] = os.environ.get("WANDB_TAGS", f"{args.chain},{base_model}").strip()
        env["WANDB_NOTES"] = os.environ.get(
            "WANDB_NOTES",
            f"dalbitalba {args.chain} run from {github_repo}@{github_ref}",
        ).strip()
    if sentry_dsn:
        env["SENTRY_DSN"] = sentry_dsn
    if ntfy_topic:
        env["NTFY_TOPIC"] = ntfy_topic

    payload = {
        "name": f"dalbitalba-train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
        "imageName": args.container_image,
        "computeType": "GPU",
        "cloudType": cloud_type,
        "gpuCount": 1,
        "gpuTypePriority": "availability",
        "volumeInGb": 80,
        "containerDiskInGb": 40,
        "minVCPUPerGPU": 8,
        "minRAMPerGPU": 48,
        "volumeMountPath": "/workspace",
        "ports": [],
        "interruptible": False,
        "env": env,
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-lc", startup_cmd],
    }

    if args.dry_run:
        dry_run_payload = dict(payload)
        dry_run_payload["gpuTypeIds"] = gpu_types
        print(json.dumps(redact_payload(dry_run_payload), indent=2, ensure_ascii=False))
        return

    last_error: RuntimeError | None = None
    chosen_gpu: str | None = None
    data: dict | None = None
    for gpu_type in gpu_types:
        candidate_payload = dict(payload)
        candidate_payload["gpuTypeIds"] = [gpu_type]
        try:
            data = runpod_request(api_key, candidate_payload)
            chosen_gpu = gpu_type
            break
        except RuntimeError as exc:
            last_error = exc
            print(f"[warn] gpu launch failed for {gpu_type}: {exc}")
            if "balance is too low" in str(exc).lower():
                break

    if data is None or chosen_gpu is None:
        raise SystemExit(str(last_error or RuntimeError("RunPod launch failed")))

    pod_id = data["id"]
    save_state(
        pod_id,
        {
            "github_repo": github_repo,
            "github_ref": github_ref,
            "hf_username": hf_username,
            "base_model": base_model,
            "gpu_type": chosen_gpu,
            "cloud_type": cloud_type,
            "gpu_type_candidates": gpu_types,
            "train_chain": args.chain,
            "train_cpt_jsonl": train_cpt_jsonl,
            "train_sft_pair_jsonl": train_sft_pair_jsonl,
            "train_val_jsonl": train_val_jsonl,
            "cpt_num_epochs": cpt_num_epochs,
            "sft_num_epochs": sft_num_epochs,
            "cpt_lr": cpt_lr,
            "sft_lr": sft_lr,
        },
    )
    print(f"[done] pod_id={pod_id}")
    print(f"[state] {STATE_FILE}")


if __name__ == "__main__":
    main()
