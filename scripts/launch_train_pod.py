#!/usr/bin/env python3
"""
Launch a RunPod training pod that clones this repo, runs chain_train.sh,
uploads adapters to Hugging Face, and pushes run metadata back to GitHub.
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
    args = parser.parse_args()

    api_key = require_env("RUNPOD_API_KEY")
    hf_token = require_env("HF_TOKEN")
    hf_username = require_env("HF_USERNAME")
    github_token = require_env("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "unoa-eng/dalbitalba-train-data").strip()
    github_ref = os.environ.get("GITHUB_REF", "").strip() or detect_git_ref()
    ntfy_topic = os.environ.get("NTFY_TOPIC", "").strip()
    cloud_type = os.environ.get("RUNPOD_CLOUD_TYPE", "COMMUNITY").strip().upper() or "COMMUNITY"
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base").strip() or "Qwen/Qwen3-8B-Base"
    train_cpt_jsonl = os.environ.get(
        "TRAIN_CPT_JSONL", "/workspace/data/cpt_corpus.v2.jsonl"
    ).strip() or "/workspace/data/cpt_corpus.v2.jsonl"
    train_sft_pair_jsonl = os.environ.get(
        "TRAIN_SFT_PAIR_JSONL", "/workspace/data/sft_pairs.v2.jsonl"
    ).strip() or "/workspace/data/sft_pairs.v2.jsonl"
    train_val_jsonl = os.environ.get(
        "TRAIN_VAL_JSONL", "/workspace/data/val_set.v2.jsonl"
    ).strip() or "/workspace/data/val_set.v2.jsonl"
    cpt_num_epochs = os.environ.get("CPT_NUM_EPOCHS", "1").strip() or "1"
    sft_num_epochs = os.environ.get("SFT_NUM_EPOCHS", "2").strip() or "2"
    cpt_lr = os.environ.get("CPT_LR", "2e-4").strip() or "2e-4"
    sft_lr = os.environ.get("SFT_LR", "5e-5").strip() or "5e-5"
    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    wandb_project = os.environ.get("WANDB_PROJECT", "dalbitalba-v2").strip() or "dalbitalba-v2"
    sentry_dsn = os.environ.get("SENTRY_DSN", "").strip()
    gpu_types = parse_gpu_types(args.gpu_type)

    # SECURITY: never embed the PAT in dockerStartCmd — RunPod stores the pod
    # spec (including this command) and exposes it in pod detail API responses.
    # The token is injected via env.GITHUB_TOKEN below; we reference it only
    # through shell variable expansion, which the RunPod API does not persist.
    #
    # Also: logs tee'd to /workspace/logs/chain.log so they survive pod EXIT
    # (network volume at /workspace persists across pod lifecycle).
    startup_cmd = (
        "mkdir -p /workspace/logs /workspace/data /workspace/scripts "
        "/workspace/out /workspace/hf_cache && "
        "export HF_HOME=/workspace/hf_cache && "
        "export TOKENIZERS_PARALLELISM=false && "
        "rm -rf /workspace/repo && "
        f"git clone --branch {github_ref} --single-branch "
        f"\"https://x-access-token:${{GITHUB_TOKEN}}@github.com/{github_repo}.git\" "
        "/workspace/repo && "
        # copy v2 data + train scripts + merge script
        "cp /workspace/repo/*.jsonl /workspace/data/ 2>/dev/null || true && "
        "cp /workspace/repo/train_*.py /workspace/ 2>/dev/null || true && "
        "mkdir -p /workspace/scripts && "
        "cp /workspace/repo/scripts/merge_cpt_to_fp16.py /workspace/scripts/ 2>/dev/null || true && "
        "cp /workspace/repo/chain_train.sh /workspace/chain_train.sh && "
        "chmod +x /workspace/chain_train.sh && "
        # tee all logs to network volume — survives pod EXIT
        "exec bash /workspace/chain_train.sh 2>&1 | tee -a "
        "/workspace/logs/chain_$(date -u '+%Y%m%dT%H%M%SZ').log"
    )

    env = {
        "HF_TOKEN": hf_token,
        "HF_USERNAME": hf_username,
        "GITHUB_TOKEN": github_token,
        "GITHUB_REPO": github_repo,
        "RUNPOD_API_KEY": api_key,
        "BASE_MODEL": base_model,
        "INPUT_JSONL": train_cpt_jsonl,
        "SFT_PAIR_JSONL": train_sft_pair_jsonl,
        "CPT_VAL_JSONL": train_val_jsonl,
        "CPT_NUM_EPOCHS": cpt_num_epochs,
        "SFT_NUM_EPOCHS": sft_num_epochs,
        "CPT_LR": cpt_lr,
        "SFT_LR": sft_lr,
        "WANDB_PROJECT": wandb_project,
    }
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
        env["TRAIN_REPORT_TO"] = "wandb"
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
