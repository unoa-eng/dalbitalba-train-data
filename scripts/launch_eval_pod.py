#!/usr/bin/env python3
"""
Launch a RunPod evaluation pod that clones this repo, runs scripts/run_eval.sh,
and pushes evaluation artifacts back to GitHub.
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
STATE_FILE = STATE_DIR / "eval_pod_state.json"
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


def parse_gpu_types(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or ["NVIDIA A100 80GB PCIe"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch dalbitalba-train-data RunPod eval pod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-type",
        default=os.environ.get(
            "GPU_TYPE",
            "NVIDIA A100 80GB PCIe,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation",
        ),
        help="Comma-separated GPU preference list (tried in order)",
    )
    parser.add_argument(
        "--container-image",
        default=os.environ.get(
            "CONTAINER_IMAGE",
            "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
        ),
    )
    args = parser.parse_args()

    load_env()

    api_key = require_env("RUNPOD_API_KEY")
    github_token = require_env("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "unoa-eng/dalbitalba-train-data").strip()
    github_ref = os.environ.get("GITHUB_REF", "").strip() or detect_git_ref()
    hf_adapter_repo = require_env("HF_ADAPTER_REPO")
    anthropic_api_key = require_env("ANTHROPIC_API_KEY")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    ntfy_topic = os.environ.get("NTFY_TOPIC", "").strip()
    base_model = os.environ.get("BASE_MODEL", "upstage/SOLAR-10.7B-v1.0").strip()

    # SECURITY: never embed the PAT in dockerStartCmd — RunPod persists the
    # pod spec including this field. Use env.GITHUB_TOKEN via shell expansion
    # so the PAT only lives inside the running container, not in the spec.
    startup_cmd = (
        "mkdir -p /workspace/logs && "
        "rm -rf /workspace/repo && "
        f"git clone --branch {github_ref} --single-branch "
        f"\"https://x-access-token:${{GITHUB_TOKEN}}@github.com/{github_repo}.git\" "
        "/workspace/repo && "
        "chmod +x /workspace/repo/scripts/run_eval.sh && "
        "exec bash /workspace/repo/scripts/run_eval.sh"
    )

    env = {
        "GITHUB_TOKEN": github_token,
        "GITHUB_REPO": github_repo,
        "HF_ADAPTER_REPO": hf_adapter_repo,
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "RUNPOD_API_KEY": api_key,
        "BASE_MODEL": base_model,
        "RUNPOD_POD_ID": "__SELF__",
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if openai_api_key:
        env["OPENAI_API_KEY"] = openai_api_key
    if ntfy_topic:
        env["NTFY_TOPIC"] = ntfy_topic

    gpu_types = parse_gpu_types(args.gpu_type)

    payload = {
        "name": f"dalbitalba-eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
        "imageName": args.container_image,
        "computeType": "GPU",
        "cloudType": "COMMUNITY",
        "gpuCount": 1,
        "gpuTypePriority": "availability",
        "volumeInGb": 60,
        "containerDiskInGb": 30,
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
        raise SystemExit(str(last_error or RuntimeError("RunPod eval launch failed")))

    pod_id = data["id"]
    save_state(
        pod_id,
        {
            "github_repo": github_repo,
            "github_ref": github_ref,
            "hf_adapter_repo": hf_adapter_repo,
            "gpu_type": chosen_gpu,
            "gpu_type_candidates": gpu_types,
            "base_model": base_model,
        },
    )
    print(f"[done] pod_id={pod_id}")
    print(f"[state] {STATE_FILE}")


if __name__ == "__main__":
    main()
