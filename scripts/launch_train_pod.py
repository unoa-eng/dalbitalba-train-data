#!/usr/bin/env python3
"""
Launch a RunPod training pod that clones this repo, runs chain_train.sh,
uploads adapters to Hugging Face, and pushes run metadata back to GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = REPO_ROOT / ".state"
STATE_FILE = STATE_DIR / "train_pod_state.json"
RUNPOD_GQL = "https://api.runpod.io/graphql"


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


def runpod_request(api_key: str, query: str, variables: dict) -> dict:
    req = urllib.request.Request(
        f"{RUNPOD_GQL}?api_key={api_key}",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
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


def redact_payload(payload: dict) -> dict:
    redacted = json.loads(json.dumps(payload))
    for item in redacted.get("input", {}).get("env", []):
        if "value" in item:
            item["value"] = "***REDACTED***"

    start_cmd = redacted.get("input", {}).get("dockerStartCmd")
    if isinstance(start_cmd, str) and "x-access-token:" in start_cmd:
        prefix, _, rest = start_cmd.partition("x-access-token:")
        _, sep, tail = rest.partition("@github.com/")
        if sep:
            redacted["input"]["dockerStartCmd"] = f"{prefix}x-access-token:***REDACTED***@github.com/{tail}"
    return redacted


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch dalbitalba-train-data RunPod training pod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-type", default=os.environ.get("GPU_TYPE", "NVIDIA A100 80GB PCIe"))
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
    hf_token = require_env("HF_TOKEN")
    hf_username = require_env("HF_USERNAME")
    github_token = require_env("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "unoa-eng/dalbitalba-train-data").strip()
    github_ref = os.environ.get("GITHUB_REF", "main").strip() or "main"
    ntfy_topic = os.environ.get("NTFY_TOPIC", "").strip()

    clone_url = f"https://x-access-token:{github_token}@github.com/{github_repo}.git"
    startup_cmd = (
        "bash -lc "
        "\"mkdir -p /workspace/logs /workspace/data /workspace/scripts /workspace/out && "
        f"git clone --branch {github_ref} --single-branch {clone_url} /workspace/repo && "
        "cp /workspace/repo/*.jsonl /workspace/data/ 2>/dev/null || true && "
        "cp /workspace/repo/train_*.py /workspace/ 2>/dev/null || true && "
        "cp /workspace/repo/chain_train.sh /workspace/chain_train.sh && "
        "nohup bash /workspace/chain_train.sh > /workspace/logs/chain.log 2>&1 &\""
    )

    env = [
        {"key": "HF_TOKEN", "value": hf_token},
        {"key": "HF_USERNAME", "value": hf_username},
        {"key": "GITHUB_TOKEN", "value": github_token},
        {"key": "GITHUB_REPO", "value": github_repo},
        {"key": "RUNPOD_API_KEY", "value": api_key},
    ]
    if ntfy_topic:
        env.append({"key": "NTFY_TOPIC", "value": ntfy_topic})

    payload = {
        "input": {
            "cloudType": "COMMUNITY",
            "gpuCount": 1,
            "volumeInGb": 80,
            "containerDiskInGb": 40,
            "minVcpuCount": 8,
            "minMemoryInGb": 48,
            "gpuTypeId": args.gpu_type,
            "name": f"dalbitalba-train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
            "imageName": args.container_image,
            "dockerArgs": "",
            "ports": "",
            "volumeMountPath": "/workspace",
            "env": env,
            "startSsh": False,
            "startJupyter": False,
            "dockerStartCmd": startup_cmd,
        }
    }

    if args.dry_run:
        print(json.dumps(redact_payload(payload), indent=2, ensure_ascii=False))
        return

    mutation = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        desiredStatus
      }
    }
    """
    data = runpod_request(api_key, mutation, payload)
    if "errors" in data:
        raise SystemExit(json.dumps(data["errors"], ensure_ascii=False))

    pod_id = data["data"]["podFindAndDeployOnDemand"]["id"]
    save_state(
        pod_id,
        {
            "github_repo": github_repo,
            "github_ref": github_ref,
            "hf_username": hf_username,
            "gpu_type": args.gpu_type,
        },
    )
    print(f"[done] pod_id={pod_id}")
    print(f"[state] {STATE_FILE}")


if __name__ == "__main__":
    main()
