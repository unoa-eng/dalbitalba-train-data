#!/usr/bin/env python3
"""
Validate local environment variables for dalbitalba-train-data workflows.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


REQUIRED_KEYS = {
    "train": [
        "RUNPOD_API_KEY",
        "HF_TOKEN",
        "HF_USERNAME",
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
    ],
    "eval": [
        "RUNPOD_API_KEY",
        "GITHUB_TOKEN",
        "HF_ADAPTER_REPO",
        "ANTHROPIC_API_KEY",
    ],
}

OPTIONAL_KEYS = {
    "train": [
        "GITHUB_REPO",
        "NTFY_TOPIC",
        "GPU_TYPE",
        "CONTAINER_IMAGE",
        "WANDB_PROJECT",
        "WANDB_TAGS",
        "WANDB_NOTES",
        "WANDB_USERNAME",
        "BASE_MODEL_REVISION",
    ],
    "eval": [
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "GITHUB_REPO",
        "NTFY_TOPIC",
        "BASE_MODEL",
        "GPU_TYPE",
        "CONTAINER_IMAGE",
    ],
}


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
        return


def has_value(key: str) -> bool:
    return bool(os.environ.get(key, "").strip())


def check_target(target: str) -> list[str]:
    missing = [key for key in REQUIRED_KEYS[target] if not has_value(key)]

    print(f"[{target}]")
    for key in REQUIRED_KEYS[target]:
        status = "OK" if has_value(key) else "MISSING"
        print(f"  required  {key:<22} {status}")
    for key in OPTIONAL_KEYS[target]:
        status = "SET" if has_value(key) else "EMPTY"
        print(f"  optional  {key:<22} {status}")
    print()

    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local env readiness for RunPod workflows")
    parser.add_argument(
        "--target",
        choices=["train", "eval", "both"],
        default="both",
        help="Which workflow environment to validate",
    )
    args = parser.parse_args()

    load_env()

    targets = ["train", "eval"] if args.target == "both" else [args.target]
    missing_any: list[str] = []
    for target in targets:
        missing_any.extend(f"{target}:{key}" for key in check_target(target))

    if missing_any:
        print("[result] missing required keys")
        for item in missing_any:
            print(f"  - {item}")
        return 1

    print("[result] environment is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
