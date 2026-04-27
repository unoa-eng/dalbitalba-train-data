#!/usr/bin/env python3
"""
Adapter integrity sanity check (Mac mini, CPU-only, no inference).

Downloads `adapter_model.safetensors` and `adapter_config.json` from a HF repo
and verifies:
  - safetensors header parses (file is not truncated/corrupt)
  - adapter_config.json is well-formed JSON with `base_model_name_or_path` and `r`
  - tensor count > 0, total size sane (≥ 1 MiB, ≤ 2 GiB)

Does NOT load the base model. Does NOT run inference. Pure file integrity.

Usage:
    python3 scripts/check_adapter_integrity.py --repo UNOA/dalbitalba-qwen3-cpt-<stamp>
    python3 scripts/check_adapter_integrity.py --repo <repo> --json

Exit codes:
    0  PASS
    1  FAIL
    2  USAGE
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

MIN_SIZE = 1 * 1024 * 1024          # 1 MiB
MAX_SIZE = 2 * 1024 * 1024 * 1024   # 2 GiB


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


def parse_safetensors_header(path: Path) -> tuple[int, int]:
    """Return (tensor_count, total_size_bytes). Raises on malformed file."""
    with path.open("rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError("file too small to contain safetensors header")
        (header_size,) = struct.unpack("<Q", header_size_bytes)
        if header_size <= 0 or header_size > 100 * 1024 * 1024:
            raise ValueError(f"implausible header size: {header_size}")
        header_bytes = f.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("header truncated")
        header = json.loads(header_bytes.decode("utf-8"))
    tensors = {k: v for k, v in header.items() if not k.startswith("__")}
    return len(tensors), path.stat().st_size


def hf_download(repo_id: str, filename: str, token: str | None, dest: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("[FATAL] huggingface_hub not installed. pip install huggingface_hub")
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=str(dest),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="HF adapter repo (e.g. UNOA/dalbitalba-qwen3-cpt-<stamp>)")
    parser.add_argument("--filename", default="adapter_model.safetensors", help="adapter weights filename")
    parser.add_argument("--config-filename", default="adapter_config.json", help="adapter config filename")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    parser.add_argument("--json", action="store_true", help="emit JSON verdict")
    args = parser.parse_args()

    load_env()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    failures: list[str] = []
    passes: list[str] = []
    findings: dict = {"repo": args.repo}

    # Try several common adapter file paths
    candidate_paths = (args.filename, f"cpt-lora/{args.filename}", f"sft-lora/{args.filename}")
    weights_path: Path | None = None
    for cand in candidate_paths:
        try:
            weights_path = hf_download(args.repo, cand, hf_token, Path(args.cache_dir))
            findings["weights_resolved_filename"] = cand
            break
        except Exception as exc:  # noqa: BLE001
            findings.setdefault("download_attempts", []).append({"filename": cand, "error": str(exc)[:160]})

    if weights_path is None:
        failures.append(f"could not download adapter weights from {args.repo} (tried: {candidate_paths})")
    else:
        try:
            tensor_count, total_size = parse_safetensors_header(weights_path)
            findings["tensor_count"] = tensor_count
            findings["total_size_bytes"] = total_size
            if tensor_count <= 0:
                failures.append(f"safetensors has 0 tensors")
            else:
                passes.append(f"safetensors header parsed; {tensor_count} tensors")
            if total_size < MIN_SIZE:
                failures.append(f"adapter file too small: {total_size} bytes (< {MIN_SIZE})")
            elif total_size > MAX_SIZE:
                failures.append(f"adapter file too large: {total_size} bytes (> {MAX_SIZE})")
            else:
                passes.append(f"adapter size OK: {total_size / 1e6:.1f} MB")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"safetensors parse error: {exc}")

    # Adapter config sanity
    config_candidates = (args.config_filename, f"cpt-lora/{args.config_filename}", f"sft-lora/{args.config_filename}")
    config_path: Path | None = None
    for cand in config_candidates:
        try:
            config_path = hf_download(args.repo, cand, hf_token, Path(args.cache_dir))
            findings["config_resolved_filename"] = cand
            break
        except Exception:
            continue

    if config_path is None:
        failures.append(f"adapter_config.json not found in {args.repo}")
    else:
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            base = cfg.get("base_model_name_or_path", "")
            r = cfg.get("r", 0)
            findings["base_model"] = base
            findings["r"] = r
            if not base:
                failures.append("adapter_config.json missing base_model_name_or_path")
            elif "Qwen3-8B" not in base and "Qwen2.5" not in base:
                failures.append(f"unexpected base_model: {base}")
            else:
                passes.append(f"base_model OK: {base}")
            if r <= 0:
                failures.append(f"adapter_config.json invalid r={r}")
            else:
                passes.append(f"r={r}")
        except json.JSONDecodeError as exc:
            failures.append(f"adapter_config.json parse error: {exc}")

    verdict = "PASS" if not failures else "FAIL"
    findings["verdict"] = verdict
    findings["passes"] = passes
    findings["failures"] = failures

    if args.json:
        print(json.dumps(findings, indent=2, ensure_ascii=False))
    else:
        print(f"=== adapter integrity check — {verdict} ===")
        print(f"repo: {args.repo}")
        if "tensor_count" in findings:
            print(f"tensors: {findings['tensor_count']}, size: {findings['total_size_bytes'] / 1e6:.1f} MB")
        if "base_model" in findings:
            print(f"base_model: {findings['base_model']}, r: {findings['r']}")
        print()
        if passes:
            print("PASS:")
            for p in passes:
                print(f"  + {p}")
        if failures:
            print("FAIL:")
            for f in failures:
                print(f"  - {f}")
        print()
        print(f"VERDICT: {verdict}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
