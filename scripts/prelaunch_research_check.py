#!/usr/bin/env python3
"""Paper-grade prelaunch checks for round2 training."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "round2-cycle1.env"
SELECTED_BASE = "Qwen/Qwen3-8B-Base"
REQUIRED_DOCS = ("MODEL_SELECTION.md", "RESEARCH_PROTOCOL.md")
REQUIRED_WANDB_KEYS = ("WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_TAGS")
TOKENIZER_TERMS = [
    "TC", "밀빵", "쩜오", "텐카", "케어", "초이스", "강남", "이태원",
    "ㅋㅋㅋㅋ", "ㅠㅠ", "ㄹㅇ", "ㅇㅈ", "ㅈㄴ", "ㅅㅂ",
]

# Profiles that require a pinned revision (FAIL, not WARN)
STRICT_REVISION_PROFILES = {"paper8b", "budget30"}


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def check_docs(failures: list[str]) -> None:
    for rel in REQUIRED_DOCS:
        path = ROOT / rel
        if not path.exists():
            fail(f"missing required protocol doc: {rel}", failures)
        else:
            ok(f"protocol doc exists: {rel}")


def check_recipe(failures: list[str]) -> dict[str, str]:
    env = parse_env(RECIPE)
    if env.get("BASE_MODEL") != SELECTED_BASE:
        fail(f"BASE_MODEL must be {SELECTED_BASE}, got {env.get('BASE_MODEL')}", failures)
    else:
        ok(f"BASE_MODEL selected: {SELECTED_BASE}")
    for key in REQUIRED_WANDB_KEYS:
        if not env.get(key):
            fail(f"recipe missing W&B key: {key}", failures)
        else:
            ok(f"recipe W&B key present: {key}={env[key]}")
    if env.get("BUDGET_PROFILE") == "budget30":
        fail("paper-grade recipe must not claim budget30 after selecting 8B full-chain", failures)
    return env


def check_tokenizer(failures: list[str]) -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot import transformers for tokenizer audit: {exc}", failures)
        return
    tokenizer = AutoTokenizer.from_pretrained(SELECTED_BASE, trust_remote_code=True, use_fast=True)
    counts = {term: len(tokenizer.encode(term, add_special_tokens=False)) for term in TOKENIZER_TERMS}
    worst = max(counts.values())
    if worst > 8:
        fail(f"tokenizer fragmentation too high: {counts}", failures)
    else:
        ok(f"tokenizer audit max_tokens={worst} counts={counts}")


def check_required_artifacts(failures: list[str]) -> None:
    required = [
        "cpt_enriched.jsonl",
        "cpt_corpus.v3.jsonl",
        "sft_thread_conditioned.jsonl",
        "sft_thread_conditioned.eval.jsonl",
        "orpo_pairs.jsonl",
        "val_set.v2.jsonl",
        "tokenizer_v4/tokenizer.json",
        "scripts/round2_integrity_check.py",
        "scripts/phase6_eval_v2.py",
    ]
    manifest: dict[str, dict[str, int | str]] = {}
    for rel in required:
        path = ROOT / rel
        if not path.exists():
            fail(f"missing required artifact: {rel}", failures)
            continue
        rows = 0
        if path.suffix == ".jsonl":
            rows = sum(1 for line in path.open("rb") if line.strip())
        manifest[rel] = {"bytes": path.stat().st_size, "rows": rows, "sha256": sha256(path)}
    if len(manifest) == len(required):
        ok("required artifact hashes computed")
    out = ROOT / ".state" / "round2_research_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def check_phase6_generate_alignment(failures: list[str]) -> None:
    text = (ROOT / "scripts" / "phase6_generate.py").read_text(encoding="utf-8")
    if "persona_id" not in text:
        fail("phase6_generate.py does not persist persona_id; persona gate cannot prove persona conditioning", failures)
    if re.search(r"return direct_text\\[:40\\]", text):
        fail("phase6_generate.py still uses first-40-char seed eval; not paper-grade task prompt", failures)
    else:
        ok("phase6_generate.py no longer appears to use first-40-char seed eval")


def check_base_model_revision(failures: list[str]) -> None:
    """Fail when BASE_MODEL_REVISION is unset or 'main' for strict profiles (paper8b/budget30).

    Other profiles (smoke, etc.) still emit WARN-only so they remain fast-path.
    Known good Qwen3-8B-Base HF HEAD (verified 2026-05-12):
      49e3418fbbbca6ecbdf9608b4d22e5a407081db4
    """
    revision = os.environ.get("BASE_MODEL_REVISION", "").strip()
    budget_profile = os.environ.get("BUDGET_PROFILE", "").strip().lower()

    # Also parse from recipe file when env vars are absent (common in local prelaunch runs)
    if RECIPE.exists():
        for line in RECIPE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key == "BUDGET_PROFILE" and not budget_profile:
                budget_profile = val.lower()
            if key == "BASE_MODEL_REVISION" and not revision:
                revision = val

    is_strict = budget_profile in STRICT_REVISION_PROFILES

    if not revision or revision == "main":
        msg = (
            f"BASE_MODEL_REVISION is {'not set' if not revision else repr(revision)} "
            f"(floating ref). Pin to a commit SHA for reproducibility. "
            f"Known good Qwen3-8B-Base HEAD: 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 "
            f"(verified 2026-05-12)."
        )
        if is_strict:
            fail(
                f"[strict profile '{budget_profile}'] {msg}",
                failures,
            )
        else:
            print(f"[WARN] {msg}")
    else:
        ok(f"BASE_MODEL_REVISION pinned: {revision}")


def check_runtime_wandb(failures: list[str]) -> None:
    # Paper-grade gate: WANDB_API_KEY is mandatory at launch time.
    # No env-var bypass is permitted — the gate cannot be silently waived in CI/RunPod.
    if not os.environ.get("WANDB_API_KEY"):
        fail("WANDB_API_KEY missing in runtime env; paper-grade launch requires W&B", failures)
    else:
        ok("WANDB_API_KEY present in runtime env")


def main() -> int:
    failures: list[str] = []
    check_docs(failures)
    check_recipe(failures)
    check_base_model_revision(failures)
    check_tokenizer(failures)
    check_required_artifacts(failures)
    check_phase6_generate_alignment(failures)
    check_runtime_wandb(failures)
    result = {"verdict": "PASS" if not failures else "FAIL", "failures": failures}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
