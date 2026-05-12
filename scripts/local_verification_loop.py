#!/usr/bin/env python3
"""Local preflight/audit loop for dalbitalba training runs.

This script is intentionally local-first. It does not launch RunPod pods and it
does not upload to Hugging Face. It answers the question: "is this repo ready to
spend GPU money, and are the existing HF artifacts release-worthy?"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import py_compile
import re
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
ACTIVE_CPT_PATH = (
    REPO_ROOT / "cpt_corpus.v3.jsonl"
    if (REPO_ROOT / "cpt_corpus.v3.jsonl").exists()
    else REPO_ROOT / "cpt_corpus.v2.jsonl"
)
ACTIVE_SFT_PATH = (
    REPO_ROOT / "sft_thread_conditioned.jsonl"
    if (REPO_ROOT / "sft_thread_conditioned.jsonl").exists()
    else REPO_ROOT / "sft_pairs.v2.jsonl"
)
ACTIVE_EVAL_PATH = (
    REPO_ROOT / "sft_thread_conditioned.eval.jsonl"
    if (REPO_ROOT / "sft_thread_conditioned.eval.jsonl").exists()
    else REPO_ROOT / "val_set.v2.jsonl"
)
OBSIDIAN_REF_DIR = REPO_ROOT / "research" / "obsidian-ref"
OBSIDIAN_EXPORT_DIR = REPO_ROOT / "research" / "obsidian-export"
OBSIDIAN_PERSONA_PATH = REPO_ROOT / "runs" / "round2-obsidian-synthesis" / "persona-30-extracted.json"

DEFAULT_FILES = {
    "cpt": ACTIVE_CPT_PATH,
    "sft": ACTIVE_SFT_PATH,
    "eval": ACTIVE_EVAL_PATH,
    "cai": REPO_ROOT / "cai_pairs.filtered.jsonl",
}

PYTHON_SCRIPTS = [
    "train_cpt.py",
    "train_sft.py",
    "scripts/sft_format_smoke_test.py",
    "scripts/launch_train_pod.py",
    "scripts/launch_eval_pod.py",
    "scripts/phase6_generate.py",
    "scripts/phase6_eval.py",
    "scripts/phase6_eval_v2.py",
    "scripts/poll_pod.py",
    "scripts/recipe_mutator.py",
    "scripts/cycle_report.py",
]

REQUIRED_SHELL = [
    "chain_train.sh",
    "chain_train_round2.sh",
    "scripts/run_eval.sh",
    "scripts/autonomous_loop.sh",
]

LOCAL_SMOKE_SCRIPT = "scripts/sft_format_smoke_test.py"

PHONE_RE = re.compile(r"\b(?:\+?82[- ]?)?(?:0\d{1,2})[- .]?\d{3,4}[- .]?\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\bhttps?://|www\.", re.IGNORECASE)
RRN_RE = re.compile(r"\b\d{6}[-\s]?\d{7}\b")
BANK_NAMES = (
    r"(?:국민|신한|우리|하나|농협|기업|카카오뱅크|카뱅|케이뱅크|새마을|수협|"
    r"외환|SC제일|씨티|부산|대구|광주|전북|경남|우체국)"
)
ACCOUNT_RE = re.compile(
    rf"{BANK_NAMES}[^\d]{{0,20}}\d{{3,6}}[-\s]\d{{2,6}}[-\s]\d{{2,8}}(?:[-\s]\d{{2,6}})?"
)

MINOR_RE = re.compile(
    "|".join(
        re.escape(x)
        for x in [
            "미성년자",
            "미성년",
            "청소년",
            "초등학생",
            "초딩",
            "중학생",
            "중딩",
            "고등학생",
            "고딩",
            "여중생",
            "남중생",
            "여고생",
            "남고생",
        ]
    )
)
SEXUAL_RE = re.compile(
    "|".join(
        re.escape(x)
        for x in [
            "섹스",
            "성관계",
            "야동",
            "자위",
            "조건만남",
            "원나잇",
            "원나이트",
            "성매매",
            "오피",
            "풀살롱",
            "안마방",
            "대딸",
            "유사성행위",
            "오랄",
            "구강",
            "질내",
            "항문",
            "삽입",
        ]
    )
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[idx]


def compact_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "avg": round(statistics.mean(values), 2),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def encoding_profile(texts: list[str]) -> dict[str, Any]:
    total = sum(len(text) for text in texts)
    hangul = 0
    cjk = 0
    replacement = 0
    for text in texts:
        for ch in text:
            code = ord(ch)
            if 0xAC00 <= code <= 0xD7A3:
                hangul += 1
            elif 0x4E00 <= code <= 0x9FFF:
                cjk += 1
            elif ch == "\ufffd":
                replacement += 1
    return {
        "total_chars": total,
        "hangul_chars": hangul,
        "hangul_ratio": round(hangul / max(1, total), 4),
        "cjk_chars": cjk,
        "cjk_ratio": round(cjk / max(1, total), 4),
        "replacement_chars": replacement,
    }


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, [{"line": None, "error": "missing_file"}]

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line": line_no,
                        "error": exc.msg,
                        "column": exc.colno,
                    }
                )
                continue
            if not isinstance(value, dict):
                errors.append({"line": line_no, "error": "row_not_object"})
                continue
            rows.append(value)
    return rows, errors


def minor_sexual_proximity(text: str, window: int = 60) -> bool:
    minors = [m.start() for m in MINOR_RE.finditer(text)]
    if not minors:
        return False
    sexuals = [m.start() for m in SEXUAL_RE.finditer(text)]
    return any(abs(a - b) <= window for a in minors for b in sexuals)


def text_for_row(kind: str, row: dict[str, Any]) -> str:
    if kind == "cpt":
        return str(row.get("text") or "")
    if kind == "sft":
        if row.get("instruction") is not None or row.get("output") is not None:
            return "\n".join(
                part
                for part in [
                    str(row.get("instruction") or ""),
                    str(row.get("input") or ""),
                    str(row.get("output") or ""),
                    str(row.get("persona_id") or ""),
                ]
                if part
            )
        return "\n".join(
            part
            for part in [
                str(row.get("post") or ""),
                str(row.get("comment") or ""),
                str(row.get("instruction") or ""),
                str(row.get("input") or ""),
                str(row.get("output") or ""),
            ]
            if part
        )
    if kind == "eval":
        if row.get("instruction") is not None or row.get("output") is not None:
            return "\n".join(
                part
                for part in [
                    str(row.get("instruction") or ""),
                    str(row.get("input") or ""),
                    str(row.get("output") or ""),
                    str(row.get("persona_id") or ""),
                ]
                if part
            )
        return str(row.get("text") or "")
    return "\n".join(str(v) for v in row.values() if isinstance(v, str))


def validate_dataset(kind: str, path: Path) -> dict[str, Any]:
    rows, errors = load_jsonl(path)
    texts = [text_for_row(kind, row).strip() for row in rows]
    nonempty_texts = [text for text in texts if text]
    lengths = [len(text) for text in nonempty_texts]

    if kind == "cpt":
        required = {"text", "kind", "source_id", "source_field", "length_bucket"}
    elif kind == "sft":
        has_thread_schema = bool(rows and (rows[0].get("instruction") is not None or rows[0].get("output") is not None))
        if has_thread_schema:
            required = {"instruction", "input", "output", "persona_id", "loss_weight"}
        else:
            required = {"post", "comment", "thread_key", "source_id", "length_bucket"}
    elif kind == "eval":
        has_thread_schema = bool(rows and (rows[0].get("instruction") is not None or rows[0].get("output") is not None))
        if has_thread_schema:
            required = {"instruction", "input", "output", "persona_id"}
        else:
            required = {"text"}
    else:
        required = set()

    missing_required = 0
    key_counter: Counter[str] = Counter()
    for row in rows:
        key_counter.update(row.keys())
        if required and not required.issubset(row.keys()):
            missing_required += 1

    duplicate_count = len(nonempty_texts) - len(set(nonempty_texts))
    encoding = encoding_profile(nonempty_texts)
    pii = {
        "phone_like": sum(1 for text in nonempty_texts if PHONE_RE.search(text)),
        "email_like": sum(1 for text in nonempty_texts if EMAIL_RE.search(text)),
        "url_like": sum(1 for text in nonempty_texts if URL_RE.search(text)),
        "rrn_like": sum(1 for text in nonempty_texts if RRN_RE.search(text)),
        "account_like": sum(1 for text in nonempty_texts if ACCOUNT_RE.search(text)),
        "minor_sexual_proximity": sum(
            1 for text in nonempty_texts if minor_sexual_proximity(text)
        ),
    }

    kind_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for row in rows:
        if "kind" in row:
            kind_counts[str(row.get("kind"))] += 1
        if "length_bucket" in row:
            bucket_counts[str(row.get("length_bucket"))] += 1

    severe: list[str] = []
    warn: list[str] = []
    if errors:
        severe.append(f"{path.name}: invalid JSONL rows={len(errors)}")
    if missing_required:
        severe.append(f"{path.name}: missing required keys rows={missing_required}")
    if not rows:
        severe.append(f"{path.name}: no valid rows")
    if pii["phone_like"] or pii["email_like"] or pii["rrn_like"]:
        severe.append(f"{path.name}: possible direct PII remains {pii}")
    if pii["minor_sexual_proximity"]:
        severe.append(
            f"{path.name}: minor/sexual proximity rows={pii['minor_sexual_proximity']}"
        )
    if encoding["replacement_chars"]:
        severe.append(f"{path.name}: unicode replacement characters remain")
    if nonempty_texts and encoding["hangul_ratio"] < 0.15 and encoding["cjk_ratio"] > 0.10:
        severe.append(f"{path.name}: possible mojibake encoding corruption")
    if nonempty_texts:
        dup_rate = duplicate_count / len(nonempty_texts)
        if dup_rate > 0.50:
            severe.append(
                f"{path.name}: catastrophic duplicate rate={dup_rate:.3f} (>0.50) — training on this data is meaningless; regenerate corpus"
            )
        elif dup_rate > 0.35:
            warn.append(f"{path.name}: high duplicate rate={dup_rate:.3f}")
    if lengths and sum(1 for x in lengths if x < 20) / len(lengths) > 0.20:
        warn.append(f"{path.name}: many very short rows")

    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "rows": len(rows),
        "json_errors": errors[:20],
        "json_error_count": len(errors),
        "missing_required_rows": missing_required,
        "key_counts": dict(key_counter.most_common()),
        "kind_counts": dict(kind_counts.most_common()),
        "length_bucket_counts": dict(bucket_counts.most_common()),
        "char_stats": compact_stats(lengths),
        "approx_token_stats": compact_stats([max(1, round(x / 2.2)) for x in lengths]),
        "duplicates": duplicate_count,
        "duplicate_rate": round(duplicate_count / max(1, len(nonempty_texts)), 4),
        "encoding_profile": encoding,
        "pii_signals": pii,
        "warnings": warn,
        "severe": severe,
    }


def compile_scripts() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    severe: list[str] = []
    for rel in PYTHON_SCRIPTS:
        path = REPO_ROOT / rel
        if not path.exists():
            results.append({"path": rel, "ok": False, "error": "missing"})
            severe.append(f"missing python script: {rel}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            msg = str(exc).splitlines()[-1]
            results.append({"path": rel, "ok": False, "error": msg})
            severe.append(f"python compile failed: {rel}")
        else:
            results.append({"path": rel, "ok": True})

    for rel in REQUIRED_SHELL:
        path = REPO_ROOT / rel
        if not path.exists():
            results.append({"path": rel, "ok": False, "error": "missing"})
            severe.append(f"missing shell script: {rel}")
        else:
            results.append({"path": rel, "ok": True})

    return {"results": results, "severe": severe}


def read_file(rel: str) -> str:
    path = REPO_ROOT / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def verify_contract() -> dict[str, Any]:
    severe: list[str] = []
    warnings: list[str] = []
    chain = read_file("chain_train_round2.sh")
    run_eval = read_file("scripts/run_eval.sh")
    launch_eval = read_file("scripts/launch_eval_pod.py")
    launch_train = read_file("scripts/launch_train_pod.py")
    phase6_v2 = read_file("scripts/phase6_eval_v2.py")
    phase6_generate = read_file("scripts/phase6_generate.py")
    workflow_train = read_file(".github/workflows/runpod-train.yml")

    required_markers = [
        "cpt_corpus.v3.jsonl",
        "SFT_DATA",
        "phase6_eval_v2.py",
        "persist_run_artifacts",
    ]
    for marker in required_markers:
        if marker not in chain:
            severe.append(f"chain_train_round2.sh missing marker: {marker}")

    if "phase6_eval_v2.py" not in run_eval:
        severe.append("scripts/run_eval.sh does not execute phase6_eval_v2.py for phase6 mode")
    if "EVAL_INPUT_DATA" not in run_eval:
        severe.append("scripts/run_eval.sh does not wire EVAL_INPUT_DATA into phase6 evaluation")
    if "CPT_MERGED_REPO" not in run_eval or "CPT_MERGED_REPO" not in launch_eval:
        severe.append("eval launch/run scripts do not support CPT-only evaluation")
    if "CPT_MERGED_REPO" not in phase6_generate:
        severe.append("phase6_generate.py does not support CPT-only generation")
    if "min_rows" not in phase6_v2 or "row_count_mismatch" not in phase6_v2:
        severe.append("phase6_eval_v2.py does not enforce minimum sample/cardinality gates")
    if "total_rows" not in phase6_v2 or "missing" not in phase6_v2:
        severe.append("phase6_eval_v2.py does not count missing persona tags as failures")
    if "SFT_ADAPTER_REPO" not in launch_eval:
        warnings.append("launch_eval_pod.py does not expose SFT_ADAPTER_REPO explicitly")
    if "ALLOW_DIRTY_LAUNCH" not in launch_eval or "eval launch-critical local changes" not in launch_eval:
        severe.append("launch_eval_pod.py does not block dirty eval launch-critical local changes")
    if "tokenizer_v4" not in launch_train:
        severe.append("launch_train_pod.py does not copy tokenizer_v4 into the pod workspace")
    if "ALLOW_DIRTY_LAUNCH" not in launch_train or "launch-critical local changes" not in launch_train:
        severe.append("launch_train_pod.py does not block dirty launch-critical local changes")
    if "WANDB_API_KEY" not in workflow_train:
        severe.append(".github/workflows/runpod-train.yml does not pass WANDB_API_KEY")
    if "metrics.json" not in run_eval:
        severe.append("scripts/run_eval.sh does not copy metrics.json into run artifacts")

    return {"severe": severe, "warnings": warnings}


def verify_obsidian_scope() -> dict[str, Any]:
    """Verify Obsidian-derived reference data is present and wired as metadata.

    The accepted design keeps raw Obsidian markdown out of direct training and
    uses the curated persona/list artifacts as explicit metadata inputs. This
    gate makes that scope mechanical so a fresh clone cannot silently drop it.
    """
    severe: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    ref_files = sorted(OBSIDIAN_REF_DIR.rglob("*.md")) if OBSIDIAN_REF_DIR.exists() else []
    export_files = sorted(OBSIDIAN_EXPORT_DIR.rglob("*.md")) if OBSIDIAN_EXPORT_DIR.exists() else []

    if len(ref_files) < 10:
        severe.append(
            f"Obsidian reference scope too small: {len(ref_files)} markdown files under research/obsidian-ref"
        )
    if len(export_files) < 100:
        severe.append(
            f"Obsidian export scope too small: {len(export_files)} markdown files under research/obsidian-export"
        )

    persona_count = 0
    placeholder_count = 0
    persona_sources: Counter[str] = Counter()
    accepted_count: int | None = None
    if not OBSIDIAN_PERSONA_PATH.exists():
        severe.append(f"Obsidian persona list missing: {OBSIDIAN_PERSONA_PATH.relative_to(REPO_ROOT)}")
    else:
        try:
            payload = json.loads(OBSIDIAN_PERSONA_PATH.read_text(encoding="utf-8"))
            personas = payload.get("personas") or []
            if not isinstance(personas, list):
                raise ValueError("personas is not a list")
            persona_count = len(personas)
            accepted_count_raw = payload.get("personas_accepted_count")
            accepted_count = int(accepted_count_raw) if accepted_count_raw is not None else None
            for persona in personas:
                if not isinstance(persona, dict):
                    continue
                source = str(persona.get("source") or "unknown")
                persona_sources[source] += 1
                if "placeholder" in source or "placeholder" in str(persona.get("trait") or ""):
                    placeholder_count += 1
                for key in ("id", "name", "tone", "mood", "trait"):
                    if persona.get(key) in (None, ""):
                        severe.append(f"Obsidian persona row missing {key}: {persona}")
                        break
            if persona_count < 30:
                severe.append(f"Obsidian persona coverage below 30: {persona_count}")
            if accepted_count is not None and accepted_count < 30:
                severe.append(f"Obsidian personas_accepted_count below 30: {accepted_count}")
            if placeholder_count:
                notes.append(
                    f"Obsidian persona list contains {placeholder_count} synthesized placeholders; cycle-2 should replace from source persona JSON"
                )
        except Exception as exc:
            severe.append(f"Obsidian persona list unreadable: {type(exc).__name__}: {exc}")

    recipe = read_file("recipes/round2-cycle1.env")
    chain = read_file("chain_train_round2.sh")
    run_eval = read_file("scripts/run_eval.sh")
    launch_eval = read_file("scripts/launch_eval_pod.py")
    required_refs = [
        "runs/round2-obsidian-synthesis/persona-30-extracted.json",
        "SFT_PERSONA_LIST",
        "EVAL_PERSONA_LIST",
    ]
    for marker in required_refs:
        if marker not in recipe and marker not in chain and marker not in run_eval and marker not in launch_eval:
            severe.append(f"Obsidian/persona wiring missing marker: {marker}")

    return {
        "paths": {
            "ref_dir": str(OBSIDIAN_REF_DIR.relative_to(REPO_ROOT)),
            "export_dir": str(OBSIDIAN_EXPORT_DIR.relative_to(REPO_ROOT)),
            "persona_list": str(OBSIDIAN_PERSONA_PATH.relative_to(REPO_ROOT)),
        },
        "ref_markdown_files": len(ref_files),
        "export_markdown_files": len(export_files),
        "persona_count": persona_count,
        "personas_accepted_count": accepted_count,
        "persona_sources": dict(persona_sources.most_common()),
        "placeholder_count": placeholder_count,
        "severe": severe,
        "warnings": warnings,
        "notes": notes,
    }


def python_can_import(python_exe: str, module: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [python_exe, "-c", f"import {module}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0, proc.stderr[-1000:]


def resolve_smoke_python() -> tuple[str | None, list[dict[str, Any]], list[str]]:
    candidates: list[str] = []
    for candidate in [sys.executable, str(REPO_ROOT / ".venv" / "bin" / "python")]:
        if candidate not in candidates and Path(candidate).exists():
            candidates.append(candidate)

    probes: list[dict[str, Any]] = []
    warnings: list[str] = []
    for candidate in candidates:
        ok, stderr_tail = python_can_import(candidate, "transformers")
        probes.append(
            {
                "python": candidate,
                "module": "transformers",
                "ok": ok,
                "stderr_tail": stderr_tail,
            }
        )
        if ok:
            if candidate != sys.executable:
                warnings.append(
                    f"local smoke used fallback interpreter {candidate} because current interpreter lacks transformers"
                )
            return candidate, probes, warnings
    return None, probes, warnings


def run_local_smoke() -> dict[str, Any]:
    severe: list[str] = []
    warnings: list[str] = []
    runner, probes, probe_warnings = resolve_smoke_python()
    warnings.extend(probe_warnings)
    results: list[dict[str, Any]] = []

    if not Path(REPO_ROOT / LOCAL_SMOKE_SCRIPT).exists():
        severe.append(f"missing smoke script: {LOCAL_SMOKE_SCRIPT}")
        return {
            "results": results,
            "probes": probes,
            "severe": severe,
            "warnings": warnings,
        }

    if runner is None:
        severe.append(
            "no python interpreter with transformers available for local smoke; install deps in the repo venv"
        )
        return {
            "results": results,
            "probes": probes,
            "severe": severe,
            "warnings": warnings,
        }

    cmd = [runner, LOCAL_SMOKE_SCRIPT]
    tokenizer_path = REPO_ROOT / "tokenizer_v4"
    if tokenizer_path.is_dir():
        cmd.extend(["--tokenizer", str(tokenizer_path)])
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    results.append(
        {
            "name": "sft_format_smoke",
            "python": runner,
            "cmd": cmd,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    )
    if proc.returncode != 0:
        severe.append("local smoke failed: scripts/sft_format_smoke_test.py")
    return {
        "results": results,
        "probes": probes,
        "severe": severe,
        "warnings": warnings,
    }


def hf_get_json(path: str, token: str | None) -> Any:
    headers = {"User-Agent": "dalbitalba-local-verifier/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(path, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def hf_get_raw(path: str, token: str | None) -> str:
    headers = {"User-Agent": "dalbitalba-local-verifier/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(path, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def audit_hf_repo(repo_id: str, token: str | None, kind: str) -> dict[str, Any]:
    report: dict[str, Any] = {"repo": repo_id, "kind": kind, "severe": [], "warnings": []}
    try:
        info = hf_get_json(f"https://huggingface.co/api/models/{repo_id}?blobs=true", token)
    except urllib.error.HTTPError as exc:
        report["severe"].append(f"HF {kind} repo unreadable: HTTP {exc.code}")
        return report
    except Exception as exc:
        report["severe"].append(f"HF {kind} repo unreadable: {type(exc).__name__}")
        return report

    siblings = info.get("siblings") or []
    files = {item.get("rfilename"): item for item in siblings}
    report.update(
        {
            "private": info.get("private"),
            "created_at": info.get("createdAt"),
            "last_modified": info.get("lastModified"),
            "sha": info.get("sha"),
            "file_count": len(files),
            "files": {
                name: {
                    "size": meta.get("size")
                    or ((meta.get("lfs") or {}).get("size") if isinstance(meta.get("lfs"), dict) else None)
                }
                for name, meta in sorted(files.items())
            },
        }
    )

    if kind == "cpt":
        if "adapter_model.safetensors" not in files and "cpt-lora/adapter_model.safetensors" not in files:
            report["severe"].append("CPT adapter_model.safetensors missing")
        if "last-checkpoint/trainer_state.json" not in files:
            report["severe"].append("CPT trainer_state.json missing")
        else:
            try:
                raw = hf_get_raw(
                    f"https://huggingface.co/{repo_id}/raw/main/last-checkpoint/trainer_state.json",
                    token,
                )
                state = json.loads(raw)
                global_step = int(state.get("global_step") or 0)
                max_steps = int(state.get("max_steps") or 0)
                report["trainer_state"] = {
                    "global_step": global_step,
                    "max_steps": max_steps,
                    "epoch": state.get("epoch"),
                    "completion_ratio": round(global_step / max(1, max_steps), 4),
                    "last_eval_loss": last_eval_loss(state.get("log_history") or []),
                }
                if max_steps and global_step < max_steps:
                    report["severe"].append(
                        f"CPT incomplete: global_step={global_step} max_steps={max_steps}"
                    )
            except Exception as exc:
                report["severe"].append(f"CPT trainer_state unreadable: {type(exc).__name__}")
    elif kind == "sft":
        if "adapter_model.safetensors" not in files and "sft-lora/adapter_model.safetensors" not in files:
            report["severe"].append("SFT adapter_model.safetensors missing")
        if len(files) <= 1:
            report["severe"].append("SFT repo is effectively empty")

    return report


def last_eval_loss(history: list[Any]) -> float | None:
    value: float | None = None
    for item in history:
        if isinstance(item, dict) and isinstance(item.get("eval_loss"), (int, float)):
            value = float(item["eval_loss"])
    return value


def estimate_training_cost(
    cpt_rows: int,
    sft_raw_rows: int,
    sft_pair_rows: int,
    sec_per_step: float,
    hourly_usd: float,
    batch: int = 16,
    cpt_epochs: int = 1,
    sft_epochs: int = 2,
    sft_raw_ratio: float = 0.8,
    cpt_max_steps: int = 0,
    sft_max_steps: int = 0,
) -> dict[str, Any]:
    cpt_steps = math.ceil(cpt_rows / batch) * cpt_epochs
    if cpt_max_steps > 0:
        cpt_steps = min(cpt_steps, cpt_max_steps)
    if sft_epochs <= 0:
        sft_pair_take = 0
        sft_rows = 0
        sft_steps = 0
    elif sft_raw_ratio <= 0:
        sft_pair_take = sft_pair_rows
        sft_rows = sft_pair_take
        sft_steps = math.ceil(sft_rows / batch) * sft_epochs
    elif sft_raw_ratio >= 1:
        sft_pair_take = 0
        sft_rows = sft_raw_rows + sft_pair_take
        sft_steps = math.ceil(sft_rows / batch) * sft_epochs
    else:
        sft_pair_take = min(
            sft_pair_rows,
            int(round(sft_raw_rows * (1 - sft_raw_ratio) / max(sft_raw_ratio, 1e-6))),
        )
        sft_rows = sft_raw_rows + sft_pair_take
        sft_steps = math.ceil(sft_rows / batch) * sft_epochs
    if sft_max_steps > 0:
        sft_steps = min(sft_steps, sft_max_steps)
    cpt_hours = cpt_steps * sec_per_step / 3600
    sft_hours = sft_steps * sec_per_step / 3600
    return {
        "sec_per_step": sec_per_step,
        "hourly_usd": hourly_usd,
        "effective_batch": batch,
        "cpt_steps": cpt_steps,
        "cpt_hours": round(cpt_hours, 2),
        "cpt_usd": round(cpt_hours * hourly_usd, 2),
        "sft_pair_rows_used": sft_pair_take,
        "sft_train_rows": sft_rows,
        "sft_steps": sft_steps,
        "sft_hours": round(sft_hours, 2),
        "sft_usd": round(sft_hours * hourly_usd, 2),
        "total_train_hours": round(cpt_hours + sft_hours, 2),
        "total_train_usd": round((cpt_hours + sft_hours) * hourly_usd, 2),
    }


def parse_recipe_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def int_env(env: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(env.get(key, str(default)) or default)
    except ValueError:
        return default


def float_env(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, str(default)) or default)
    except ValueError:
        return default


def pick_recipe_path(profile: str) -> Path:
    mapping = {
        "default": REPO_ROOT / "recipes" / "round2-cycle1.env",
        "budget30": REPO_ROOT / "recipes" / "budget30.env",
        "smoke": REPO_ROOT / "recipes" / "smoke.env",
    }
    return mapping.get(profile, mapping["default"])


def write_reports(payload: dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    latest = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "timestamp": payload["timestamp"],
        "verdict": payload["verdict"],
        "severe_count": len(payload["severe"]),
        "warning_count": len(payload["warnings"]),
        "profile": (payload.get("recipe") or {}).get("profile"),
        "recipe": (payload.get("recipe") or {}).get("path"),
    }
    (RUNS_DIR / "latest-local-verification.json").write_text(
        json.dumps(latest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Local Verification Report",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Severe: `{len(payload['severe'])}`",
        f"- Warnings: `{len(payload['warnings'])}`",
    ]
    if payload.get("recipe"):
        lines.append(f"- Recipe: `{payload['recipe']['path']}` (`{payload['recipe']['profile']}`)")
    lines.extend([
        "",
        "## Severe Findings",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["severe"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in payload["warnings"] or ["None"])
    if payload.get("smoke"):
        lines.extend(["", "## Local Smoke", ""])
        for result in payload["smoke"].get("results", []):
            status = "PASS" if result.get("ok") else "FAIL"
            lines.append(f"- {result['name']}: `{status}` via `{result['python']}`")
        for probe in payload["smoke"].get("probes", []):
            status = "ok" if probe.get("ok") else "missing"
            lines.append(f"- Probe `{probe['python']}` import `{probe['module']}`: `{status}`")
    if payload.get("obsidian"):
        obsidian = payload["obsidian"]
        lines.extend(["", "## Obsidian Scope", ""])
        lines.append(f"- Reference markdown files: `{obsidian.get('ref_markdown_files')}`")
        lines.append(f"- Export markdown files: `{obsidian.get('export_markdown_files')}`")
        lines.append(f"- Persona count: `{obsidian.get('persona_count')}`")
        lines.append(f"- Placeholder personas: `{obsidian.get('placeholder_count')}`")
    lines.extend(["", "## Datasets", ""])
    lines.append("| Name | Rows | JSON Errors | Dup Rate | Hangul % | Avg Chars | P95 Chars | PII Signals |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, item in payload["datasets"].items():
        pii_count = sum(int(v) for v in item["pii_signals"].values())
        stats = item["char_stats"]
        encoding = item.get("encoding_profile") or {}
        hangul_pct = round(float(encoding.get("hangul_ratio", 0.0)) * 100, 2)
        lines.append(
            f"| {name} | {item['rows']} | {item['json_error_count']} | "
            f"{item['duplicate_rate']:.4f} | {hangul_pct} | {stats.get('avg', 'N/A')} | "
            f"{stats.get('p95', 'N/A')} | {pii_count} |"
        )
    if payload.get("cost_estimate"):
        cost = payload["cost_estimate"]
        lines.extend(
            [
                "",
                "## Cost Estimate",
                "",
                f"- CPT: `{cost['cpt_hours']}h`, `${cost['cpt_usd']}`",
                f"- SFT: `{cost['sft_hours']}h`, `${cost['sft_usd']}`",
                f"- Total train: `{cost['total_train_hours']}h`, `${cost['total_train_usd']}`",
            ]
        )
    if payload.get("hf"):
        lines.extend(["", "## Hugging Face Artifacts", ""])
        for key, item in payload["hf"].items():
            lines.append(f"### {key}")
            lines.append("")
            lines.append(f"- Repo: `{item.get('repo')}`")
            lines.append(f"- Files: `{item.get('file_count', 'N/A')}`")
            if item.get("trainer_state"):
                state = item["trainer_state"]
                lines.append(
                    f"- Trainer: `{state['global_step']}/{state['max_steps']}` "
                    f"({state['completion_ratio']})"
                )
            if item.get("severe"):
                lines.extend(f"- Severe: {x}" for x in item["severe"])
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-cpt-repo", default=os.environ.get("HF_CPT_REPO", ""))
    parser.add_argument("--hf-sft-repo", default=os.environ.get("HF_SFT_REPO", ""))
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--sec-per-step", type=float, default=18.43)
    parser.add_argument("--hourly-usd", type=float, default=0.79)
    parser.add_argument("--budget-usd", type=float, default=30.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("default", "budget30", "smoke"),
        default=os.environ.get("BUDGET_PROFILE", "default"),
        help="when 'budget30', a budget overrun escalates from WARN to SEVERE",
    )
    args = parser.parse_args()

    datasets = {name: validate_dataset(name, path) for name, path in DEFAULT_FILES.items()}
    compile_report = compile_scripts()
    contract = verify_contract()
    obsidian = verify_obsidian_scope()
    smoke = run_local_smoke()

    recipe_path = pick_recipe_path(args.profile)
    recipe_env = parse_recipe_env(recipe_path)

    severe: list[str] = []
    warnings: list[str] = []
    for item in datasets.values():
        severe.extend(item["severe"])
        warnings.extend(item["warnings"])
    severe.extend(compile_report["severe"])
    severe.extend(contract["severe"])
    severe.extend(obsidian["severe"])
    severe.extend(smoke["severe"])
    warnings.extend(contract["warnings"])
    warnings.extend(obsidian["warnings"])
    warnings.extend(smoke["warnings"])

    cpt_rows = datasets["cpt"]["rows"]
    sft_rows = datasets["sft"]["rows"]
    eval_rows = datasets["eval"]["rows"]
    default_sft_raw_ratio = 0.0 if ACTIVE_SFT_PATH.name == "sft_thread_conditioned.jsonl" else 0.8
    cpt_rows_for_cost = min(
        cpt_rows,
        int_env(recipe_env, "CPT_LIMIT_ROWS", cpt_rows) or cpt_rows,
    )
    sft_raw_rows_for_cost = min(
        cpt_rows,
        int_env(recipe_env, "SFT_RAW_LIMIT_ROWS", cpt_rows) or cpt_rows,
    )
    sft_pair_rows_for_cost = min(
        sft_rows,
        int_env(recipe_env, "SFT_PAIR_LIMIT_ROWS", sft_rows) or sft_rows,
    )
    cpt_epochs = int_env(recipe_env, "CPT_NUM_EPOCHS", 1)
    sft_epochs = int_env(recipe_env, "SFT_NUM_EPOCHS", 2)
    if recipe_env.get("SKIP_SFT") == "1":
        sft_epochs = 0
    cost = estimate_training_cost(
        cpt_rows=cpt_rows_for_cost,
        sft_raw_rows=sft_raw_rows_for_cost,
        sft_pair_rows=sft_pair_rows_for_cost,
        sec_per_step=args.sec_per_step,
        hourly_usd=args.hourly_usd,
        cpt_epochs=cpt_epochs,
        sft_epochs=sft_epochs,
        sft_raw_ratio=float_env(recipe_env, "SFT_RAW_RATIO", default_sft_raw_ratio),
        cpt_max_steps=int_env(recipe_env, "CPT_MAX_STEPS", 0),
        sft_max_steps=int_env(recipe_env, "SFT_MAX_STEPS", 0),
    )
    # Profile-aware budget gating.
    # default: full CPT+SFT vs budget (warn only, since the operator may pick a sub-profile).
    # budget30: CPT-only — overrun is SEVERE (a paid run under this profile must fit $30).
    # smoke: budget bound is much smaller; overrun is SEVERE.
    profile = args.profile
    if profile == "budget30":
        duplicate_warnings = [item for item in warnings if "high duplicate rate" in item]
        if duplicate_warnings:
            severe.extend(f"[budget30] {item}" for item in duplicate_warnings)
    if profile == "budget30":
        cpt_cost = cost["cpt_usd"]
        if cpt_cost > args.budget_usd:
            severe.append(
                f"[budget30] estimated CPT cost ${cpt_cost} exceeds ceiling ${args.budget_usd}; refuse to launch"
            )
        elif cpt_cost > args.budget_usd * 0.92:
            warnings.append(
                f"[budget30] CPT cost ${cpt_cost} ≥ 92% of ${args.budget_usd}; safety margin almost gone"
            )
    elif profile == "smoke":
        if cost["cpt_usd"] > 8:
            severe.append(
                f"[smoke] CPT cost estimate ${cost['cpt_usd']} > $8 ceiling for plumbing test"
            )
    else:
        if cost["total_train_usd"] > args.budget_usd:
            warnings.append(
                f"estimated full train cost ${cost['total_train_usd']} exceeds budget ${args.budget_usd}; use a smoke or CPT-only budget recipe"
            )
        elif cost["total_train_usd"] > args.budget_usd * 0.85:
            warnings.append(
                f"estimated train cost ${cost['total_train_usd']} is close to budget ${args.budget_usd}"
            )
    if eval_rows < 500:
        warnings.append("validation set is small for stable generation metrics")

    hf: dict[str, Any] = {}
    token = os.environ.get(args.hf_token_env)
    if args.hf_cpt_repo:
        hf["cpt"] = audit_hf_repo(args.hf_cpt_repo, token, "cpt")
        severe.extend(hf["cpt"].get("severe", []))
        warnings.extend(hf["cpt"].get("warnings", []))
    if args.hf_sft_repo:
        hf["sft"] = audit_hf_repo(args.hf_sft_repo, token, "sft")
        severe.extend(hf["sft"].get("severe", []))
        warnings.extend(hf["sft"].get("warnings", []))

    verdict = "FAIL" if severe else ("WARN" if warnings else "PASS")
    payload = {
        "timestamp": utc_now(),
        "verdict": verdict,
        "severe": severe,
        "warnings": warnings,
        "datasets": datasets,
        "scripts": compile_report,
        "contract": contract,
        "obsidian": obsidian,
        "smoke": smoke,
        "recipe": {
            "path": str(recipe_path.relative_to(REPO_ROOT)),
            "profile": args.profile,
            "env": recipe_env,
        },
        "cost_estimate": cost,
        "hf": hf,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = RUNS_DIR / f"local-verification-{stamp}"
    write_reports(payload, run_dir)

    print(json.dumps({"verdict": verdict, "report": str(run_dir / "report.md")}, ensure_ascii=False))
    if severe and args.strict:
        return 2
    return 0 if not severe else 2


if __name__ == "__main__":
    sys.exit(main())
