#!/usr/bin/env python3
"""Round-2 deterministic integrity checks before spending GPU money."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def check_tc_sft(failures: list[str]) -> None:
    path = ROOT / "sft_thread_conditioned.jsonl"
    eval_path = ROOT / "sft_thread_conditioned.eval.jsonl"
    rows = load_jsonl(path)
    eval_rows = load_jsonl(eval_path) if eval_path.exists() else []
    bad = [
        i for i, row in enumerate(rows, start=1)
        if not row.get("instruction") or not row.get("input") or not row.get("output")
    ]
    weighted = sum(1 for row in rows if float(row.get("loss_weight", 1.0) or 1.0) > 1.0)
    persona = sum(1 for row in rows if row.get("persona_id"))
    if len(rows) < 8000:
        fail(f"TC-SFT too small: rows={len(rows)} < 8000", failures)
    elif len(eval_rows) < 100:
        # Threshold updated post cycle-3 C1 eval thread holdout (1139→322 rows).
        # Paper-grade minimum is 100 (matches remove_val_train_leak.py MIN_EVAL_ROWS).
        fail(f"TC-SFT heldout eval too small: rows={len(eval_rows)} < 100", failures)
    elif bad:
        fail(f"TC-SFT malformed instruction/input/output rows: first={bad[:5]}", failures)
    elif weighted == 0:
        fail("TC-SFT has no loss_weight>1.0 rows", failures)
    elif persona / max(1, len(rows)) < 0.95:
        fail(f"TC-SFT persona coverage low: {persona}/{len(rows)}", failures)
    else:
        ok(f"TC-SFT schema rows={len(rows)} eval_rows={len(eval_rows)} weighted={weighted} persona={persona}")


def check_orpo_leak(failures: list[str]) -> None:
    # C5-1: audit v3 primary, fall back to v2 with warning; also include eval holdout.
    v3_path = ROOT / "val_set.v3.jsonl"
    v2_path = ROOT / "val_set.v2.jsonl"
    eval_path = ROOT / "sft_thread_conditioned.eval.jsonl"
    if v3_path.exists():
        val_source = v3_path
    elif v2_path.exists():
        print(f"[WARN] val_set.v3.jsonl not found; falling back to val_set.v2.jsonl for ORPO leak audit")
        val_source = v2_path
    else:
        fail("ORPO leak audit: neither val_set.v3.jsonl nor val_set.v2.jsonl found", failures)
        return
    val = {str(row.get("text") or "").strip() for row in load_jsonl(val_source)}
    if eval_path.exists():
        val |= {str(row.get("output") or "").strip() for row in load_jsonl(eval_path)}
        val |= {str(row.get("text") or "").strip() for row in load_jsonl(eval_path)}
    else:
        print("[WARN] sft_thread_conditioned.eval.jsonl not found; skipping eval holdout ORPO leak check")
    rows = load_jsonl(ROOT / "orpo_pairs.jsonl")
    exact_hits = [
        i for i, row in enumerate(rows, start=1)
        if str(row.get("chosen") or "").strip() in val
    ]
    chosen_sources = Counter(str(row.get("source_run_chosen") or "") for row in rows)
    if len(rows) < 500:
        fail(f"ORPO pairs too small: rows={len(rows)} < 500", failures)
    elif exact_hits:
        fail(f"ORPO chosen leaks exact val rows: count={len(exact_hits)} first={exact_hits[:5]}", failures)
    elif any("val_set" in source for source in chosen_sources):
        fail(f"ORPO chosen source references validation set: {dict(chosen_sources)}", failures)
    else:
        ok(f"ORPO leak audit pairs={len(rows)} chosen_sources={dict(chosen_sources)}")


def check_persona_gate_fixture(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        ai = td / "ai.jsonl"
        raw = td / "raw.jsonl"
        persona = td / "persona.json"
        out = td / "out.json"
        rows = [{"text": "안녕ㅋㅋ", "kind": "comment"} for _ in range(3)]
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        ai.write_text(payload, encoding="utf-8")
        raw.write_text(payload, encoding="utf-8")
        persona.write_text(
            json.dumps({"personas": [{"id": 1, "name": "p1"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/phase6_eval_v2.py",
                "--ai",
                str(ai),
                "--raw",
                str(raw),
                "--persona-list",
                str(persona),
                "--out",
                str(out),
                "--skip-mauve",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            fail("phase6_eval_v2 persona-missing fixture unexpectedly passed", failures)
            return
        report = json.loads(out.read_text(encoding="utf-8"))
        violations = report.get("overall", {}).get("violations", [])
        if not any("persona_consistency" in item for item in violations):
            fail(f"persona fixture failed for wrong reason: {violations}", failures)
        else:
            ok("phase6_eval_v2 rejects missing persona tags when persona-list is required")


def check_persona_identity_pass(failures: list[str]) -> None:
    src = ROOT / "sft_thread_conditioned.eval.jsonl"
    persona = ROOT / "runs" / "round2-obsidian-synthesis" / "persona-30-extracted.json"
    if not src.exists() or not persona.exists():
        fail("persona identity fixture inputs missing", failures)
        return
    rows = load_jsonl(src)[:64]
    if not rows:
        fail("persona identity fixture has no rows", failures)
        return
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        sample = td / "sample.jsonl"
        out = td / "out.json"
        sample.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/phase6_eval_v2.py",
                "--ai",
                str(sample),
                "--raw",
                str(sample),
                "--persona-list",
                str(persona),
                "--out",
                str(out),
                "--skip-mauve",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            fail(f"phase6_eval_v2 identity/persona fixture failed rc={proc.returncode}", failures)
            return
        report = json.loads(out.read_text(encoding="utf-8"))
        if report.get("overall", {}).get("verdict") != "PASS":
            fail(
                f"phase6_eval_v2 identity/persona fixture verdict={report.get('overall', {}).get('verdict')}",
                failures,
            )
        else:
            ok("phase6_eval_v2 passes identity/persona fixture on thread-conditioned eval rows")


def check_thread_holdout(failures: list[str]) -> None:
    """Rebuild safety gate: sft_train ∩ sft_eval root_ids = ∅ AND sft_train ∩ val thread_keys = ∅."""
    sft_path = ROOT / "sft_thread_conditioned.jsonl"
    eval_path = ROOT / "sft_thread_conditioned.eval.jsonl"
    val_path = ROOT / "val_set.v3.jsonl"

    def root_set(path: Path, keys: tuple[str, ...]) -> set[str]:
        out: set[str] = set()
        if not path.exists():
            return out
        for row in load_jsonl(path):
            for k in keys:
                v = row.get(k)
                if v:
                    out.add(str(v))
                    break
        return out

    sft_roots = root_set(sft_path, ("root_id", "thread_key"))
    eval_roots = root_set(eval_path, ("root_id", "thread_key"))
    val_roots = root_set(val_path, ("thread_key", "root_id"))

    overlap_eval = sft_roots & eval_roots
    overlap_val = sft_roots & val_roots
    if overlap_eval:
        fail(f"sft_eval root_id intersect sft_train: {len(overlap_eval)} (must be 0)", failures)
    if overlap_val:
        fail(f"val_set.v3 thread_key intersect sft_train: {len(overlap_val)} (must be 0)", failures)
    if not overlap_eval and not overlap_val:
        ok(f"thread holdout: sft_train={len(sft_roots)} eval={len(eval_roots)} val={len(val_roots)} intersect=0")


def main() -> int:
    failures: list[str] = []
    check_tc_sft(failures)
    check_orpo_leak(failures)
    check_thread_holdout(failures)
    check_persona_gate_fixture(failures)
    check_persona_identity_pass(failures)
    print(json.dumps({"verdict": "PASS" if not failures else "FAIL", "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
