#!/usr/bin/env python3
"""Remove val-thread leakage from the v3 CPT patch and rebuild v3."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VAL_PATH = ROOT / "val_set.v2.jsonl"
V2_PATH = ROOT / "cpt_corpus.v2.jsonl"
PATCH_PATH = ROOT / "cpt_patch_gap_repair_dedup.jsonl"
V3_PATH = ROOT / "cpt_corpus.v3.jsonl"
REPORT_PATH = ROOT / "runs" / "val-leak-fix.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_thread_id(source_id: Any) -> str:
    return str(source_id).split(":", 1)[0]


def main() -> None:
    val_rows = load_jsonl(VAL_PATH)
    v2_rows = load_jsonl(V2_PATH)
    patch_rows = load_jsonl(PATCH_PATH)

    val_source_ids = {str(row["source_id"]) for row in val_rows}
    val_thread_ids = {canonical_thread_id(row["source_id"]) for row in val_rows}

    removed_rows: list[dict[str, Any]] = []
    cleaned_patch: list[dict[str, Any]] = []
    removal_reason_counts: Counter[str] = Counter()
    removed_kind_counts: Counter[str] = Counter()
    removed_thread_ids: set[str] = set()

    for row in patch_rows:
        source_id = str(row["source_id"])
        thread_id = canonical_thread_id(source_id)
        exact_match = source_id in val_source_ids
        thread_match = thread_id in val_thread_ids
        if exact_match or thread_match:
            reason = "exact_source_id" if exact_match else "thread_root_source_id"
            removal_reason_counts[reason] += 1
            removed_kind_counts[str(row.get("kind") or "unknown")] += 1
            removed_thread_ids.add(thread_id)
            removed_rows.append(
                {
                    "source_id": source_id,
                    "thread_id": thread_id,
                    "kind": row.get("kind"),
                    "source_field": row.get("source_field"),
                    "reason": reason,
                    "text": row.get("text"),
                }
            )
            continue
        cleaned_patch.append(row)

    rebuilt_v3 = [*v2_rows, *cleaned_patch]

    write_jsonl(PATCH_PATH, cleaned_patch)
    write_jsonl(V3_PATH, rebuilt_v3)

    residual_exact = 0
    residual_thread = 0
    for row in cleaned_patch:
        source_id = str(row["source_id"])
        thread_id = canonical_thread_id(source_id)
        if source_id in val_source_ids:
            residual_exact += 1
        if thread_id in val_thread_ids:
            residual_thread += 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {
            "val": str(VAL_PATH.relative_to(ROOT)),
            "v2": str(V2_PATH.relative_to(ROOT)),
            "patch": str(PATCH_PATH.relative_to(ROOT)),
            "v3": str(V3_PATH.relative_to(ROOT)),
        },
        "val": {
            "rows": len(val_rows),
            "unique_source_ids": len(val_source_ids),
            "unique_thread_ids": len(val_thread_ids),
        },
        "patch_before": {"rows": len(patch_rows)},
        "patch_removed": {
            "rows": len(removed_rows),
            "reason_counts": dict(removal_reason_counts),
            "kind_counts": dict(removed_kind_counts),
            "unique_threads": len(removed_thread_ids),
        },
        "patch_after": {
            "rows": len(cleaned_patch),
            "residual_exact_source_id_overlap": residual_exact,
            "residual_thread_overlap": residual_thread,
        },
        "v3_after": {
            "rows": len(rebuilt_v3),
            "expected_rows_from_v2_plus_patch_after": len(v2_rows) + len(cleaned_patch),
        },
        "removed_rows_preview": removed_rows[:25],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
