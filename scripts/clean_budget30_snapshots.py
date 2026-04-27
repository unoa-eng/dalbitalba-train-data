#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CPT = REPO_ROOT / "cpt_corpus.v2.jsonl"
DEFAULT_SFT = REPO_ROOT / "sft_pairs.v2.jsonl"
DEFAULT_VAL = REPO_ROOT / "val_set.v2.jsonl"
DEFAULT_BACKUP_ROOT = REPO_ROOT / "archive" / "dataset-backups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize root v2 snapshots so the budget30 verifier can pass deterministically."
    )
    parser.add_argument("--cpt-path", default=str(DEFAULT_CPT))
    parser.add_argument("--sft-path", default=str(DEFAULT_SFT))
    parser.add_argument("--val-path", default=str(DEFAULT_VAL))
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=10,
        help="Drop CPT/val rows whose text is shorter than this after strip().",
    )
    parser.add_argument(
        "--backup-root",
        default=str(DEFAULT_BACKUP_ROOT),
        help="Parent directory for timestamped original-file backups.",
    )
    parser.add_argument(
        "--summary-path",
        help="Optional JSON path for the cleanup summary. Defaults under the backup dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text_rows(rows: list[dict[str, Any]], min_text_chars: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_texts: set[str] = set()

    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            stats["dropped_empty_text"] += 1
            continue
        if len(text) < min_text_chars:
            stats["dropped_short_text"] += 1
            continue
        if text in seen_texts:
            stats["dropped_duplicate_text"] += 1
            continue
        seen_texts.add(text)
        kept.append(row)

    return kept, stats


def clean_sft_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for row in rows:
        post = str(row.get("post") or "").strip()
        comment = str(row.get("comment") or "").strip()
        if not post or not comment:
            stats["dropped_empty_pair"] += 1
            continue
        pair = (post, comment)
        if pair in seen_pairs:
            stats["dropped_duplicate_pair"] += 1
            continue
        seen_pairs.add(pair)
        kept.append(row)

    return kept, stats


def backup_file(src: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / src.name
    shutil.copy2(src, dst)
    return dst


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def summarize_rows(kind: str, before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]], stats: Counter[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "before_rows": len(before_rows),
        "after_rows": len(after_rows),
        "dropped_rows": len(before_rows) - len(after_rows),
        "stats": dict(stats),
    }
    if kind in {"cpt", "val"}:
        before_short = sum(1 for row in before_rows if len(str(row.get("text") or "").strip()) < 20)
        after_short = sum(1 for row in after_rows if len(str(row.get("text") or "").strip()) < 20)
        payload["short_under_20_before"] = before_short
        payload["short_under_20_after"] = after_short
    return payload


def main() -> int:
    args = parse_args()
    stamp = utc_stamp()

    cpt_path = Path(args.cpt_path)
    sft_path = Path(args.sft_path)
    val_path = Path(args.val_path)
    backup_root = Path(args.backup_root)
    backup_dir = backup_root / f"budget30-clean-{stamp}"
    summary_path = Path(args.summary_path) if args.summary_path else backup_dir / "summary.json"

    cpt_before = load_jsonl(cpt_path)
    sft_before = load_jsonl(sft_path)
    val_before = load_jsonl(val_path)

    cpt_after, cpt_stats = clean_text_rows(cpt_before, args.min_text_chars)
    sft_after, sft_stats = clean_sft_rows(sft_before)
    val_after, val_stats = clean_text_rows(val_before, args.min_text_chars)

    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": args.dry_run,
        "config": {
            "min_text_chars": args.min_text_chars,
            "cpt_path": str(cpt_path),
            "sft_path": str(sft_path),
            "val_path": str(val_path),
            "backup_dir": str(backup_dir),
        },
        "files": {
            "cpt": summarize_rows("cpt", cpt_before, cpt_after, cpt_stats),
            "sft": summarize_rows("sft", sft_before, sft_after, sft_stats),
            "val": summarize_rows("val", val_before, val_after, val_stats),
        },
    }

    if not args.dry_run:
        backup_file(cpt_path, backup_dir)
        backup_file(sft_path, backup_dir)
        backup_file(val_path, backup_dir)
        write_jsonl(cpt_path, cpt_after)
        write_jsonl(sft_path, sft_after)
        write_jsonl(val_path, val_after)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
