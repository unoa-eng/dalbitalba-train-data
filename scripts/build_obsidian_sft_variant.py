#!/usr/bin/env python3
"""Build an opt-in SFT variant using Obsidian membership as a sampling signal.

This script does not modify prompts or targets. It only:
1. splits rows into matched/unseen subsets by source_id/root_id
2. produces a variant JSONL that mildly oversamples matched rows

The goal is to create a safe ablation path that preserves inference format.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


RNG = random.Random(7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-jsonl", type=Path, required=True)
    parser.add_argument("--obsidian-map", type=Path, required=True)
    parser.add_argument("--variant-out", type=Path, required=True)
    parser.add_argument("--matched-out", type=Path, required=True)
    parser.add_argument("--unseen-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.08,
        help="Desired matched-row ratio after oversampling; must stay conservative",
    )
    parser.add_argument(
        "--max-extra-rows",
        type=int,
        default=5000,
        help="Hard cap on duplicated rows added to the variant",
    )
    return parser.parse_args()


def load_style_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or {}
    return {str(key) for key in records.keys()}


def resolve_source_id(row: dict[str, Any]) -> str:
    for key in ("source_id", "root_id", "obsidian_source_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value.split(":", 1)[0]
    return ""


def matched_task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("task") or row.get("kind") or "unknown")] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    if not (0.05 <= args.target_ratio <= 0.12):
        raise SystemExit("--target-ratio must stay within 0.05..0.12 for conservative opt-in runs")
    keys = load_style_keys(args.obsidian_map)
    rows = [json.loads(line) for line in args.sft_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    matched: list[dict[str, Any]] = []
    unseen: list[dict[str, Any]] = []
    for row in rows:
        sid = resolve_source_id(row)
        if sid and sid in keys:
            tagged = dict(row)
            tagged["obsidian_matched"] = True
            matched.append(tagged)
        else:
            unseen.append(row)

    total_rows = len(rows)
    matched_rows = len(matched)
    current_ratio = matched_rows / total_rows if total_rows else 0.0
    duplicates_needed = 0
    if matched_rows and current_ratio < args.target_ratio:
        duplicates_needed = math.ceil((args.target_ratio * total_rows - matched_rows) / (1.0 - args.target_ratio))
        duplicates_needed = max(0, min(duplicates_needed, args.max_extra_rows))

    duplicates: list[dict[str, Any]] = []
    if duplicates_needed and matched:
        pool = list(matched)
        RNG.shuffle(pool)
        for idx in range(duplicates_needed):
            dup = dict(pool[idx % len(pool)])
            dup["obsidian_variant_duplicate"] = True
            duplicates.append(dup)

    variant_rows = list(rows) + duplicates

    for path in (args.variant_out, args.matched_out, args.unseen_out, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    for path, payload in (
        (args.variant_out, variant_rows),
        (args.matched_out, matched),
        (args.unseen_out, unseen),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    variant_matched_rows = matched_rows + len(duplicates)
    summary = {
        "input": str(args.sft_jsonl),
        "obsidian_map": str(args.obsidian_map),
        "rows": total_rows,
        "matched_rows": matched_rows,
        "unseen_rows": len(unseen),
        "matched_ratio": round(current_ratio, 6),
        "target_ratio": args.target_ratio,
        "duplicates_added": len(duplicates),
        "variant_rows": len(variant_rows),
        "variant_matched_rows": variant_matched_rows,
        "variant_matched_ratio": round((variant_matched_rows / len(variant_rows)) if variant_rows else 0.0, 6),
        "matched_by_task": matched_task_counts(matched),
        "duplicate_by_task": matched_task_counts(duplicates),
        "policy": {
            "prompt_mutation": False,
            "target_mutation": False,
            "membership_signal_only": True,
        },
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
