#!/usr/bin/env python3
"""
Create a blind evaluation set by mixing generated AI samples with human crawl data.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


TEXT_KEYS = ("text", "content", "body", "post_text", "generated_text", "raw_text")
TITLE_KEYS = ("title", "subject", "headline")
TOPIC_KEYS = ("topic", "cluster", "category")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pick_text(row: dict) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pick_meta(row: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize(rows: list[dict], truth: str, source: str) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        text = pick_text(row)
        if not text:
            continue
        normalized.append(
            {
                "source_id": row.get("id") or row.get("source_id"),
                "truth": truth,
                "source": source,
                "title": pick_meta(row, TITLE_KEYS),
                "topic": pick_meta(row, TOPIC_KEYS),
                "kind": row.get("kind") or row.get("pair_type") or "post",
                "text": text,
            }
        )
    return normalized


def stratified_sample(ai_rows: list[dict], human_rows: list[dict], n: int) -> list[dict]:
    ai_by_kind: dict[str, list[dict]] = defaultdict(list)
    human_by_kind: dict[str, list[dict]] = defaultdict(list)

    for row in ai_rows:
        ai_by_kind[str(row.get("kind") or "post")].append(row)
    for row in human_rows:
        human_by_kind[str(row.get("kind") or "post")].append(row)

    common_kinds = sorted(set(ai_by_kind) & set(human_by_kind))
    if not common_kinds:
        sample_size = min(n, len(ai_rows), len(human_rows))
        if sample_size == 0:
            return []
        return random.sample(ai_rows, sample_size) + random.sample(human_rows, sample_size)

    target_by_kind = {kind: n // len(common_kinds) for kind in common_kinds}
    for kind in common_kinds[: n % len(common_kinds)]:
        target_by_kind[kind] += 1

    kind_limits = {
        kind: min(target_by_kind[kind], len(ai_by_kind[kind]), len(human_by_kind[kind]))
        for kind in common_kinds
    }

    remaining = n - sum(kind_limits.values())
    while remaining > 0:
        grew = False
        for kind in common_kinds:
            max_available = min(len(ai_by_kind[kind]), len(human_by_kind[kind]))
            if kind_limits[kind] < max_available:
                kind_limits[kind] += 1
                remaining -= 1
                grew = True
                if remaining == 0:
                    break
        if not grew:
            break

    combined: list[dict] = []
    for kind in common_kinds:
        take = kind_limits[kind]
        if take <= 0:
            continue
        combined.extend(random.sample(ai_by_kind[kind], take))
        combined.extend(random.sample(human_by_kind[kind], take))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Create blind eval samples")
    parser.add_argument("--ai-output", required=True, type=Path)
    parser.add_argument("--crawl", required=True, type=Path)
    parser.add_argument("--n", type=int, default=30, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--answer-key", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    ai_rows = normalize(read_jsonl(args.ai_output), "AI", "generated")
    human_rows = normalize(read_jsonl(args.crawl), "HUMAN", "crawl")

    combined = stratified_sample(ai_rows, human_rows, args.n)
    if not combined:
        raise SystemExit("[ERROR] insufficient AI or HUMAN samples")

    random.shuffle(combined)

    answer_key: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answer_key_path = args.answer_key or args.output.with_name("eval_key.json")

    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(combined, start=1):
            record = {
                "id": index,
                "title": row["title"],
                "topic": row["topic"],
                "kind": row.get("kind"),
                "text": row["text"],
                "truth": row["truth"],
                "source": row["source"],
                "source_id": row["source_id"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            answer_key.append(
                {
                    "id": index,
                    "truth": row["truth"],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "kind": row.get("kind"),
                }
            )

    answer_key_path.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {args.output} and {answer_key_path}")


if __name__ == "__main__":
    main()
