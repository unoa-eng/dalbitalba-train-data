#!/usr/bin/env python3
"""Build a metadata-only style map from the Obsidian sample vault.

This script is the explicit curation step required by the repo policy.
It extracts only frontmatter metadata keyed by `source_id`.
It never copies markdown body/title/comment text into the output map.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

HONORIFIC_LABELS = {
    "banmal": "반말",
    "jondaetmal": "존댓말",
    "mixed": "혼용",
    "haeyo": "해요체",
    "haera": "해라체",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vault-root",
        type=Path,
        default=Path("research/obsidian-export"),
        help="Path to the Obsidian export root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-slang-tokens",
        type=int,
        default=4,
        help="Maximum slang tokens retained in the compact profile",
    )
    return parser.parse_args()


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    return value.strip("'\"")


def parse_frontmatter(text: str) -> dict[str, Any]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    payload: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        payload[key.strip()] = parse_scalar(raw)
    return payload


def should_index(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if path.name in {"README.md", "_INDEX.md"}:
        return False
    if any(part.startswith("_") for part in rel.parts):
        return False
    return True


def honorific_label(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return HONORIFIC_LABELS.get(value, value or "미상")


def build_profile(record: dict[str, Any], max_slang_tokens: int) -> str:
    fields = [
        f"주제={record.get('topic_label') or record.get('topic_cluster') or '미상'}",
        f"길이={record.get('length_bucket') or '미상'}",
        f"말투={honorific_label(record.get('honorific'))}",
        f"질문={'예' if record.get('has_question') else '아니오'}",
        f"바이럴={'예' if record.get('is_viral') else '아니오'}",
    ]
    slang_tokens = list(record.get("slang_tokens") or [])[:max_slang_tokens]
    if slang_tokens:
        fields.append(f"슬랭={','.join(slang_tokens)}")
    conventions = list(record.get("meta_conventions") or [])[:3]
    if conventions:
        fields.append(f"표현습관={','.join(conventions)}")
    return " | ".join(fields)


def main() -> int:
    args = parse_args()
    if not args.vault_root.is_dir():
        raise SystemExit(f"vault root not found: {args.vault_root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, Any]] = {}
    topics: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    honorifics: Counter[str] = Counter()

    for path in sorted(args.vault_root.rglob("*.md")):
        if not should_index(path, args.vault_root):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        frontmatter = parse_frontmatter(text)
        source_id = str(frontmatter.get("source_id") or "").strip()
        if not source_id:
            continue

        record = {
            "source_id": source_id,
            "topic_cluster": str(frontmatter.get("topic_cluster") or ""),
            "topic_label": str(frontmatter.get("topic_label") or ""),
            "length_bucket": str(frontmatter.get("length_bucket") or ""),
            "length_chars": int(frontmatter.get("length_chars") or 0),
            "is_viral": bool(frontmatter.get("is_viral")),
            "views": int(frontmatter.get("views") or 0),
            "likes": int(frontmatter.get("likes") or 0),
            "comment_count": int(frontmatter.get("comment_count") or 0),
            "date": str(frontmatter.get("date") or ""),
            "honorific": str(frontmatter.get("honorific") or ""),
            "has_question": bool(frontmatter.get("has_question")),
            "meta_conventions": list(frontmatter.get("meta_conventions") or []),
            "slang_tokens": list(frontmatter.get("slang_tokens") or []),
            "tags": list(frontmatter.get("tags") or []),
            "obsidian_path": path.relative_to(args.vault_root).as_posix(),
        }
        record["profile"] = build_profile(record, args.max_slang_tokens)
        records[source_id] = record

        topics[record["topic_label"] or record["topic_cluster"] or "미상"] += 1
        lengths[record["length_bucket"] or "미상"] += 1
        honorifics[honorific_label(record["honorific"])] += 1

    payload = {
        "vault_root": str(args.vault_root),
        "records": records,
        "summary": {
            "record_count": len(records),
            "topics": dict(sorted(topics.items(), key=lambda item: (-item[1], item[0]))),
            "length_buckets": dict(sorted(lengths.items(), key=lambda item: (-item[1], item[0]))),
            "honorifics": dict(sorted(honorifics.items(), key=lambda item: (-item[1], item[0]))),
            "allowed_fields": [
                "source_id",
                "topic_cluster",
                "topic_label",
                "length_bucket",
                "length_chars",
                "is_viral",
                "views",
                "likes",
                "comment_count",
                "date",
                "honorific",
                "has_question",
                "meta_conventions",
                "slang_tokens",
                "tags",
                "obsidian_path",
                "profile",
            ],
            "forbidden_fields": [
                "markdown_body",
                "title_text",
                "comment_text",
                "analysis_note_body",
            ],
        },
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
