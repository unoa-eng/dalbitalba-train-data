#!/usr/bin/env python3
"""Integrity audit for Phase 2 final threads_v4.jsonl."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
SEED = REPO / "runs/cycle10-phase1-claude-direct/threads_v3.jsonl"
POST_PARTS = BASE / "post_parts"
BULK_PARTS = BASE / "bulk_parts"

INDEX_RE = re.compile(r"^\[(\d+)\]\s+")
COMPAT_JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
AD_RE = re.compile(
    r"\[전화번호\]|\[URL\]|https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+|"
    r"\b010[.\-\s]?\d|카톡\s*(?:한번|아이디|주세요|주세|문의)|"
    r"카카오\s*[A-Za-z0-9]|카카오톡|오픈\s*채팅|오픈채팅|오픈톡|텔레그램|"
    r"문의\s*(?:주세요|주세|환영|가능)|연락\s*(?:주세요|주세|바랍니다|가능|부탁)|"
    r"상담\s*(?:문의|환영|가능)|24\s*시\s*영업|개인\s*이벤트|담당\s*고르|"
    r"돈\s*벌러\s*오|오세요~|출근시\s*첫|첫\s*티씨|첫티씨|"
    r"풀티\s*지급|TC\s*\+|풀상주|칼수금|스타트톡|개수톡|출퇴근\s*차량|"
    r"여성용품|매점|비품|밀빵|잘챙겨드릴게요|모십니다|지원해드립니다|"
    r"도와드리겠습니다|와주시면\s*감사",
    re.I,
)
TEXT_KEYS = ("title", "content")


def iter_jsonl(path: Path):
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                yield i, json.loads(line)


def strip_prefix(text: str) -> str:
    return INDEX_RE.sub("", text or "").strip()


def compact_opening(text: str, n: int = 14) -> str:
    x = strip_prefix(text)
    x = re.sub(r"\s+", "", x)
    return x[:n]


def main() -> int:
    path = Path(sys.argv[1])
    threads = [row for _, row in iter_jsonl(path)]
    seed = [row for _, row in iter_jsonl(SEED)]
    seed_pairs = {
        (t["id"], i)
        for t in seed
        for i, _ in enumerate(t.get("comments") or [])
    }
    post_ids = {
        str(row["id"])
        for part in sorted(POST_PARTS.glob("part*.jsonl"))
        for _, row in iter_jsonl(part)
    }
    bulk_pairs = {
        (str(row["root_id"]), int(row["comment_index"]))
        for part in sorted(BULK_PARTS.glob("part*.jsonl"))
        for _, row in iter_jsonl(part)
    }

    errors = []
    warnings = []
    if len(threads) != 3204:
        errors.append(f"thread count {len(threads)} != 3204")

    final_pairs = set()
    titles, bodies, comments = [], [], []
    opening = Counter()
    crawl_by_id = {}

    for line_no, t in iter_jsonl(path):
        tid = str(t.get("id"))
        for key in ["id", "title", "author", "date", "likes", "views", "commentCount", "content", "comments", "boardName", "crawledAt"]:
            if key not in t:
                errors.append(f"line {line_no} missing key {key}")
        if t.get("author") != "비회원":
            errors.append(f"{tid} author {t.get('author')!r}")
        if not isinstance(t.get("views"), str) or not isinstance(t.get("likes"), str) or not isinstance(t.get("commentCount"), str):
            errors.append(f"{tid} views/likes/commentCount not strings")
        if not str(t.get("views", "")).isdigit() or not str(t.get("likes", "")).isdigit() or not str(t.get("commentCount", "")).isdigit():
            errors.append(f"{tid} numeric string field invalid")
        for key in TEXT_KEYS:
            if COMPAT_JAMO_RE.search(t.get(key, "")):
                errors.append(f"{tid} compatibility jamo in post {key}")
            if AD_RE.search(t.get(key, "")):
                errors.append(f"{tid} ad/contact pattern in post {key}")

        cs = t.get("comments") or []
        if str(len(cs)) != t.get("commentCount"):
            errors.append(f"{tid} commentCount {t.get('commentCount')} != {len(cs)}")
        titles.append(t.get("title", ""))
        bodies.append(t.get("content", ""))
        try:
            crawl_dt = datetime.strptime(t["crawledAt"], "%Y-%m-%dT%H:%M:%S")
            crawl_by_id[tid] = crawl_dt
        except Exception:
            errors.append(f"{tid} bad crawledAt {t.get('crawledAt')!r}")
            crawl_dt = None
        prev = None
        for i, c in enumerate(cs):
            final_pairs.add((tid, i))
            if c.get("author") != "비회원":
                errors.append(f"{tid}:{i} comment author {c.get('author')!r}")
            m = INDEX_RE.match(c.get("content", ""))
            if not m or int(m.group(1)) != i + 1:
                errors.append(f"{tid}:{i} bad comment prefix {c.get('content')!r}")
            if COMPAT_JAMO_RE.search(c.get("content", "")):
                errors.append(f"{tid}:{i} compatibility jamo in comment")
            if AD_RE.search(c.get("content", "")):
                errors.append(f"{tid}:{i} ad/contact pattern in comment")
            try:
                dt = datetime.strptime(c["date"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                errors.append(f"{tid}:{i} bad comment date {c.get('date')!r}")
                continue
            if prev and dt <= prev:
                errors.append(f"{tid}:{i} comment date not increasing")
            if crawl_dt and dt >= crawl_dt:
                errors.append(f"{tid}:{i} comment date after crawl")
            prev = dt
            comments.append(strip_prefix(c.get("content", "")))
            op = compact_opening(c.get("content", ""))
            if len(op) >= 8:
                opening[op] += 1

    missing_seed = seed_pairs - final_pairs
    extra_seed = final_pairs - seed_pairs
    if missing_seed:
        errors.append(f"missing seed comment pairs {len(missing_seed)}")
    if extra_seed:
        errors.append(f"extra comment pairs {len(extra_seed)}")
    missing_bulk = bulk_pairs - final_pairs
    if missing_bulk:
        errors.append(f"missing bulk pairs {len(missing_bulk)}")
    final_post_ids = {t["id"] for t in threads}
    if post_ids - final_post_ids:
        errors.append(f"post part ids absent from final {len(post_ids - final_post_ids)}")

    uniq = {
        "title_unique_pct": round(len(set(titles)) / len(titles) * 100, 2),
        "body_unique_pct": round(len(set(bodies)) / len(bodies) * 100, 2),
        "comment_unique_pct": round(len(set(comments)) / len(comments) * 100, 2),
    }
    for key, value in uniq.items():
        if value <= 95.0:
            errors.append(f"{key} {value} <= 95")

    repeated_openings = [(k, v) for k, v in opening.most_common(20) if v >= 8]
    if repeated_openings:
        warnings.append({
            "repeated_openings_ge8": repeated_openings[:20],
            "note": "manual review recommended; not a hard schema failure",
        })

    report = {
        "path": str(path),
        "threads": len(threads),
        "comments": len(comments),
        "post_rework_ids": len(post_ids),
        "bulk_pairs": len(bulk_pairs),
        **uniq,
        "warnings": warnings,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
