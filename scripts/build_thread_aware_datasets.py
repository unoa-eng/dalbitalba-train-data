#!/usr/bin/env python3
"""
build_thread_aware_datasets.py — P1 thread-aware SFT v3 builder.

Reads cpt_corpus.v2.jsonl + sft_pairs.v2.jsonl + val_set.v2.jsonl and
produces the v3 schema:

  {
    "post_title":        str,            # always "" (no title field in v2)
    "post_body_excerpt": str,            # <= 256 chars of post text
    "parent_comment":    str | null,     # root comment text for replies, null for root
    "target_comment":    str,            # the comment being trained on
    "thread_key":        str,
    "depth":             int,            # 0=post_continuation 1=root_comment 2+=reply
    "task_type":         "post_continuation" | "root_comment" | "reply",
    "length_bucket":     "xs|sm|md|lg|xl|xxl",
    "source_id":         str,
  }

Filter rules (applied in order):
  1. DROP placeholder comments: [N] 삭제된 댓글입니다. / [N-M] 신고에 의해 블라인드 ...
  2. DROP comments matching AD_RE from scripts/clean_ad_spam.py
  3. DROP empty / whitespace-only target_comment
  4. DO NOT drop short slang (len < 20 is valid)
  5. PRESERVE root comments (parent_comment=null)

Val split: rows whose source_id appears in val_set.v2.jsonl go to val_set.v3.jsonl.

Usage:
    python3 scripts/build_thread_aware_datasets.py \\
        --in-cpt  cpt_corpus.v2.jsonl \\
        --in-sft  sft_pairs.v2.jsonl \\
        --in-val  val_set.v2.jsonl \\
        --out     sft_pairs.v3.jsonl \\
        --val-out val_set.v3.jsonl \\
        --summary .planning/calibration/sft_v3_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# AD patterns — imported directly to keep stdlib-only (no import of sibling
# module which may not be on sys.path when run from repo root).
# These are identical to the AD_PATTERNS list in scripts/clean_ad_spam.py.
# ---------------------------------------------------------------------------
AD_PATTERNS = [
    r'카카오\s*(?:톡\s*)?(?:아이디|ID)?\s*[:\s]\s*[A-Za-z0-9_\-]{3,}',
    r'카톡\s*(?:아이디|ID)?\s*[:\s]\s*[A-Za-z0-9_\-]{3,}',
    r'카카오\s+[a-zA-Z][a-zA-Z0-9]{2,}',
    r'카톡\s+[a-zA-Z][a-zA-Z0-9]{2,}',
    r'[Kk]akao\s*(?:ID)?\s*[:：]\s*[A-Za-z0-9_\-]{3,}',
    r'\[전화번호\].*카[카톡]',
    r'카[카톡].*\[전화번호\]',
    r'밀빵\s*확실',
    r'밀빵\s*가능',
    r'하루\s*평균\s*\d+\s*방',
    r'문의\s*(주|하)\s*(세요|십시오|시면)',
    r'풀상주\s*풀케어',
    r'출근\s*문의',
    r'스타트톡\s*개수톡',
    r'[1-3]부\s*\d+인\s*\d+조',
    r'G팀\s*\d+방',
    r'\d+방\s*팀사장',
    r'(?:저희|우리)\s*(?:가게|업소|샵|샾|매장)',
    r'출근시?\s*연락(?:주|드|바람)',
    r'언제든\s*편하게\s*(?:연락|문의)',
    r'편하게\s*톡\s*주세요',
    r'편하게\s*전화\s*주세요',
    r'VIP\s*(?:전용|관리|코스)',
    r'(?:풀|올)\s*케어',
    r'개수\s*보장',
    r'(?:라인|line)\s*(?:ID|아이디)?\s*[:：]\s*[a-zA-Z0-9_]{3,}',
    r'(?:텔레|텔그|텔레그램)\s*(?:ID|아이디)?\s*[:：@]?\s*[a-zA-Z0-9_]{3,}',
    r'(?:오픈\s*)?오픈톡|오픈\s*카톡',
    r'시간\s*(?:당|시)?\s*\d{2,4}\s*만원',
    r'일\s*(?:당|급)\s*\d{2,4}\s*만원',
    r'(?:선|즉)\s*입금',
    r'\d{1,2}\s*시간\s*기본',
    r'\d{1,2}\s*[~∼–-]\s*\d{1,2}\s*시\s*출근',
]
AD_RE = re.compile('|'.join(AD_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Placeholder patterns (from dedup_minhash_sft_v2.json top clusters)
# Matches the full comment text including the [N] / [N-M] prefix.
# ---------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(
    r'^\s*(?:\S+\s+)?\[\d+(?:-\d+)?\]\s+'
    r'(?:삭제된 댓글입니다\.'
    r'|신고에 의해 블라인드 처리 되었습니다\.+)\s*$'
)

# ---------------------------------------------------------------------------
# Comment prefix: [N] root comment, [N-M] reply, optional "비회원 " / "작성자 "
# prefix before the bracket.
# ---------------------------------------------------------------------------
PREFIX_RE = re.compile(r'^(?:[^\[\n]{0,30})?\[(\d+)(?:-(\d+))?\]\s*')

# ---------------------------------------------------------------------------
# Length bucket thresholds (from phase1_data_pipeline.py)
# ---------------------------------------------------------------------------
BUCKET_BREAKS = [
    ("xs",  0,    19),
    ("sm",  20,   49),
    ("md",  50,   99),
    ("lg",  100,  199),
    ("xl",  200,  499),
    ("xxl", 500,  10**9),
]


def length_bucket(n: int) -> str:
    for name, lo, hi in BUCKET_BREAKS:
        if lo <= n <= hi:
            return name
    return "xxl"


def post_body_excerpt(text: str, limit: int = 256) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def parse_comment_prefix(comment_text: str) -> tuple[str | None, str | None, int]:
    """Return (root_index, sub_index, depth) from [N] or [N-M] prefix.

    root_index: the N (string)
    sub_index:  the M (string) or None for root comments
    depth:      1 for root [N], 2 for immediate reply [N-1], etc.
    Returns (None, None, -1) if no prefix found.
    """
    m = PREFIX_RE.match(comment_text)
    if not m:
        return None, None, -1
    root = m.group(1)
    sub = m.group(2)
    if sub is None:
        return root, None, 1
    return root, sub, 1 + int(sub)


def strip_prefix(comment_text: str) -> str:
    """Remove the [N] / [N-M] prefix (and optional author tag) from comment."""
    m = PREFIX_RE.match(comment_text)
    if m:
        return comment_text[m.end():].strip()
    return comment_text.strip()


def is_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_RE.match(text))


def is_ad(text: str) -> bool:
    return bool(AD_RE.search(text))


def is_empty(text: str) -> bool:
    return not text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 thread-aware SFT v3 builder")
    p.add_argument("--in-cpt",  required=True, help="cpt_corpus.v2.jsonl")
    p.add_argument("--in-sft",  required=True, help="sft_pairs.v2.jsonl")
    p.add_argument("--in-val",  required=True, help="val_set.v2.jsonl")
    p.add_argument("--out",     required=True, help="sft_pairs.v3.jsonl output")
    p.add_argument("--val-out", required=True, help="val_set.v3.jsonl output")
    p.add_argument("--summary", required=True, help="sft_v3_summary.json output")
    return p.parse_args()


def load_val_source_ids(val_path: str) -> set[str]:
    """Collect the unique source_ids that appear in the val set.

    Note: val_set.v2.jsonl is a CPT-format held-out set whose source_ids do
    not overlap with the SFT thread source_ids.  We keep this function for
    forward compatibility but the primary val split is performed by
    is_val_thread_key() below.
    """
    ids: set[str] = set()
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("source_id")
            if sid:
                ids.add(str(sid))
    return ids


def is_val_thread_key(thread_key: str, val_fraction: float = 0.05) -> bool:
    """Deterministic 5% val split based on MD5 hash of the thread_key.

    Using a hash keeps the split stable across re-runs without needing a
    pre-built lookup set.  val_fraction=0.05 → ~5% of threads go to val.
    """
    h = int(hashlib.md5(thread_key.encode("utf-8")).hexdigest(), 16)
    return (h % 10000) < int(val_fraction * 10000)


def load_post_texts(cpt_path: str) -> dict[str, str]:
    """Return {source_id: post_text} for kind=='post' rows in cpt_corpus."""
    posts: dict[str, str] = {}
    with open(cpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "post":
                sid = str(r.get("source_id", ""))
                if sid and sid not in posts:
                    posts[sid] = r.get("text", "")
    return posts


def build_thread_index(sft_path: str) -> dict[str, list[dict]]:
    """Group all sft_pairs.v2 rows by thread_key, preserving order."""
    threads: dict[str, list[dict]] = defaultdict(list)
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tk = str(r.get("thread_key", r.get("source_id", "")))
            threads[tk].append(r)
    return dict(threads)


def process_threads(
    threads: dict[str, list[dict]],
    post_texts: dict[str, str],
    val_ids: set[str],
    out_path: str,
    val_out_path: str,
    summary_path: str,
) -> None:
    # Drop counters
    dropped_placeholder = 0
    dropped_ad = 0
    dropped_empty = 0
    input_rows = 0

    task_type_counts: Counter = Counter()
    depth_hist: Counter = Counter()
    bucket_dist: Counter = Counter()
    kept_rows = 0

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path(val_out_path).parent
    val_dir.mkdir(parents=True, exist_ok=True)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f, \
         open(val_out_path, "w", encoding="utf-8") as val_f:

        for thread_key, rows in threads.items():
            if not rows:
                continue

            # Get post body from first row's 'post' field (already clean)
            # Fallback to cpt_corpus lookup
            post_text = rows[0].get("post", "")
            source_id = str(rows[0].get("source_id", thread_key))

            if not post_text:
                post_text = post_texts.get(source_id, post_texts.get(thread_key, ""))

            excerpt = post_body_excerpt(post_text)
            is_val_thread = is_val_thread_key(thread_key) or source_id in val_ids

            # Build root-comment lookup for this thread:
            # {root_index_str: cleaned_root_comment_text}
            root_map: dict[str, str] = {}

            # First pass: collect root comments (depth=1, [N] prefix)
            for row in rows:
                comment_raw = row.get("comment", "")
                input_rows += 1
                root_idx, sub_idx, depth = parse_comment_prefix(comment_raw)
                if depth == 1 and root_idx is not None and sub_idx is None:
                    cleaned = strip_prefix(comment_raw)
                    if cleaned and not is_placeholder(comment_raw) and not is_ad(cleaned):
                        root_map[root_idx] = cleaned

            # Second pass: emit v3 rows
            for row in rows:
                comment_raw = row.get("comment", "")
                root_idx, sub_idx, depth = parse_comment_prefix(comment_raw)
                cleaned = strip_prefix(comment_raw)

                # Filter 1: placeholder (check on raw text which still has prefix)
                if is_placeholder(comment_raw):
                    dropped_placeholder += 1
                    continue

                # Filter 2: AD pattern (check on cleaned text)
                if is_ad(cleaned):
                    dropped_ad += 1
                    continue

                # Filter 3: empty after stripping
                if is_empty(cleaned):
                    dropped_empty += 1
                    continue

                # Determine task_type and parent_comment
                if depth == 1 and sub_idx is None:
                    task_type = "root_comment"
                    parent_comment = None
                    actual_depth = 1
                elif depth >= 2 and root_idx is not None:
                    task_type = "reply"
                    parent_comment = root_map.get(root_idx)
                    # If parent was dropped (ad/placeholder), use None but still emit
                    actual_depth = depth
                else:
                    # No recognized prefix: treat as root_comment
                    task_type = "root_comment"
                    parent_comment = None
                    actual_depth = 1

                bucket = length_bucket(len(cleaned))

                v3_row = {
                    "post_title": "",
                    "post_body_excerpt": excerpt,
                    "parent_comment": parent_comment,
                    "target_comment": cleaned,
                    "thread_key": thread_key,
                    "depth": actual_depth,
                    "task_type": task_type,
                    "length_bucket": bucket,
                    "source_id": source_id,
                }

                line_out = json.dumps(v3_row, ensure_ascii=False) + "\n"
                if is_val_thread:
                    val_f.write(line_out)
                else:
                    out_f.write(line_out)

                kept_rows += 1
                task_type_counts[task_type] += 1
                depth_hist[actual_depth] += 1
                bucket_dist[bucket] += 1

    # Also add post_continuation rows from cpt_corpus posts
    # (task_type="post_continuation", depth=0)
    # These use the post text itself as target_comment given an empty/title prompt.
    # We emit these only for training (non-val) threads.
    post_continuation_count = 0
    with open(out_path, "a", encoding="utf-8") as out_f, \
         open(val_out_path, "a", encoding="utf-8") as val_f:
        for source_id, post_text in post_texts.items():
            if not post_text.strip():
                continue
            # Filter ads on post text
            if is_ad(post_text):
                continue

            excerpt = post_body_excerpt(post_text)
            bucket = length_bucket(len(post_text.strip()))

            v3_row = {
                "post_title": "",
                "post_body_excerpt": excerpt,
                "parent_comment": None,
                "target_comment": post_text.strip(),
                "thread_key": source_id,
                "depth": 0,
                "task_type": "post_continuation",
                "length_bucket": bucket,
                "source_id": source_id,
            }

            line_out = json.dumps(v3_row, ensure_ascii=False) + "\n"
            if is_val_thread_key(source_id) or source_id in val_ids:
                val_f.write(line_out)
            else:
                out_f.write(line_out)

            post_continuation_count += 1
            kept_rows += 1
            task_type_counts["post_continuation"] += 1
            depth_hist[0] += 1
            bucket_dist[bucket] += 1

    total_for_fractions = sum(bucket_dist.values()) or 1
    bucket_fractions = {k: round(v / total_for_fractions, 4) for k, v in sorted(bucket_dist.items())}

    summary = {
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "dropped": {
            "placeholder": dropped_placeholder,
            "ad_pattern": dropped_ad,
            "empty": dropped_empty,
        },
        "task_type_counts": dict(task_type_counts),
        "depth_histogram": {str(k): v for k, v in sorted(depth_hist.items())},
        "length_bucket_distribution": bucket_fractions,
    }

    Path(summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()

    print("Loading val source_ids ...", file=sys.stderr)
    val_ids = load_val_source_ids(args.in_val)
    print(f"  val threads: {len(val_ids)}", file=sys.stderr)

    print("Loading post texts from cpt_corpus ...", file=sys.stderr)
    post_texts = load_post_texts(args.in_cpt)
    print(f"  posts loaded: {len(post_texts)}", file=sys.stderr)

    print("Building thread index from sft_pairs ...", file=sys.stderr)
    threads = build_thread_index(args.in_sft)
    print(f"  unique threads: {len(threads)}", file=sys.stderr)

    print("Processing threads and emitting v3 rows ...", file=sys.stderr)
    process_threads(
        threads=threads,
        post_texts=post_texts,
        val_ids=val_ids,
        out_path=args.out,
        val_out_path=args.val_out,
        summary_path=args.summary,
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
