#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = Path("/Users/unoa/Downloads/crawled-data-v2")
DEFAULT_BASE_CPT = PROJECT_ROOT / "cpt_corpus.v3.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "cpt_context_stream.jsonl"
DEFAULT_STATS = PROJECT_ROOT / "runs" / "context-cpt-stats.json"

COMMENT_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
COMMENT_REF_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")
PHONE_RE = re.compile(r"\b(?:\+?82[- ]?)?(?:0\d{1,2})[- .]?\d{3,4}[- .]?\d{4}\b")
URL_RE = re.compile(r"\bhttps?://\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PROMO_KW_RE = re.compile(
    r"(문의|카톡|텔레|라인|지원금|TC|실장|부장|출근문의|픽업|풀상주|면접|당일지급|지명비|이벤트|광고)",
    re.IGNORECASE,
)

LENGTH_BUCKETS = [
    ("xs", 0, 19),
    ("sm", 20, 49),
    ("md", 50, 99),
    ("lg", 100, 199),
    ("xl", 200, 499),
    ("xxl", 500, 10**9),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mixed CPT stream with thread-context comment rows."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--base-cpt", type=Path, default=DEFAULT_BASE_CPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--post-excerpt-chars", type=int, default=220)
    parser.add_argument(
        "--target-context-ratio",
        type=float,
        default=0.25,
        help="Fraction of the final CPT stream to serialize as context_comment rows.",
    )
    parser.add_argument(
        "--sample-unmatched",
        type=int,
        default=10,
        help="Number of unmatched CPT examples to retain in stats.",
    )
    args = parser.parse_args()
    if not (0.20 <= args.target_context_ratio <= 0.30):
        raise SystemExit("--target-context-ratio must stay within the 0.20-0.30 task range")
    return args


def load_json(path: Path):
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def normalize_text(text: str) -> str:
    return (text or "").replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n").strip()


def scrub_light_pii(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = PHONE_RE.sub("[전화번호]", text)
    text = EMAIL_RE.sub("[이메일]", text)
    text = URL_RE.sub("[URL]", text)
    return normalize_text(text)


def clean_comment(text: str) -> tuple[str | None, str]:
    text = normalize_text(text)
    key = None
    match = COMMENT_TAG_RE.match(text)
    if match:
        key = match.group(1)
        text = text[match.end() :].strip()
    while True:
        match = COMMENT_REF_RE.match(text)
        if not match:
            break
        text = text[match.end() :].strip()
    return key, text


def parent_key(key: str | None) -> str | None:
    if not key or "-" not in key:
        return None
    return key.rsplit("-", 1)[0]


def length_bucket(length: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length <= hi:
            return name
    return "unk"


def truncate_excerpt(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def is_promo_like(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text or "")
    return bool(
        PHONE_RE.search(normalized)
        or URL_RE.search(normalized)
        or (PROMO_KW_RE.search(normalized) and len(normalized.strip()) > 60)
    )


def context_text(
    title: str,
    post_body: str,
    post_excerpt_chars: int,
    comment_text: str,
    parent_text: str | None,
) -> str:
    lines = [
        f"제목: {title}" if title else "제목:",
        f"원글: {truncate_excerpt(post_body, post_excerpt_chars)}",
    ]
    if parent_text:
        lines.append(f"부모댓글: {parent_text}")
        lines.append(f"답글: {comment_text}")
    else:
        lines.append(f"댓글: {comment_text}")
    return "\n".join(lines)


def stable_rank(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def iter_source_posts(raw_dir: Path):
    for path in sorted(raw_dir.glob("*.json")):
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for post in data:
            if isinstance(post, dict):
                yield path.name, post


def load_base_rows(base_cpt: Path):
    rows: list[dict] = []
    queue_by_thread: dict[str, dict[str, deque[int]]] = defaultdict(lambda: defaultdict(deque))
    kind_counts: Counter[str] = Counter()
    patch_rows = 0
    plain_comment_rows = 0

    with base_cpt.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            rows.append(row)
            kind = str(row.get("kind") or "unknown")
            kind_counts[kind] += 1
            if kind != "comment":
                continue
            source_id = str(row.get("source_id") or "")
            if ":" in source_id:
                patch_rows += 1
                continue
            plain_comment_rows += 1
            queue_by_thread[source_id][normalize_text(str(row.get("text") or ""))].append(row_index)

    return rows, queue_by_thread, kind_counts, plain_comment_rows, patch_rows


def build_candidates(
    raw_dir: Path,
    queue_by_thread: dict[str, dict[str, deque[int]]],
    post_excerpt_chars: int,
) -> tuple[list[dict], Counter[str], int]:
    candidates: list[dict] = []
    matched_counts: Counter[str] = Counter()
    threads_scanned = 0

    for source_file, post in iter_source_posts(raw_dir):
        post_id = str(post.get("id") or "")
        if not post_id or post_id not in queue_by_thread:
            continue
        threads_scanned += 1

        thread_queues = queue_by_thread[post_id]
        title = scrub_light_pii(str(post.get("title") or ""))
        post_body = scrub_light_pii(str(post.get("content") or ""))

        parsed_comments: list[dict] = []
        comment_map: dict[str, str] = {}
        for comment_index, comment in enumerate(post.get("comments") or []):
            if not isinstance(comment, dict):
                continue
            raw_text = normalize_text(str(comment.get("content") or ""))
            if not raw_text:
                continue
            key, cleaned = clean_comment(raw_text)
            cleaned_scrubbed = scrub_light_pii(cleaned)
            parsed = {
                "comment_index": comment_index,
                "raw_text": raw_text,
                "raw_scrubbed": scrub_light_pii(raw_text),
                "comment_key": key,
                "comment_text": cleaned_scrubbed or scrub_light_pii(raw_text),
                "parent_comment_key": parent_key(key),
            }
            parsed_comments.append(parsed)
            if key is not None and parsed["comment_text"]:
                comment_map[key] = parsed["comment_text"]

        for parsed in parsed_comments:
            if is_promo_like(parsed["raw_text"]):
                matched_counts["promo_like_candidates_skipped"] += 1
                continue

            match_variants = []
            for variant in (
                parsed["raw_scrubbed"],
                parsed["raw_text"],
                parsed["comment_text"],
                normalize_text(clean_comment(parsed["raw_text"])[1]),
            ):
                variant = normalize_text(str(variant or ""))
                if variant and variant not in match_variants:
                    match_variants.append(variant)

            row_index = None
            matched_variant = None
            for variant in match_variants:
                queue = thread_queues.get(variant)
                if queue:
                    row_index = queue.popleft()
                    matched_variant = variant
                    break
            if row_index is None:
                continue

            parent_text = None
            if parsed["parent_comment_key"]:
                parent_text = comment_map.get(parsed["parent_comment_key"])

            serialized = context_text(
                title=title,
                post_body=post_body,
                post_excerpt_chars=post_excerpt_chars,
                comment_text=parsed["comment_text"],
                parent_text=parent_text,
            )
            context_mode = "reply" if parent_text else "root"
            matched_counts[context_mode] += 1
            candidates.append(
                {
                    "row_index": row_index,
                    "source_id": post_id,
                    "source_file": source_file,
                    "comment_index": parsed["comment_index"],
                    "comment_key": parsed["comment_key"],
                    "parent_comment_key": parsed["parent_comment_key"],
                    "context_mode": context_mode,
                    "context_text": serialized,
                    "matched_variant": matched_variant,
                }
            )

    return candidates, matched_counts, threads_scanned


def collect_unmatched_examples(
    rows: list[dict],
    queue_by_thread: dict[str, dict[str, deque[int]]],
    sample_limit: int,
) -> list[dict]:
    samples: list[dict] = []
    for source_id, text_queues in queue_by_thread.items():
        for text, queue in text_queues.items():
            while queue and len(samples) < sample_limit:
                row_index = queue.popleft()
                row = rows[row_index]
                samples.append(
                    {
                        "row_index": row_index,
                        "source_id": source_id,
                        "text": row.get("text", ""),
                    }
                )
            if len(samples) >= sample_limit:
                return samples
    return samples


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)

    rows, queue_by_thread, kind_counts, plain_comment_rows, patch_rows = load_base_rows(args.base_cpt)
    candidates, matched_counts, threads_scanned = build_candidates(
        raw_dir=args.raw_dir,
        queue_by_thread=queue_by_thread,
        post_excerpt_chars=args.post_excerpt_chars,
    )

    total_rows = len(rows)
    target_context_rows = min(len(candidates), round(total_rows * args.target_context_ratio))
    ranked_candidates = sorted(
        candidates,
        key=lambda item: stable_rank(
            f"{item['source_id']}|{item['comment_key'] or ''}|{item['row_index']}|{item['context_text']}"
        ),
    )
    selected_rows = ranked_candidates[:target_context_rows]
    selected_by_index = {item["row_index"]: item for item in selected_rows}

    selected_counts: Counter[str] = Counter(item["context_mode"] for item in selected_rows)
    selected_threads = len({item["source_id"] for item in selected_rows})

    with args.out.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            new_row = dict(row)
            selected = selected_by_index.get(row_index)
            if selected is not None:
                new_row["text"] = selected["context_text"]
                new_row["kind"] = "context_comment"
                new_row["context_mode"] = selected["context_mode"]
                new_row["comment_key"] = selected["comment_key"]
                new_row["parent_comment_key"] = selected["parent_comment_key"]
                new_row["length_bucket"] = length_bucket(len(selected["context_text"]))
            handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    remaining_unmatched = sum(
        len(queue)
        for text_queues in queue_by_thread.values()
        for queue in text_queues.values()
    )
    unmatched_examples = collect_unmatched_examples(rows, queue_by_thread, args.sample_unmatched)

    stats = {
        "raw_dir": str(args.raw_dir),
        "base_cpt": str(args.base_cpt),
        "output_path": str(args.out),
        "stats_path": str(args.stats_out),
        "post_excerpt_chars": args.post_excerpt_chars,
        "target_context_ratio": args.target_context_ratio,
        "total_rows": total_rows,
        "input_kind_counts": dict(kind_counts),
        "plain_comment_rows": plain_comment_rows,
        "patch_rows_left_standalone": patch_rows,
        "threads_scanned": threads_scanned,
        "matched_candidate_rows": len(candidates),
        "matched_candidate_counts": dict(matched_counts),
        "selected_context_rows": len(selected_rows),
        "selected_context_counts": dict(selected_counts),
        "selected_threads": selected_threads,
        "standalone_rows": total_rows - len(selected_rows),
        "achieved_context_ratio": round(len(selected_rows) / max(1, total_rows), 6),
        "remaining_unmatched_candidate_rows": remaining_unmatched,
        "unmatched_examples": unmatched_examples,
    }
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
