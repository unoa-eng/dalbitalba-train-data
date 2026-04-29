#!/usr/bin/env python3
"""
build_structured_cpt_v3.py — 구조 토큰 삽입 CPT 코퍼스 생성

TRAINING_DESIGN_V3.md Stage 0 + Stage 1 구현:
- <|post|>/<|/post|> 마커로 게시글 경계 표시
- <|comment depth=N|>/<|/comment|> 마커로 댓글 구조 표시
- Thread 연결 시퀀스 (게시글 + 댓글 체인) 생성
- is_promo_v2() 정밀 필터

Usage:
    python scripts/build_structured_cpt_v3.py --raw-dir /path/to/raw --out-dir ./v3-data
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ── Special tokens ──────────────────────────────────────────────────────
POST_START = "<|post|>"
POST_END = "<|/post|>"
COMMENT_START_FMT = "<|comment depth={depth}|>"
COMMENT_END = "<|/comment|>"
THREAD_START = "<|thread|>"
THREAD_END = "<|/thread|>"

# ── Regex patterns ──────────────────────────────────────────────────────
PHONE_RE = re.compile(r"(?:\+?82[-.\s]?)?01\d[-.\s]?\d{3,4}[-.\s]?\d{4}")
URL_RE = re.compile(r"(https?://|open\.kakao|t\.me/|www\.)", re.IGNORECASE)
CONTACT_ID_RE = re.compile(
    r"카카오\s*(?:톡\s*)?(?:아이디|ID)?\s*[:：.\-\s]?\s*[A-Za-z0-9_.\-]{3,}|"
    r"카톡\s*(?:아이디|ID)?\s*[:：.\-\s]?\s*[A-Za-z0-9_.\-]{3,}|"
    r"kakao\s*(?:talk\s*)?(?:id)?\s*[:：.\-\s]?\s*[A-Za-z0-9_.\-]{3,}|"
    r"line\s*(?:id)?\s*[:：.\-\s]?\s*[A-Za-z0-9_.\-]{3,}|"
    r"라인\s*(?:아이디|ID)?\s*[:：.\-\s]?\s*[A-Za-z0-9_.\-]{3,}",
    re.IGNORECASE,
)
# 고밀도 광고 패턴 (연락처와 함께 등장할 때만 의미 있는 키워드는 제외)
AD_DENSE_RE = re.compile(
    r"(밀빵\s*확실|밀빵\s*가능|풀상주\s*풀케어|스타트톡\s*개수톡|"
    r"하루\s*평균\s*\d+\s*방|[1-3]부\s*\d+인\s*\d+조|"
    r"G팀\s*\d+방|\d+방\s*팀사장)",
    re.IGNORECASE,
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

COMMENT_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
COMMENT_REF_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")

BLOCKED_TEXTS = {
    "",
    "삭제된 댓글입니다.",
    "광고 권한이 없습니다.",
    "신고에 의해 블라인드 처리 되었습니다..",
}

LENGTH_BUCKETS = [
    ("xs", 0, 19),
    ("sm", 20, 49),
    ("md", 50, 99),
    ("lg", 100, 199),
    ("xl", 200, 499),
    ("xxl", 500, 10**9),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="구조 토큰 삽입 CPT 코퍼스 생성 (v3)")
    parser.add_argument("--raw-dir", required=True, help="raw crawl JSON 디렉터리")
    parser.add_argument("--out-dir", required=True, help="산출물 디렉터리")
    parser.add_argument("--min-chars", type=int, default=8, help="최소 텍스트 길이")
    return parser.parse_args()


def load_json(path: Path):
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def iter_rows(raw_dir: Path):
    for path in sorted(raw_dir.glob("*.json")):
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict):
                yield row


def is_promo_v2(text: str) -> bool:
    """정밀 프로모 필터: 연락처 동반 시에만 프로모 판정."""
    text = (text or "").strip()
    if not text:
        return False

    has_contact = bool(
        PHONE_RE.search(text) or URL_RE.search(text) or CONTACT_ID_RE.search(text)
    )
    if has_contact:
        return True

    # 연락처 없으면 고밀도 광고 패턴이 다수일 때만 필터
    ad_matches = AD_DENSE_RE.findall(text)
    if len(ad_matches) >= 2:
        return True

    return False


def minor_sexual_proximity(text: str, window: int = 60) -> bool:
    minors = [m.start() for m in MINOR_RE.finditer(text or "")]
    if not minors:
        return False
    sexuals = [m.start() for m in SEXUAL_RE.finditer(text or "")]
    return any(abs(a - b) <= window for a in minors for b in sexuals)


def clean_comment(text: str) -> tuple[str | None, str]:
    text = (text or "").replace("\xa0", " ").strip()
    key = None
    match = COMMENT_TAG_RE.match(text)
    if match:
        key = match.group(1)
        text = text[match.end():].strip()
    while True:
        match = COMMENT_REF_RE.match(text)
        if not match:
            break
        text = text[match.end():].strip()
    return key, text


def parent_key(key: str | None) -> str | None:
    if not key or "-" not in key:
        return None
    return key.rsplit("-", 1)[0]


def key_depth(key: str | None) -> int:
    if not key:
        return 0
    return key.count("-")


def length_bucket(text: str) -> str:
    length = len(text or "")
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length <= hi:
            return name
    return "unk"


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpt_merged = out_dir / "cpt_structured_v3.jsonl"
    cpt_individual = out_dir / "cpt_structured_individual.jsonl"
    cpt_threads = out_dir / "cpt_structured_threads.jsonl"
    summary_path = out_dir / "cpt_v3_summary.json"

    stats = Counter()

    with (
        cpt_merged.open("w", encoding="utf-8") as merged_f,
        cpt_individual.open("w", encoding="utf-8") as ind_f,
        cpt_threads.open("w", encoding="utf-8") as thread_f,
    ):
        for row in iter_rows(raw_dir):
            stats["raw_posts"] += 1

            post_id = str(row.get("id") or "")
            title = str(row.get("title") or "").strip()
            content = str(row.get("content") or "").strip()
            merged_post = "\n".join([p for p in (title, content) if p]).strip()

            if not content or len(content) < args.min_chars:
                stats["dropped_short_post"] += 1
                continue

            if is_promo_v2(merged_post):
                stats["dropped_promo_post"] += 1
                continue
            if minor_sexual_proximity(merged_post):
                stats["dropped_minor_sexual_post"] += 1
                continue

            # === Individual post entry ===
            post_text = f"{POST_START}"
            if title:
                post_text += f"제목: {title}\n"
            post_text += f"{content}{POST_END}"
            post_row = {
                "text": post_text,
                "kind": "post",
                "source_id": post_id,
                "source_field": "post",
                "length_bucket": length_bucket(post_text),
            }

            serialized = json.dumps(post_row, ensure_ascii=False) + "\n"
            ind_f.write(serialized)
            merged_f.write(serialized)
            stats["cpt_posts"] += 1
            stats["cpt_total_rows"] += 1

            # === Process comments ===
            comment_map: dict[str, str] = {}
            ordered_comments: list[tuple[str | None, str, int]] = []

            for comment in row.get("comments") or []:
                if not isinstance(comment, dict):
                    continue
                stats["raw_comments"] += 1
                key, cleaned = clean_comment(str(comment.get("content") or ""))

                if cleaned in BLOCKED_TEXTS:
                    stats["dropped_blocked"] += 1
                    continue
                if len(cleaned) < args.min_chars:
                    stats["dropped_short_comment"] += 1
                    continue
                if is_promo_v2(cleaned):
                    stats["dropped_promo_comment"] += 1
                    continue
                if minor_sexual_proximity(cleaned):
                    stats["dropped_minor_sexual_comment"] += 1
                    continue

                depth = key_depth(key)
                ordered_comments.append((key, cleaned, depth))
                if key is not None:
                    comment_map[key] = cleaned

                # Individual comment entry
                comment_marker = COMMENT_START_FMT.format(depth=depth)
                comment_text = f"{comment_marker}{cleaned}{COMMENT_END}"
                comment_row = {
                    "text": comment_text,
                    "kind": "comment",
                    "source_id": post_id,
                    "source_field": "comment",
                    "length_bucket": length_bucket(comment_text),
                    "depth": depth,
                }
                serialized = json.dumps(comment_row, ensure_ascii=False) + "\n"
                ind_f.write(serialized)
                merged_f.write(serialized)
                stats["cpt_comments"] += 1
                stats["cpt_total_rows"] += 1

            # === Thread sequence (post + all comments in order) ===
            if ordered_comments:
                thread_parts = [f"{THREAD_START}", post_text]
                for key, cleaned, depth in ordered_comments:
                    comment_marker = COMMENT_START_FMT.format(depth=depth)
                    thread_parts.append(f"{comment_marker}{cleaned}{COMMENT_END}")
                thread_parts.append(THREAD_END)
                thread_text = "\n".join(thread_parts)

                thread_row = {
                    "text": thread_text,
                    "kind": "thread",
                    "source_id": post_id,
                    "source_field": "thread",
                    "length_bucket": length_bucket(thread_text),
                    "comment_count": len(ordered_comments),
                }
                serialized = json.dumps(thread_row, ensure_ascii=False) + "\n"
                thread_f.write(serialized)
                merged_f.write(serialized)
                stats["cpt_threads"] += 1
                stats["cpt_total_rows"] += 1

    payload = {"out_dir": str(out_dir), "stats": dict(stats)}
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
