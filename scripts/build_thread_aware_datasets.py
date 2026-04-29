#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PHONE_RE = re.compile(r"01\d[-\s]?\d{3,4}[-\s]?\d{4}")
URL_RE = re.compile(r"(https?://|open\.kakao|텔레|카톡|라인)", re.IGNORECASE)
PROMO_KW_RE = re.compile(
    r"(문의|카톡|텔레|라인|지원금|TC|실장|부장|출근문의|픽업|풀상주|면접|당일지급|지명비|이벤트|광고)",
    re.IGNORECASE,
)
HANGUL_RE = re.compile(r"[가-힣]")
SEPARATOR_RE = re.compile(r"([^\w\s가-힣]|ㅡ)\1{2,}")
COMMENT_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
COMMENT_REF_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")

BLOCKED_TEXTS = {
    "",
    "삭제된 댓글입니다.",
    "광고 권한이 없습니다.",
    "신고에 의해 블라인드 처리 되었습니다..",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="raw crawl에서 thread-aware 학습 데이터셋 생성")
    parser.add_argument("--raw-dir", required=True, help="raw crawl JSON 디렉터리")
    parser.add_argument("--out-dir", required=True, help="산출물 디렉터리")
    parser.add_argument("--post-excerpt-chars", type=int, default=220, help="comment SFT에 넣을 원글 길이")
    parser.add_argument("--max-promo-duplicates", type=int, default=2, help="동일 광고 텍스트 허용 개수")
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


def is_promo(text: str) -> bool:
    text = (text or "").strip()
    return bool(PHONE_RE.search(text) or URL_RE.search(text) or (PROMO_KW_RE.search(text) and len(text) > 60))


def is_valid_hybrid_user_part(text: str) -> bool:
    text = (text or "").strip()
    return len(text) > 13 and bool(HANGUL_RE.search(text)) and not is_promo(text)


def extract_hybrid_user_part(text: str) -> str | None:
    lines = [(line or "").strip() for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    if len(lines) < 2:
        return None

    for idx, line in enumerate(lines):
        if not line or not SEPARATOR_RE.search(line):
            continue
        candidate = "\n".join(lines[:idx]).strip()
        suffix = "\n".join(lines[idx:]).strip()
        if candidate and is_valid_hybrid_user_part(candidate) and is_promo(suffix):
            return candidate

    return None


def clean_comment(text: str) -> tuple[str | None, str]:
    text = (text or "").replace("\xa0", " ").strip()
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


def post_excerpt(title: str, content: str, limit: int) -> str:
    content = content.strip().replace("\r\n", "\n")
    if len(content) > limit:
        content = content[:limit].rstrip() + "..."
    if title:
        return f"제목: {title}\n원글: {content}"
    return f"원글: {content}"


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpt_path = out_dir / "cpt_corpus.filtered.jsonl"
    sft_path = out_dir / "sft_pairs.thread_aware.jsonl"
    summary_path = out_dir / "summary.json"

    promo_counter: Counter[str] = Counter()
    summary = Counter()

    with cpt_path.open("w", encoding="utf-8") as cpt_handle, sft_path.open("w", encoding="utf-8") as sft_handle:
        for row in iter_rows(raw_dir):
            summary["raw_posts"] += 1

            post_id = str(row.get("id") or "")
            title = str(row.get("title") or "").strip()
            content = str(row.get("content") or "").strip()
            merged_post = "\n".join([part for part in (title, content) if part]).strip()
            post_is_promo = is_promo(merged_post)

            if content and not post_is_promo:
                cpt_handle.write(
                    json.dumps(
                        {
                            "text": content,
                            "kind": "post",
                            "source_id": post_id,
                            "title": title,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                summary["cpt_posts"] += 1

                if title:
                    for instruction in (
                        f"다음 제목으로 커뮤니티 글을 자연스럽게 써라: {title}",
                        f"이 주제로 짧은 커뮤니티 글을 써라: {title}",
                    ):
                        sft_handle.write(
                            json.dumps(
                                {
                                    "instruction": instruction,
                                    "input": "",
                                    "output": content,
                                    "source_id": post_id,
                                    "pair_type": "post",
                                    "task_type": "post_from_title",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        summary["sft_post_pairs"] += 1

                first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
                if first_line and first_line != content:
                    sft_handle.write(
                        json.dumps(
                            {
                                "instruction": f"다음 첫 문장으로 이어지는 글을 써라: {first_line}",
                                "input": "",
                                "output": content,
                                "source_id": post_id,
                                "pair_type": "post",
                                "task_type": "post_continue",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    summary["sft_post_pairs"] += 1
            elif post_is_promo:
                summary["dropped_promo_posts"] += 1

            comment_map: dict[str, str] = {}
            normalized_comments: list[tuple[str | None, str]] = []
            for comment in row.get("comments") or []:
                if not isinstance(comment, dict):
                    continue
                summary["raw_comments"] += 1
                key, cleaned = clean_comment(str(comment.get("content") or ""))
                if cleaned in BLOCKED_TEXTS:
                    summary["dropped_blocked_comments"] += 1
                    continue

                promo = is_promo(cleaned)
                recovered_hybrid = extract_hybrid_user_part(cleaned) if promo else None
                effective_cleaned = recovered_hybrid or cleaned
                effective_promo = promo and recovered_hybrid is None

                if recovered_hybrid:
                    summary["recovered_hybrid_comments"] += 1

                if promo:
                    promo_counter[cleaned] += 1
                    if effective_promo and promo_counter[cleaned] > args.max_promo_duplicates:
                        summary["dropped_duplicate_promo_comments"] += 1
                        continue

                normalized_comments.append((key, effective_cleaned))
                if key is not None:
                    comment_map[key] = effective_cleaned

                if not effective_promo:
                    cpt_handle.write(
                        json.dumps(
                            {
                                "text": effective_cleaned,
                                "kind": "comment",
                                "source_id": post_id,
                                "comment_key": key,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    summary["cpt_comments"] += 1
                else:
                    summary["promo_comments_kept_for_context_only"] += 1

            if not content:
                continue

            base_context = post_excerpt(title, content, args.post_excerpt_chars)
            for key, cleaned in normalized_comments:
                if is_promo(cleaned):
                    continue

                parent = parent_key(key)
                if parent and comment_map.get(parent):
                    sft_handle.write(
                        json.dumps(
                            {
                                "instruction": "다음 커뮤니티 상황에서 자연스럽게 답글을 달아라.",
                                "input": f"{base_context}\n부모 댓글: {comment_map[parent]}",
                                "output": cleaned,
                                "source_id": post_id,
                                "pair_type": "comment",
                                "task_type": "reply_comment",
                                "comment_key": key,
                                "parent_comment_key": parent,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    summary["sft_reply_pairs"] += 1
                else:
                    sft_handle.write(
                        json.dumps(
                            {
                                "instruction": "다음 커뮤니티 글에 자연스럽게 댓글을 달아라.",
                                "input": base_context,
                                "output": cleaned,
                                "source_id": post_id,
                                "pair_type": "comment",
                                "task_type": "root_comment",
                                "comment_key": key,
                                "parent_comment_key": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    summary["sft_root_comment_pairs"] += 1

    payload = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "summary": dict(summary),
        "promo_comment_unique_templates": len(promo_counter),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
