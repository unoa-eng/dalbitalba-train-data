#!/usr/bin/env python3
"""
build_5task_sft_v3.py — 5-Task SFT 데이터셋 생성

TRAINING_DESIGN_V3.md Stage 2 구현:
- T1: 제목 → 본문 생성
- T2: 게시글 → Root 댓글 생성 (full post context)
- T3: 게시글 + 부모댓글 → Reply 댓글 생성 (thread context)
- T4: 주제 → 짧은 게시글 생성
- T5: 첫 문장 → 이어쓰기

Usage:
    python scripts/build_5task_sft_v3.py --raw-dir /path/to/raw --out-dir ./v3-data
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


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
AD_DENSE_RE = re.compile(
    r"(밀빵\s*확실|밀빵\s*가능|풀상주\s*풀케어|스타트톡\s*개수톡|"
    r"하루\s*평균\s*\d+\s*방|[1-3]부\s*\d+인\s*\d+조)",
    re.IGNORECASE,
)
COMMENT_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
COMMENT_REF_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")
BLOCKED_TEXTS = {
    "",
    "삭제된 댓글입니다.",
    "광고 권한이 없습니다.",
    "신고에 의해 블라인드 처리 되었습니다..",
}
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

# Topic keywords for T4 instruction generation
TOPIC_TEMPLATES = [
    "다음 주제로 퀸알바 커뮤니티에 올릴 글을 써라: {topic}",
    "이 주제로 짧은 커뮤니티 글을 써라: {topic}",
]
TOPIC_EXTRACTORS = [
    re.compile(r"(출근|퇴근|컨디션|매출|손님|술|가게|이직|면접|다이어트|성형|세금|재테크|옷|명품|경기|시세)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-Task SFT 데이터셋 v3")
    parser.add_argument("--raw-dir", required=True, help="raw crawl JSON 디렉터리")
    parser.add_argument("--out-dir", required=True, help="산출물 디렉터리")
    parser.add_argument("--post-excerpt-chars", type=int, default=300, help="댓글 SFT에 넣을 원글 최대 길이")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path):
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=enc))
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
    text = (text or "").strip()
    if not text:
        return False
    if bool(PHONE_RE.search(text) or URL_RE.search(text) or CONTACT_ID_RE.search(text)):
        return True
    if len(AD_DENSE_RE.findall(text)) >= 2:
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


def post_context(title: str, content: str, limit: int) -> str:
    content = content.strip().replace("\r\n", "\n")
    if len(content) > limit:
        content = content[:limit].rstrip() + "..."
    parts = []
    if title:
        parts.append(f"제목: {title}")
    parts.append(f"원글: {content}")
    return "\n".join(parts)


def extract_topic(title: str, content: str) -> str | None:
    combined = f"{title} {content}"
    for pat in TOPIC_EXTRACTORS:
        m = pat.search(combined)
        if m:
            return m.group(0)
    return None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_path = out_dir / "sft_5task_v3.jsonl"
    summary_path = out_dir / "sft_v3_summary.json"
    stats = Counter()

    with sft_path.open("w", encoding="utf-8") as f:
        for row in iter_rows(raw_dir):
            stats["raw_posts"] += 1

            post_id = str(row.get("id") or "")
            title = str(row.get("title") or "").strip()
            content = str(row.get("content") or "").strip()
            merged = "\n".join([p for p in (title, content) if p]).strip()

            if not content or is_promo_v2(merged) or minor_sexual_proximity(merged):
                stats["skipped_post"] += 1
                continue

            # === T1: 제목 → 본문 ===
            if title and len(content) >= 10:
                f.write(
                    json.dumps(
                        {
                            "task": "T1_title_to_body",
                            "instruction": f"다음 제목으로 퀸알바 커뮤니티 글을 자연스럽게 써라: {title}",
                            "input": "",
                            "output": content,
                            "source_id": post_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["T1_title_to_body"] += 1

            # === T4: 주제 → 짧은 게시글 ===
            topic = extract_topic(title, content)
            if topic and len(content) <= 200:
                template = random.choice(TOPIC_TEMPLATES)
                f.write(
                    json.dumps(
                        {
                            "task": "T4_topic_to_post",
                            "instruction": template.format(topic=topic),
                            "input": "",
                            "output": f"{title}\n{content}" if title else content,
                            "source_id": post_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["T4_topic_to_post"] += 1

            # === T5: 첫 문장 → 이어쓰기 ===
            first_line = next(
                (line.strip() for line in content.splitlines() if line.strip()), ""
            )
            if first_line and first_line != content and len(content) >= 20:
                f.write(
                    json.dumps(
                        {
                            "task": "T5_continue",
                            "instruction": f"다음 첫 문장으로 이어지는 커뮤니티 글을 써라: {first_line}",
                            "input": "",
                            "output": content,
                            "source_id": post_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["T5_continue"] += 1

            # === Process comments for T2/T3 ===
            comment_map: dict[str, str] = {}
            normalized: list[tuple[str | None, str]] = []

            for comment in row.get("comments") or []:
                if not isinstance(comment, dict):
                    continue
                key, cleaned = clean_comment(str(comment.get("content") or ""))
                if cleaned in BLOCKED_TEXTS or len(cleaned) < 5:
                    continue
                if is_promo_v2(cleaned) or minor_sexual_proximity(cleaned):
                    continue
                normalized.append((key, cleaned))
                if key is not None:
                    comment_map[key] = cleaned

            if not content:
                continue

            base_ctx = post_context(title, content, args.post_excerpt_chars)

            for key, cleaned in normalized:
                parent = parent_key(key)

                if parent and comment_map.get(parent):
                    # === T3: Thread → Reply 댓글 ===
                    ctx = f"{base_ctx}\n부모 댓글: {comment_map[parent]}"
                    f.write(
                        json.dumps(
                            {
                                "task": "T3_reply_comment",
                                "instruction": "다음 커뮤니티 대화에서 자연스럽게 답글을 달아라.",
                                "input": ctx,
                                "output": cleaned,
                                "source_id": post_id,
                                "comment_key": key,
                                "parent_key": parent,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["T3_reply_comment"] += 1
                else:
                    # === T2: 게시글 → Root 댓글 ===
                    f.write(
                        json.dumps(
                            {
                                "task": "T2_root_comment",
                                "instruction": "다음 커뮤니티 글에 자연스럽게 댓글을 달아라.",
                                "input": base_ctx,
                                "output": cleaned,
                                "source_id": post_id,
                                "comment_key": key,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["T2_root_comment"] += 1

    payload = {"out_dir": str(out_dir), "stats": dict(stats)}
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
