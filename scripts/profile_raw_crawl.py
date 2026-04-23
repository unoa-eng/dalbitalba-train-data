#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median


PHONE_RE = re.compile(r"01\d[-\s]?\d{3,4}[-\s]?\d{4}")
URL_RE = re.compile(r"(https?://|open\.kakao|텔레|카톡|라인)", re.IGNORECASE)
PROMO_KW_RE = re.compile(
    r"(문의|카톡|텔레|라인|지원금|TC|실장|부장|출근문의|픽업|풀상주|면접|당일지급|지명비|이벤트|광고)",
    re.IGNORECASE,
)
COMMENT_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
COMMENT_REF_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")

STOPWORDS = {
    "언니",
    "언니들",
    "진짜",
    "그냥",
    "지금",
    "근데",
    "이거",
    "저거",
    "이런",
    "그런",
    "하는",
    "있는",
    "같은",
    "너무",
    "요즘",
    "오늘",
    "이제",
    "이렇게",
    "하면서",
    "해서",
    "하면",
    "하고",
    "있는데",
    "아니고",
    "아니에요",
    "가게",
    "담당",
    "손님",
    "댓글",
    "게시글",
    "커뮤니티",
    "작성자",
    "비회원",
}

SLANG_PATTERNS = {
    "laugh": re.compile(r"ㅋ{2,}|ㅎ{2,}"),
    "cry": re.compile(r"ㅠ+|ㅜ+"),
    "abbrev": re.compile(r"ㄹㅇ|ㅈㄴ|ㄱㄱ|ㅂㅅ|ㄴㄴ|ㅁㅈ|ㅇㅈ|ㄱㅊ|하띠|상띠|텐카|하퍼|쩜오|초이스"),
    "formal": re.compile(r"(습니다|입니다|드립니다|하시길)$"),
}

CHOSUNG_PATTERNS = [r"ㅈㄴ", r"ㅇㅈ", r"ㄹㅇ", r"ㅅㅂ", r"ㅃ", r"ㄱㅊ", r"ㄱㄱ", r"ㄴㄴ", r"ㅂㅅ", r"ㄷㄷ", r"ㅁㅊ", r"ㄲㅃ", r"ㅇㅎ", r"ㅊㅇㅅ", r"ㅈㅂ", r"ㄹㅈㄷ"]
DOMAIN_SLANG_PATTERNS = [
    r"하퍼",
    r"초이스",
    r"밀빵",
    r"쩜오",
    r"수위",
    r"업진",
    r"진상",
    r"텐카",
    r"셔츠",
    r"상띠",
    r"퍼블",
    r"손놈",
    r"보도",
    r"가라",
    r"노도",
    r"하띠",
    r"뺑이",
    r"골타",
    r"유흥",
    r"파트너",
    r"룸보도",
    r"뼈가씨",
]
EMOTION_PATTERNS = [
    r"ㅋ{2,}",
    r"ㅠ+",
    r"ㅜ+",
    r"ㅎ{2,}",
    r"존나",
    r"후회",
    r"짜증",
    r"무서",
    r"억울",
    r"현타",
    r"개빡",
    r"불안",
    r"웃겨",
    r"소름",
    r"미치겠",
    r"열받",
    r"빡침",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="raw crawl 디렉터리 프로파일링")
    parser.add_argument("--raw-dir", required=True, help="raw crawl JSON 디렉터리")
    parser.add_argument("--output", help="결과 JSON 저장 경로")
    parser.add_argument("--top-k", type=int, default=30, help="상위 항목 개수")
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


def percentile(values: list[int], q: float) -> int | float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def top_tokens(counter: Counter, top_k: int) -> list[list[object]]:
    return [[token, count] for token, count in counter.most_common(top_k)]


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)

    board_counter: Counter[str] = Counter()
    post_year_counter: Counter[str] = Counter()
    crawl_month_counter: Counter[str] = Counter()
    comment_depth_counter: Counter[int] = Counter()
    post_token_counter: Counter[str] = Counter()
    title_token_counter: Counter[str] = Counter()
    comment_token_counter: Counter[str] = Counter()
    duplicate_post_counter: Counter[str] = Counter()
    duplicate_comment_counter: Counter[str] = Counter()
    chosung_counter: Counter[str] = Counter()
    domain_slang_counter: Counter[str] = Counter()
    emotion_counter: Counter[str] = Counter()

    post_lengths: list[int] = []
    comment_lengths: list[int] = []
    comments_per_post: list[int] = []

    post_count = 0
    comment_count = 0
    promo_post_count = 0
    promo_comment_count = 0
    deleted_comment_count = 0
    malformed_comment_prefix = 0
    comment_question_count = 0
    comment_exclaim_count = 0
    comment_ellipsis_count = 0
    comment_mixed_punct_count = 0
    comment_short_turn_count = 0
    slang_counts = {f"post_{name}": 0 for name in SLANG_PATTERNS}
    slang_counts.update({f"comment_{name}": 0 for name in SLANG_PATTERNS})
    chosung_compiled = [(pattern, re.compile(pattern)) for pattern in CHOSUNG_PATTERNS]
    domain_slang_compiled = [(pattern, re.compile(pattern)) for pattern in DOMAIN_SLANG_PATTERNS]
    emotion_compiled = [(pattern, re.compile(pattern)) for pattern in EMOTION_PATTERNS]

    for row in iter_rows(raw_dir):
        post_count += 1
        board_counter[str(row.get("boardName") or "(none)")] += 1

        date = str(row.get("date") or "")
        if len(date) >= 4 and date[:4].isdigit():
            post_year_counter[date[:4]] += 1

        crawled = str(row.get("crawledAt") or "")
        if len(crawled) >= 7:
            crawl_month_counter[crawled[:7]] += 1

        title = str(row.get("title") or "").strip()
        content = str(row.get("content") or "").strip()
        merged_post = "\n".join([part for part in (title, content) if part]).strip()

        if content:
            post_lengths.append(len(content))
            duplicate_post_counter[content] += 1
            for name, pattern in SLANG_PATTERNS.items():
                slang_counts[f"post_{name}"] += int(bool(pattern.search(content)))
            for pattern, compiled in chosung_compiled:
                chosung_counter[pattern] += int(bool(compiled.search(content)))
            for pattern, compiled in domain_slang_compiled:
                domain_slang_counter[pattern] += int(bool(compiled.search(content)))
            for pattern, compiled in emotion_compiled:
                emotion_counter[pattern] += int(bool(compiled.search(content)))

        if is_promo(merged_post):
            promo_post_count += 1
        elif merged_post:
            for token in TOKEN_RE.findall(merged_post):
                if token not in STOPWORDS:
                    post_token_counter[token] += 1
            for token in TOKEN_RE.findall(title):
                if token not in STOPWORDS:
                    title_token_counter[token] += 1

        comments = row.get("comments") or []
        comments_per_post.append(len(comments))
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            raw_text = str(comment.get("content") or "").strip()
            if not raw_text:
                continue

            comment_count += 1
            key, cleaned = clean_comment(raw_text)
            comment_lengths.append(len(raw_text))
            duplicate_comment_counter[cleaned] += 1

            if key is None:
                malformed_comment_prefix += 1
            else:
                comment_depth_counter[key.count("-")] += 1

            if cleaned == "삭제된 댓글입니다.":
                deleted_comment_count += 1

            for name, pattern in SLANG_PATTERNS.items():
                slang_counts[f"comment_{name}"] += int(bool(pattern.search(raw_text)))
            for pattern, compiled in chosung_compiled:
                chosung_counter[pattern] += int(bool(compiled.search(raw_text)))
            for pattern, compiled in domain_slang_compiled:
                domain_slang_counter[pattern] += int(bool(compiled.search(raw_text)))
            for pattern, compiled in emotion_compiled:
                emotion_counter[pattern] += int(bool(compiled.search(raw_text)))

            comment_question_count += int("?" in raw_text)
            comment_exclaim_count += int("!" in raw_text)
            comment_ellipsis_count += int(".." in raw_text or "..." in raw_text)
            comment_mixed_punct_count += int("?" in raw_text and "!" in raw_text)
            comment_short_turn_count += int(len(raw_text.strip()) <= 12)

            if is_promo(raw_text):
                promo_comment_count += 1
            elif cleaned:
                for token in TOKEN_RE.findall(cleaned):
                    if token not in STOPWORDS:
                        comment_token_counter[token] += 1

    report = {
        "raw_dir": str(raw_dir),
        "post_count": post_count,
        "comment_count": comment_count,
        "boards": [[board, count] for board, count in board_counter.most_common(args.top_k)],
        "post_years": [[year, count] for year, count in post_year_counter.most_common()],
        "crawl_months": [[month, count] for month, count in crawl_month_counter.most_common()],
        "post_length": {
            "avg": round(sum(post_lengths) / max(1, len(post_lengths)), 1),
            "p50": median(post_lengths) if post_lengths else 0,
            "p90": percentile(post_lengths, 0.9),
        },
        "comment_length": {
            "avg": round(sum(comment_lengths) / max(1, len(comment_lengths)), 1),
            "p50": median(comment_lengths) if comment_lengths else 0,
            "p90": percentile(comment_lengths, 0.9),
        },
        "comments_per_post": {
            "avg": round(sum(comments_per_post) / max(1, len(comments_per_post)), 2),
            "p50": median(comments_per_post) if comments_per_post else 0,
            "p90": percentile(comments_per_post, 0.9),
            "max": max(comments_per_post) if comments_per_post else 0,
        },
        "comment_depth": [[depth, count] for depth, count in sorted(comment_depth_counter.items())],
        "promo": {
            "posts": promo_post_count,
            "comments": promo_comment_count,
            "post_rate": round(promo_post_count / max(1, post_count), 4),
            "comment_rate": round(promo_comment_count / max(1, comment_count), 4),
        },
        "deleted_comment_count": deleted_comment_count,
        "malformed_comment_prefix": malformed_comment_prefix,
        "slang_markers": slang_counts,
        "chosung_top": top_tokens(chosung_counter, args.top_k),
        "domain_slang_top": top_tokens(domain_slang_counter, args.top_k),
        "emotion_top": top_tokens(emotion_counter, args.top_k),
        "comment_dialogue_style": {
            "question_rate": round(comment_question_count / max(1, comment_count), 4),
            "exclaim_rate": round(comment_exclaim_count / max(1, comment_count), 4),
            "ellipsis_rate": round(comment_ellipsis_count / max(1, comment_count), 4),
            "mixed_punct_rate": round(comment_mixed_punct_count / max(1, comment_count), 4),
            "short_turn_rate": round(comment_short_turn_count / max(1, comment_count), 4),
        },
        "top_post_tokens": top_tokens(post_token_counter, args.top_k),
        "top_title_tokens": top_tokens(title_token_counter, args.top_k),
        "top_comment_tokens": top_tokens(comment_token_counter, args.top_k),
        "duplicate_posts_over1": sum(count > 1 for count in duplicate_post_counter.values()),
        "duplicate_comments_over1": sum(count > 1 for count in duplicate_comment_counter.values()),
        "top_duplicate_comments": top_tokens(
            Counter({text: count for text, count in duplicate_comment_counter.items() if count > 1}),
            args.top_k,
        ),
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
