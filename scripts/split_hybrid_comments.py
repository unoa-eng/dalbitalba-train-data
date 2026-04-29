#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PHONE_RE = re.compile(r"01\d[-\s.]?\d{3,4}[-\s.]?\d{4}")
URL_RE = re.compile(r"(https?://|open\.kakao|텔레|카톡|라인|kakao)", re.IGNORECASE)
PROMO_KW_RE = re.compile(
    r"(문의|카톡|텔레|라인|지원금|TC|실장|부장|출근문의|픽업|풀상주|면접|당일지급|지명비|이벤트|광고)",
    re.IGNORECASE,
)
COMMENT_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
COMMENT_REF_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")
KOREAN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
STRONG_PROMO_HEADER_RE = re.compile(
    r"(가라오케|부장|실장|팀장|출근문의|퍼블|셔츠|계좌이체|소개비|풀상주|지명비|출비|카톡|라인|kakao)",
    re.IGNORECASE,
)
DECORATIVE_START_RE = re.compile(r"^[❤♥♡💖★☆✅♾️ෆ✦✧✨◆◇▪️•]+")
INLINE_DECORATION_RE = re.compile(r"[?※✴✨★☆❤♥♡💖]{2,}")
VENUE_HEADER_RE = re.compile(
    r"^(?:[가-힣0-9.,·ㆍ: ]{0,24})?(도파민|세이렌|엘리트|가라오케|텐카페|유앤미|앤미|홀릭|터치|루이즈|달리는토끼|달토|베이직)(?:[가-힣0-9.,·ㆍ: ]{0,24})$"
)
PROMOISH_USER_RE = re.compile(
    r"(출근 문의|출근문의|카톡|kakao|라인|T\.?C\b|TC\b|지명비|당일지급|계좌이체|소개비|면접비|만근비|출퇴근 차량|풀상주|사물함|비품|2시간|3시간|4시간|5시간|6시간|9시 전|9시 후)",
    re.IGNORECASE,
)
SEPARATOR_CHARS = set("ㅡ~=-_★☆❤♥♡💖✨✦✧•·ㆍ◦◉○●◆◇□■═─━┌┐└┘│§-")

BLOCKED_TEXTS = {
    "",
    "삭제된 댓글입니다.",
    "광고 권한이 없습니다.",
    "신고에 의해 블라인드 처리 되었습니다..",
}

DEFAULT_GAP_TERMS = [
    {"term": "TC", "ratio": 0.1048},
    {"term": "밀빵", "ratio": 0.2289},
    {"term": "케어", "ratio": 0.2564},
    {"term": "쩜오", "ratio": 0.3979},
    {"term": "하이퍼", "ratio": 0.3486},
    {"term": "선릉", "ratio": 0.3980},
    {"term": "갯수", "ratio": 0.4588},
    {"term": "도파민", "ratio": 0.4992},
    {"term": "ㅡㅡ", "ratio": 0.4915},
]


@dataclass
class SplitCandidate:
    user_part: str
    promo_part: str
    split_reason: str
    separator: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="promo로 분류된 hybrid 댓글에서 사용자 파트를 복구 분석")
    parser.add_argument(
        "--raw-dir",
        default="/Users/unoa/Downloads/crawled-data-v2",
        help="raw crawl JSON 디렉터리",
    )
    parser.add_argument(
        "--diagnostic",
        default="runs/refinement-20260427-163226/cycle-1/diagnostic.json",
        help="GAP 용어를 읽을 diagnostic.json 경로",
    )
    parser.add_argument(
        "--out",
        default="runs/hybrid-comment-analysis.json",
        help="분석 결과 JSON 경로",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="용어/사유별 예시 최대 개수",
    )
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
                yield path, row


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


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip()


def contains_korean(text: str) -> bool:
    return bool(KOREAN_RE.search(text))


def is_valid_user_part(text: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= 13:
        reasons.append("too_short")
    if not contains_korean(text):
        reasons.append("no_korean")
    if is_promo(text) or PHONE_RE.search(text) or URL_RE.search(text) or PROMOISH_USER_RE.search(text):
        reasons.append("still_promo")
    return not reasons, reasons


def _line_separator_token(line: str) -> str:
    return re.sub(r"\s+", "", line).replace("\ufe0f", "")


def is_explicit_separator(line: str) -> bool:
    token = _line_separator_token(line)
    if len(token) < 3:
        return False
    if any(ch not in SEPARATOR_CHARS for ch in token):
        return False
    counts = Counter(token)
    return max(counts.values()) >= 3 and len(counts) <= 3


def starts_decorative_promo(line: str) -> bool:
    stripped = line.strip()
    return bool(DECORATIVE_START_RE.match(stripped))


def looks_like_promo_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        starts_decorative_promo(stripped)
        or INLINE_DECORATION_RE.search(stripped)
        or VENUE_HEADER_RE.search(stripped)
        or PHONE_RE.search(stripped)
        or URL_RE.search(stripped)
        or STRONG_PROMO_HEADER_RE.search(stripped)
    )


def join_lines(lines: list[str]) -> str:
    return normalize_text("\n".join(lines))


def extract_hybrid_candidate(text: str) -> SplitCandidate | None:
    lines = normalize_text(text).splitlines()
    if len(lines) < 2:
        return None

    for idx, line in enumerate(lines[1:], start=1):
        if not is_explicit_separator(line):
            continue
        user_part = join_lines(lines[:idx])
        promo_part = join_lines(lines[idx + 1 :])
        if user_part and promo_part and is_promo(promo_part):
            return SplitCandidate(
                user_part=user_part,
                promo_part=promo_part,
                split_reason="explicit_separator",
                separator=line.strip(),
            )

    for idx, line in enumerate(lines[1:], start=1):
        if not looks_like_promo_header(line):
            continue
        user_part = join_lines(lines[:idx])
        promo_part = join_lines(lines[idx:])
        if not user_part or not promo_part or not is_promo(promo_part):
            continue
        if looks_like_promo_header(lines[0]) and not is_explicit_separator(lines[idx - 1]):
            continue
        return SplitCandidate(
            user_part=user_part,
            promo_part=promo_part,
            split_reason="promo_header",
            separator=line.strip(),
        )

    return None


def load_gap_terms(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return DEFAULT_GAP_TERMS
    data = load_json(path)
    gap_terms = []
    for gap in data.get("gaps", []):
        if gap.get("phase") != "1a_term_freq":
            continue
        gap_terms.append({"term": gap["term"], "ratio": gap["ratio"]})
    return gap_terms or DEFAULT_GAP_TERMS


def add_limited_sample(bucket: list[dict[str, object]], sample: dict[str, object], limit: int) -> None:
    if len(bucket) < limit:
        bucket.append(sample)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gap_terms = load_gap_terms(Path(args.diagnostic))
    term_stats = {
        entry["term"]: {
            "gap_ratio": entry["ratio"],
            "recovered_comments": 0,
            "recovered_occurrences": 0,
            "samples": [],
        }
        for entry in gap_terms
    }

    summary = Counter()
    split_reason_counts = Counter()
    invalid_reason_counts = Counter()
    split_reason_samples: dict[str, list[dict[str, object]]] = {}
    invalid_reason_samples: dict[str, list[dict[str, object]]] = {}
    recovered_samples: list[dict[str, object]] = []

    for source_file, row in iter_rows(raw_dir):
        summary["raw_posts"] += 1
        source_id = str(row.get("id") or "")
        comments = row.get("comments") or []
        for idx, comment in enumerate(comments):
            if not isinstance(comment, dict):
                summary["non_dict_comments"] += 1
                continue

            summary["raw_comments"] += 1
            key, cleaned = clean_comment(str(comment.get("content") or ""))
            if cleaned in BLOCKED_TEXTS:
                summary["blocked_comments"] += 1
                continue
            if not is_promo(cleaned):
                continue

            summary["promo_comments"] += 1
            candidate = extract_hybrid_candidate(cleaned)
            if candidate is None:
                summary["promo_comments_without_split"] += 1
                continue

            summary["hybrid_split_candidates"] += 1
            split_reason_counts[candidate.split_reason] += 1
            base_sample = {
                "source_file": source_file.name,
                "source_id": source_id,
                "comment_index": idx,
                "comment_key": key,
                "separator": candidate.separator,
                "user_part": candidate.user_part,
            }
            add_limited_sample(
                split_reason_samples.setdefault(candidate.split_reason, []),
                base_sample,
                args.sample_limit,
            )

            valid, reasons = is_valid_user_part(candidate.user_part)
            if not valid:
                summary["hybrid_invalid_user_parts"] += 1
                for reason in reasons:
                    invalid_reason_counts[reason] += 1
                    add_limited_sample(
                        invalid_reason_samples.setdefault(reason, []),
                        base_sample,
                        args.sample_limit,
                    )
                continue

            summary["recovered_valid_comments"] += 1
            add_limited_sample(
                recovered_samples,
                {
                    **base_sample,
                    "split_reason": candidate.split_reason,
                    "promo_excerpt": candidate.promo_part[:240],
                },
                args.sample_limit * 3,
            )

            for term, stats in term_stats.items():
                occurrences = candidate.user_part.count(term)
                if occurrences <= 0:
                    continue
                stats["recovered_comments"] += 1
                stats["recovered_occurrences"] += occurrences
                add_limited_sample(
                    stats["samples"],
                    {
                        "source_file": source_file.name,
                        "source_id": source_id,
                        "comment_index": idx,
                        "comment_key": key,
                        "split_reason": candidate.split_reason,
                        "user_part": candidate.user_part,
                    },
                    args.sample_limit,
                )

    result = {
        "raw_dir": str(raw_dir),
        "diagnostic": str(Path(args.diagnostic)),
        "summary": dict(summary),
        "split_reason_counts": dict(split_reason_counts),
        "invalid_reason_counts": dict(invalid_reason_counts),
        "gap_term_recovery": term_stats,
        "split_reason_samples": split_reason_samples,
        "invalid_reason_samples": invalid_reason_samples,
        "recovered_samples": recovered_samples,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
