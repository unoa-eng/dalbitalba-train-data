#!/usr/bin/env python3
"""Audit final Phase 2 style and metadata against source-facing targets."""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path("/Users/unoa/dalbitalba-train-data")
FINAL = Path(sys.argv[1])
SOURCE_FLAT = REPO / "runs/cycle10/data-representative-v1-clean/threads_flat/train.jsonl"
CPT = REPO / "cpt_enriched.jsonl"

CONTAM_RE = re.compile(
    r"신고에\s*의해\s*블라인드|블라인드\s*(?:처리|되었습니다|된\s*게시)|"
    r"관리자에 의해|신고가 접수|이용이 제한|운영원칙|"
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
INDEX_RE = re.compile(r"^\[\d+(?:-\d+)?\]\s*")
HONORIFIC_RE = re.compile(
    r"(습니다|습니까|세요|셔요|어요|아요|해요|네요|겠네요|군요|죠|나요|"
    r"드릴게요|하시|해주세요|해보세요|같아요|좋아요|됩니다|입니다|입니까|"
    r"인데요|거예요|게요|되세요|보세요)"
)
COMPAT_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")

SLANG = [
    "티씨", "tc", "TC", "마담", "담당", "초이스", "콜", "하퍼", "퍼블릭", "텐카",
    "룸", "오카", "오픈채팅", "가게", "출근", "퇴근", "손님", "진상", "정산",
    "팁", "떼초", "베팅", "롱치", "셔츠", "풀티", "보도", "웨터", "언니",
]
INITIALISMS = [
    "ㅇㅈ", "ㄱㅊ", "ㄴㄴ", "ㄹㅇ", "ㅈㄴ", "ㅎㄷㄷ", "ㄷㄷ", "ㅅㅂ", "ㅂㄷ",
    "ㄴㄷ", "ㅇㅇ", "ㄱㄱ", "ㅊㅊ", "ㅈ됨", "개꿀", "존맛",
    "ᄋᄌ", "ᄀᄎ", "ᄂᄂ", "ᄅᄋ", "ᄌᄂ", "ᄒᄃᄃ", "ᄃᄃ", "ᄉᄇ", "ᄇᄃ",
    "ᄂᄃ", "ᄋᄋ", "ᄀᄀ", "ᄎᄎ", "ᄌ됨",
]


def pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0


def num_stats(xs: list[int | float]) -> dict:
    ys = sorted(xs)
    n = len(ys)
    return {
        "mean": round(statistics.mean(ys), 2),
        "median": statistics.median(ys),
        "p90": ys[int(0.9 * (n - 1))],
        "max": max(ys),
    }


def text_stats(xs: list[str]) -> dict:
    lens = [len(x) for x in xs]
    return {
        "n": len(xs),
        "avg_len": round(statistics.mean(lens), 2),
        "median_len": statistics.median(lens),
        "p90_len": sorted(lens)[int(0.9 * (len(lens) - 1))],
        "newline_pct": pct(sum("\n" in x for x in xs), len(xs)),
        "honorific_pct": pct(sum(bool(HONORIFIC_RE.search(x)) for x in xs), len(xs)),
        "digit_pct": pct(sum(bool(re.search(r"\d", x)) for x in xs), len(xs)),
        "compat_jamo_rows": sum(bool(COMPAT_RE.search(x)) for x in xs),
        "old_jamo_k_rows": sum("ᄏ" in x for x in xs),
        "unique_pct": pct(len(set(xs)), len(xs)),
        "slang_pct": pct(sum(any(tok in x for tok in SLANG) for x in xs), len(xs)),
        "initialism_pct": pct(sum(any(tok in x for tok in INITIALISMS) for x in xs), len(xs)),
        "ad_contact_rows": sum(bool(CONTAM_RE.search(x)) for x in xs),
    }


def token_counts(xs: list[str], tokens: list[str]) -> dict:
    counts = Counter()
    for x in xs:
        for tok in tokens:
            if tok in x:
                counts[tok] += 1
    return dict(counts.most_common())


def load_source_texts() -> tuple[list[str], list[str]]:
    source_all, source_comments = [], []
    for line in SOURCE_FLAT.open():
        row = json.loads(line)
        text = (row.get("text") or "").strip()
        if len(text) <= 3 or CONTAM_RE.search(text):
            continue
        source_all.append(text)
        if row.get("kind") in {None, "comment"}:
            source_comments.append(text)
    return source_all, source_comments


def load_source_metadata() -> dict:
    views, comments = [], []
    for line in CPT.open():
        row = json.loads(line)
        if row.get("kind") == "post" and isinstance(row.get("views"), int) and isinstance(row.get("comment_count"), int):
            views.append(row["views"])
            comments.append(row["comment_count"])
    return {
        "posts": len(views),
        "views_empirical": num_stats(views),
        "comment_count_empirical": num_stats(comments),
    }


def main() -> int:
    final_comments: list[str] = []
    final_posts: list[str] = []
    views: list[int] = []
    likes: list[int] = []
    counts: list[int] = []
    hours: list[int] = []
    gaps: list[float] = []
    timing_order_errors = 0

    for line in FINAL.open():
        thread = json.loads(line)
        final_posts.extend([thread["title"], thread["content"]])
        views.append(int(thread["views"]))
        likes.append(int(thread["likes"]))
        counts.append(int(thread["commentCount"]))
        prev = None
        for comment in thread.get("comments") or []:
            final_comments.append(INDEX_RE.sub("", comment["content"]).strip())
            dt = datetime.strptime(comment["date"], "%Y-%m-%d %H:%M:%S")
            hours.append(dt.hour)
            if prev is not None:
                gaps.append((dt - prev).total_seconds() / 3600)
                if dt <= prev:
                    timing_order_errors += 1
            prev = dt

    source_all, source_comments = load_source_texts()
    report = {
        "final_comments": text_stats(final_comments),
        "source_comments": text_stats(source_comments),
        "source_audit_all_texts": text_stats(source_all),
        "final_comment_slang_counts": token_counts(final_comments, SLANG),
        "source_comment_slang_counts": token_counts(source_comments, SLANG),
        "final_comment_initialism_counts": token_counts(final_comments, INITIALISMS),
        "source_comment_initialism_counts": token_counts(source_comments, INITIALISMS),
        "final_posts": text_stats(final_posts),
        "final_metadata": {
            "threads": len(views),
            "views": num_stats(views),
            "likes": num_stats(likes),
            "commentCount": num_stats(counts),
            "likes_zero_pct": pct(sum(x == 0 for x in likes), len(likes)),
            "comment_timing_order_errors": timing_order_errors,
            "comment_hour_top": Counter(hours).most_common(8),
            "gap_hours_median": round(statistics.median(gaps), 3),
            "gap_hours_p90": round(sorted(gaps)[int(0.9 * (len(gaps) - 1))], 3),
        },
        "source_metadata_reference": load_source_metadata(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if timing_order_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
