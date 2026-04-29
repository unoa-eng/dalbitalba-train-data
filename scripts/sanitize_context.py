#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = Path("/Users/unoa/Downloads/crawled-data-v2")
DEFAULT_BASE_CORPUS = PROJECT_ROOT / "cpt_corpus.v2.jsonl"
DEFAULT_PATCH_OUT = PROJECT_ROOT / "cpt_patch_gap_repair.jsonl"
DEFAULT_MERGED_OUT = PROJECT_ROOT / "cpt_corpus.v3.jsonl"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "runs" / "cpt_patch_gap_repair_report.md"

PHONE_RE = re.compile(r"01\d[-\s]?\d{3,4}[-\s]?\d{4}")
URL_RE = re.compile(r"(https?://\S+|www\.\S+|open\.kakao\S*)", re.IGNORECASE)
URL_LIKE_RE = re.compile(r"(https?://|www\.|open\.kakao)", re.IGNORECASE)
PROMO_KW_RE = re.compile(
    r"(문의|카톡|텔레|라인|지원금|TC|실장|부장|출근문의|픽업|풀상주|면접|당일지급|지명비|이벤트|광고)",
    re.IGNORECASE,
)
COMMENT_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
COMMENT_REF_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")
HANGUL_RE = re.compile(r"[가-힣]")

BLOCKED_TEXTS = {
    "",
    "삭제된 댓글입니다.",
    "광고 권한이 없습니다.",
    "신고에 의해 블라인드 처리 되었습니다..",
}

PII_PATTERNS = [
    re.compile(r"카카오\s*(?:톡\s*)?(?:아이디|ID)?\s*[:：]?\s*[A-Za-z0-9_.\-]{3,}", re.IGNORECASE),
    re.compile(r"카톡\s*(?:아이디|ID)?\s*[:：]?\s*[A-Za-z0-9_.\-]{3,}", re.IGNORECASE),
    re.compile(r"카카오\s+[A-Za-z][A-Za-z0-9_.\-]{2,}", re.IGNORECASE),
    re.compile(r"카톡\s+[A-Za-z][A-Za-z0-9_.\-]{2,}", re.IGNORECASE),
    re.compile(r"kakao\s*(?:id)?\s*[:：]?\s*[A-Za-z0-9_.\-]{3,}", re.IGNORECASE),
    re.compile(r"라인\s*(?:아이디|ID)?\s*[:：]?\s*[A-Za-z0-9_.\-]{3,}", re.IGNORECASE),
    re.compile(r"line\s*(?:id)?\s*[:：]?\s*[A-Za-z0-9_.\-]{3,}", re.IGNORECASE),
]

TERM_PATTERNS = {
    "TC": re.compile(r"(?<![A-Za-z])T\.?\s*C\.?(?![A-Za-z])", re.IGNORECASE),
    "밀빵": re.compile(r"밀빵"),
    "케어": re.compile(r"케어"),
    "쩜오": re.compile(r"쩜오"),
    "하이퍼": re.compile(r"하이퍼"),
    "도파민": re.compile(r"도파민"),
    "갯수": re.compile(r"갯수"),
    "선릉": re.compile(r"선릉"),
    "ㅡㅡ": re.compile(r"ㅡㅡ"),
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
    parser = argparse.ArgumentParser(description="promo 댓글에서 GAP 용어 패치 코퍼스 생성")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="raw crawl JSON 디렉터리")
    parser.add_argument("--base-corpus", type=Path, default=DEFAULT_BASE_CORPUS, help="기존 cpt corpus")
    parser.add_argument("--patch-out", type=Path, default=DEFAULT_PATCH_OUT, help="패치 JSONL 출력 경로")
    parser.add_argument("--merged-out", type=Path, default=DEFAULT_MERGED_OUT, help="append 결과 JSONL 출력 경로")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT, help="리포트 Markdown 출력 경로")
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
    return bool(PHONE_RE.search(text) or URL_LIKE_RE.search(text) or (PROMO_KW_RE.search(text) and len(text) > 60))


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


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def strip_pii(text: str) -> tuple[str, Counter]:
    counts: Counter = Counter()

    def remove_with(pattern: re.Pattern[str], label: str, value: str) -> str:
        def repl(match: re.Match[str]) -> str:
            counts[label] += 1
            return " "

        return pattern.sub(repl, value)

    text = remove_with(PHONE_RE, "phone", text)
    text = remove_with(URL_RE, "url", text)
    for pattern in PII_PATTERNS:
        text = remove_with(pattern, "contact_id", text)

    cleaned_lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = normalize_spaces(raw_line)
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip(), counts


def split_sentences(text: str) -> list[str]:
    segments: list[str] = []
    for line in text.splitlines():
        line = normalize_spaces(line)
        if not line:
            continue
        parts = re.split(r"(?<=[.!?！？])\s+|(?<=[다요죠네함임])\s{2,}", line)
        for part in parts:
            candidate = normalize_spaces(part)
            if candidate:
                segments.append(candidate)
    return segments


def sentence_terms(text: str) -> list[str]:
    matched = []
    for term, pattern in TERM_PATTERNS.items():
        if pattern.search(text):
            matched.append(term)
    return matched


def keep_sentence(text: str) -> bool:
    return len(text) >= 13 and bool(HANGUL_RE.search(text))


def length_bucket(text: str) -> str:
    length = len(text or "")
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length <= hi:
            return name
    return "unk"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def build_report(
    *,
    raw_dir: Path,
    base_corpus: Path,
    patch_out: Path,
    merged_out: Path,
    summary: Counter,
    term_counts: Counter,
    pii_totals: Counter,
    term_samples: dict[str, list[str]],
) -> str:
    lines = [
        "# GAP Repair Patch Report",
        "",
        f"- Raw dir: `{raw_dir}`",
        f"- Base corpus: `{base_corpus.name}`",
        f"- Patch file: `{patch_out.name}`",
        f"- Merged corpus: `{merged_out.name}`",
        "",
        "## Summary",
        "",
        f"- Promo comments scanned: {summary['promo_comments']:,}",
        f"- Promo comments kept after extraction: {summary['comments_with_patch']:,}",
        f"- Patch rows written: {summary['patch_rows']:,}",
        f"- Base corpus rows: {summary['base_rows']:,}",
        f"- Merged corpus rows: {summary['merged_rows']:,}",
        "",
        "## PII Removed",
        "",
        f"- Phone numbers removed: {pii_totals['phone']:,}",
        f"- Kakao/Line IDs removed: {pii_totals['contact_id']:,}",
        f"- URLs removed: {pii_totals['url']:,}",
        "",
        "## GAP Term Counts",
        "",
        "| Term | Added Rows | Sample |",
        "| --- | ---: | --- |",
    ]

    for term in TERM_PATTERNS:
        sample = term_samples.get(term, ["-"])[0]
        lines.append(f"| {term} | {term_counts[term]:,} | {sample.replace('|', '\\|')} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    patch_rows: list[dict[str, str]] = []
    pii_totals: Counter = Counter()
    term_counts: Counter = Counter()
    term_samples: dict[str, list[str]] = defaultdict(list)
    summary: Counter = Counter()

    for _, row in iter_rows(args.raw_dir):
        post_id = str(row.get("id") or "")
        for idx, comment in enumerate(row.get("comments") or [], start=1):
            if not isinstance(comment, dict):
                continue
            _, cleaned = clean_comment(str(comment.get("content") or ""))
            if cleaned in BLOCKED_TEXTS or not cleaned:
                continue
            if not is_promo(cleaned):
                continue

            summary["promo_comments"] += 1
            pii_cleaned, pii_counts = strip_pii(cleaned)
            pii_totals.update(pii_counts)
            if not pii_cleaned:
                continue

            kept_any = False
            source_id = str(comment.get("id") or f"{post_id}:comment:{idx}")
            for sentence in split_sentences(pii_cleaned):
                matched_terms = sentence_terms(sentence)
                if not matched_terms:
                    continue
                if not keep_sentence(sentence):
                    continue

                patch_rows.append(
                    {
                        "text": sentence,
                        "kind": "comment",
                        "source_id": source_id,
                        "source_field": "comment",
                        "length_bucket": length_bucket(sentence),
                    }
                )
                summary["patch_rows"] += 1
                kept_any = True
                for term in matched_terms:
                    term_counts[term] += 1
                    if len(term_samples[term]) < 3:
                        term_samples[term].append(sentence)

            if kept_any:
                summary["comments_with_patch"] += 1

    with args.patch_out.open("w", encoding="utf-8") as handle:
        for row in patch_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["base_rows"] = count_lines(args.base_corpus)
    summary["merged_rows"] = summary["base_rows"] + summary["patch_rows"]

    with args.base_corpus.open("r", encoding="utf-8") as src, args.merged_out.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
        for row in patch_rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(
        raw_dir=args.raw_dir,
        base_corpus=args.base_corpus,
        patch_out=args.patch_out,
        merged_out=args.merged_out,
        summary=summary,
        term_counts=term_counts,
        pii_totals=pii_totals,
        term_samples=term_samples,
    )
    args.report_out.write_text(report, encoding="utf-8")

    print(f"patch_rows={summary['patch_rows']}")
    print(f"base_rows={summary['base_rows']}")
    print(f"merged_rows={summary['merged_rows']}")
    for term in TERM_PATTERNS:
        print(f"{term}\t{term_counts[term]}")


if __name__ == "__main__":
    main()
