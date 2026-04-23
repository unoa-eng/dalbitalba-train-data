from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CPT = ROOT / "cpt_corpus.jsonl"
DEFAULT_SFT = ROOT / "sft_pairs_v2.jsonl"
LEAD_TAG_RE = re.compile(r"^(?:[^\[\n]{0,30})?\[(\d+(?:-\d+)*)\]\s*")
REF_TAG_RE = re.compile(r"^(?:\S+\s+)?\[(\d+(?:-\d+)*)\]\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="원천 crawl JSON과 학습 스냅샷 간 정합성을 검증합니다."
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="원천 crawl JSON 디렉터리 경로",
    )
    parser.add_argument(
        "--cpt-path",
        default=str(DEFAULT_CPT),
        help="검증할 cpt_corpus.jsonl 경로",
    )
    parser.add_argument(
        "--sft-path",
        default=str(DEFAULT_SFT),
        help="검증할 sft_pairs_v2.jsonl 경로",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=8,
        help="불일치 예시를 몇 개까지 포함할지",
    )
    parser.add_argument(
        "--output",
        help="결과 JSON을 저장할 경로",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def iter_raw_rows(raw_dir: Path):
    for path in sorted(raw_dir.glob("*.json")):
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict):
                yield row


def parse_comment_text(text: str) -> tuple[str | None, str]:
    text = (text or "").replace("\xa0", " ").strip()
    key = None

    lead_match = LEAD_TAG_RE.match(text)
    if lead_match:
        key = lead_match.group(1)
        text = text[lead_match.end() :].strip()

    while True:
        ref_match = REF_TAG_RE.match(text)
        if not ref_match:
            break
        text = text[ref_match.end() :].strip()

    return key, text


def normalize_comment_text(text: str) -> str:
    _, body = parse_comment_text(text)
    return body


def parent_key(key: str | None) -> str | None:
    if not key or "-" not in key:
        return None
    return key.rsplit("-", 1)[0]


def load_cpt_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_sft_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def validate(raw_dir: Path, cpt_path: Path, sft_path: Path, sample_limit: int) -> dict[str, Any]:
    raw_by_id: dict[str, dict[str, Any]] = {}
    raw_post_texts: set[str] = set()
    raw_comment_texts_normalized: set[str] = set()
    raw_comment_count = 0

    for row in iter_raw_rows(raw_dir):
        row_id = row.get("id")
        if row_id is not None:
            raw_by_id[str(row_id)] = row
        content = row.get("content")
        if content:
            raw_post_texts.add(content)
        for comment in row.get("comments") or []:
            if isinstance(comment, dict) and comment.get("content"):
                raw_comment_count += 1
                raw_comment_texts_normalized.add(normalize_comment_text(comment["content"]))

    cpt_rows = load_cpt_rows(cpt_path)
    sft_rows = load_sft_rows(sft_path)

    cpt_kind_counter = Counter(row.get("kind") for row in cpt_rows)
    sft_pair_counter = Counter(row.get("pair_type") for row in sft_rows)

    cpt_post_total = 0
    cpt_post_exact_match = 0
    cpt_comment_total = 0
    cpt_comment_normalized_match = 0
    cpt_comment_unmatched: list[str] = []

    for row in cpt_rows:
        text = (row.get("text") or "").replace("\xa0", " ").strip()
        if row.get("kind") == "post":
            cpt_post_total += 1
            if text in raw_post_texts:
                cpt_post_exact_match += 1
        elif row.get("kind") == "comment":
            cpt_comment_total += 1
            if text in raw_comment_texts_normalized:
                cpt_comment_normalized_match += 1
            elif len(cpt_comment_unmatched) < sample_limit:
                cpt_comment_unmatched.append(text)

    sft_post_total = 0
    sft_post_exact_match = 0
    sft_comment_total = 0
    sft_comment_instruction_match = 0
    sft_comment_output_match = 0
    sft_comment_reply_pair_match = 0
    sft_missing_source_ids: list[str] = []
    sft_comment_unmatched: list[dict[str, Any]] = []

    for row in sft_rows:
        source_id = str(row.get("source_id"))
        raw_row = raw_by_id.get(source_id)
        if raw_row is None:
            if len(sft_missing_source_ids) < sample_limit:
                sft_missing_source_ids.append(source_id)
            continue

        pair_type = row.get("pair_type")
        if pair_type == "post":
            sft_post_total += 1
            if (row.get("output") or "") == (raw_row.get("content") or ""):
                sft_post_exact_match += 1
            continue

        if pair_type != "comment":
            continue

        sft_comment_total += 1
        comments: list[tuple[str | None, str]] = []
        key_to_text: dict[str, str] = {}
        for comment in raw_row.get("comments") or []:
            if not isinstance(comment, dict) or not comment.get("content"):
                continue
            key, body = parse_comment_text(comment["content"])
            comments.append((key, body))
            if key is not None:
                key_to_text[key] = body

        normalized_comment_texts = [body for _, body in comments]
        instruction = normalize_comment_text((row.get("instruction") or "").split(": ", 1)[-1])
        output = normalize_comment_text(row.get("output") or "")
        instruction_ok = instruction in normalized_comment_texts
        output_ok = output in normalized_comment_texts
        if instruction_ok:
            sft_comment_instruction_match += 1
        if output_ok:
            sft_comment_output_match += 1

        reply_pair_ok = False
        for key, body in comments:
            if body != output:
                continue
            parent = parent_key(key)
            if parent and key_to_text.get(parent) == instruction:
                reply_pair_ok = True
                break

        if reply_pair_ok:
            sft_comment_reply_pair_match += 1
        elif len(sft_comment_unmatched) < sample_limit:
            sft_comment_unmatched.append(
                {
                    "source_id": source_id,
                    "instruction": instruction,
                    "output": output,
                    "comment_samples": normalized_comment_texts[:6],
                }
            )

    def rate(matched: int, total: int) -> float:
        if total == 0:
            return 0.0
        return round(matched / total, 4)

    return {
        "raw": {
            "raw_dir": str(raw_dir),
            "post_count": len(raw_post_texts),
            "comment_count": raw_comment_count,
            "source_id_count": len(raw_by_id),
        },
        "cpt": {
            "path": str(cpt_path),
            "row_count": len(cpt_rows),
            "kinds": dict(cpt_kind_counter),
            "post_exact_match": {
                "matched": cpt_post_exact_match,
                "total": cpt_post_total,
                "rate": rate(cpt_post_exact_match, cpt_post_total),
            },
            "comment_normalized_match": {
                "matched": cpt_comment_normalized_match,
                "total": cpt_comment_total,
                "rate": rate(cpt_comment_normalized_match, cpt_comment_total),
            },
            "comment_unmatched_examples": cpt_comment_unmatched,
        },
        "sft": {
            "path": str(sft_path),
            "row_count": len(sft_rows),
            "pair_types": dict(sft_pair_counter),
            "post_exact_match": {
                "matched": sft_post_exact_match,
                "total": sft_post_total,
                "rate": rate(sft_post_exact_match, sft_post_total),
            },
            "comment_instruction_match": {
                "matched": sft_comment_instruction_match,
                "total": sft_comment_total,
                "rate": rate(sft_comment_instruction_match, sft_comment_total),
            },
            "comment_output_match": {
                "matched": sft_comment_output_match,
                "total": sft_comment_total,
                "rate": rate(sft_comment_output_match, sft_comment_total),
            },
            "comment_reply_pair_match": {
                "matched": sft_comment_reply_pair_match,
                "total": sft_comment_total,
                "rate": rate(sft_comment_reply_pair_match, sft_comment_total),
            },
            "missing_source_id_examples": sft_missing_source_ids,
            "comment_unmatched_examples": sft_comment_unmatched,
        },
    }


def main() -> None:
    args = parse_args()
    report = validate(
        raw_dir=Path(args.raw_dir),
        cpt_path=Path(args.cpt_path),
        sft_path=Path(args.sft_path),
        sample_limit=args.sample_limit,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
