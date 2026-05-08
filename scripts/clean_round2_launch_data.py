#!/usr/bin/env python3
"""Clean active round2 launch JSONL files in-place.

The filter removes deleted/blocked rows, contact/promo templates, and duplicate
canonical text before a paid research run. It writes a JSON report under runs/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs" / "round2-launch-clean-report.json"

TARGETS = [
    "cpt_enriched.jsonl",
    "cpt_corpus.v3.jsonl",
    "sft_thread_conditioned.jsonl",
    "val_set.v2.jsonl",
]

BLOCKED_RE = re.compile(
    r"삭제된\s*댓글|삭제된\s*글|신고에\s*의해|블라인드\s*처리|광고\s*권한이\s*없|"
    r"관리자에\s*의해|운영자에\s*의해"
)
PHONE_RE = re.compile(r"(?:\[전화번호\]|\b01[016789][- .]?\d{3,4}[- .]?\d{4}\b|\+?82[- .]?10[- .]?\d{3,4}[- .]?\d{4})")
PROMO_RE = re.compile(
    r"오픈채팅|카카오톡|카톡|텔레그램|문의\s*(?:주세요|가능|환영)|예약\s*문의|"
    r"영업\s*시간|운영\s*시간|이벤트\s*적용|렌탈\s*(?:샵|저녁|문의)|"
    r"헤메\s*받|메이크업\s*베이스\s*포함|붙임머리|구두\s*1콩|가방,\s*벨트"
)
URL_RE = re.compile(r"https?://|www\.|open\.kakao|t\.me/", re.IGNORECASE)
REPLY_PREFIX_RE = re.compile(r"^\s*\[\d+(?:-\d+)*\]\s*")


def row_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("text", "instruction", "input", "output", "prompt", "chosen", "rejected"):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def canonical(text: str) -> str:
    text = REPLY_PREFIX_RE.sub("", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = URL_RE.sub("[URL]", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def is_bad(text: str) -> str | None:
    if BLOCKED_RE.search(text):
        return "blocked_deleted"
    if PHONE_RE.search(text) or URL_RE.search(text):
        return "contact_or_url"
    if PROMO_RE.search(text):
        return "promo_template"
    return None


def clean_file(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    report = {"input": 0, "kept": 0, "removed": {}, "deduped": 0}
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        report["input"] += 1
        row = json.loads(line)
        text = row_text(row)
        reason = is_bad(text)
        if reason:
            report["removed"][reason] = report["removed"].get(reason, 0) + 1
            continue
        key = canonical(text)
        if key in seen:
            report["deduped"] += 1
            continue
        seen.add(key)
        rows.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    report["kept"] = len(rows)
    return report


def clean_orpo(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    report = {"input": 0, "kept": 0, "removed": {}, "deduped": 0}
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        report["input"] += 1
        row = json.loads(line)
        reason = is_bad(str(row.get("chosen") or "")) or is_bad(str(row.get("rejected") or ""))
        if reason:
            report["removed"][reason] = report["removed"].get(reason, 0) + 1
            continue
        key = canonical(str(row.get("prompt") or "") + "\n" + str(row.get("chosen") or ""))
        if key in seen:
            report["deduped"] += 1
            continue
        seen.add(key)
        rows.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    report["kept"] = len(rows)
    return report


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for rel in TARGETS:
        payload[rel] = clean_file(ROOT / rel)
    payload["orpo_pairs.jsonl"] = clean_orpo(ROOT / "orpo_pairs.jsonl")
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
