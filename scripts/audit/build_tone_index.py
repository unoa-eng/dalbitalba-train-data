"""Stage 0 — Heuristic tone labeling for cpt_corpus.

4-class: 격식 / 일상 / 감정적 / 유머 (heuristic).
"""
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "cpt_corpus.v3.jsonl"
OUT = REPO / "runs/audit/tone_labels.jsonl"
SUMMARY = REPO / "runs/audit/tone_summary.json"

FORMAL = re.compile(r"(습니다|합니다|입니다|드립니다|드려요|되었습니다)")
LOL = re.compile(r"ㅋ{2,}|ㅎ{2,}|개꿀|레알|존맛")
SAD = re.compile(r"ㅠ{2,}|힘들|우울|아프|상처|불안")
SLANG = re.compile(r"ㅈㄴ|존나|ㅅㅂ|존맛|개꿀|레알|ㄹㅇ")


def classify(text: str) -> str:
    if FORMAL.search(text):
        return "격식"
    if LOL.search(text) or SLANG.search(text):
        return "유머"
    if SAD.search(text):
        return "감정적"
    return "일상"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    n = 0
    with OUT.open("w") as f:
        for ln in SRC.open():
            d = json.loads(ln)
            n += 1
            t = classify(d.get("text", ""))
            counts[t] += 1
            f.write(json.dumps({"source_id": d.get("source_id"), "tone": t}, ensure_ascii=False) + "\n")

    sm = {
        "total_docs": n,
        "tone_dist": {k: v/n for k, v in counts.items()},
        "tone_count": dict(counts),
    }
    SUMMARY.write_text(json.dumps(sm, ensure_ascii=False, indent=2))
    print(f"[tone_index] {n} docs")
    for t, c in counts.most_common():
        print(f"  {t}: {c} ({c/n:.1%})")


if __name__ == "__main__":
    main()
