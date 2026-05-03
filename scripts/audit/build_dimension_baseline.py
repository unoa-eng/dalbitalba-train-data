"""Stage 0 — Compute raw-corpus baseline per V4 dimension.

Aggregates lexical / topical / stylistic / structural stats from cpt_corpus.v3.
Used as ground-truth distribution for downstream training / eval comparison.
"""
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "cpt_corpus.v3.jsonl"
OUT = REPO / "runs/audit/dimension_baseline.json"

CORE_DOMAIN_TERMS = [
    "쩜오", "텐카", "호빠", "도파민", "하이퍼", "퍼펙트", "마담", "TC", "티씨",
    "빠꾸", "밀빵", "풀묶", "팁", "초이스", "케어", "갯수", "강남", "역삼", "선릉", "논현",
]
SLANG_PATTERNS = {
    "ㅋ_streak": re.compile(r"ㅋ{2,}"),
    "ㅠ_streak": re.compile(r"ㅠ{2,}"),
    "ㅎ_streak": re.compile(r"ㅎ{2,}"),
    "ㅈㄴ_변형": re.compile(r"ㅈ\s?ㄴ"),
    "초성_general": re.compile(r"[ㄱ-ㅎ]{2,}"),
}
FORMAL_MARKERS = re.compile(r"(습니다|합니다|입니다|되었습니다|드립니다)")
INFORMAL_MARKERS = re.compile(r"(임\b|음\b|함\b|는듯|얌|용\b|영\b|네\b)")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    kind = Counter()
    length_bucket = Counter()
    domain_term_counts: Counter[str] = Counter()
    domain_term_docs: Counter[str] = Counter()
    slang_total = Counter()
    formal_docs = 0
    informal_docs = 0
    total_chars = 0
    char_lengths = []
    sentence_endings = Counter()

    for ln in SRC.open():
        d = json.loads(ln)
        n += 1
        text = d.get("text", "")
        total_chars += len(text)
        char_lengths.append(len(text))
        kind[d.get("kind", "?")] += 1
        length_bucket[d.get("length_bucket", "?")] += 1

        # domain terms
        for term in CORE_DOMAIN_TERMS:
            cnt = text.count(term)
            if cnt:
                domain_term_counts[term] += cnt
                domain_term_docs[term] += 1

        # slang
        for tag, pat in SLANG_PATTERNS.items():
            slang_total[tag] += len(pat.findall(text))

        # formality
        if FORMAL_MARKERS.search(text):
            formal_docs += 1
        if INFORMAL_MARKERS.search(text):
            informal_docs += 1

        # endings
        if text.endswith("?"):
            sentence_endings["question"] += 1
        elif text.endswith("ㅋㅋ") or text.endswith("ㅋㅋㅋ"):
            sentence_endings["lol"] += 1
        elif text.endswith("ㅠ") or text.endswith("ㅠㅠ"):
            sentence_endings["sad"] += 1

    char_lengths.sort()
    median = char_lengths[len(char_lengths)//2] if char_lengths else 0
    p90 = char_lengths[int(len(char_lengths)*0.9)] if char_lengths else 0

    out = {
        "corpus": str(SRC.name),
        "total_docs": n,
        "kind": dict(kind),
        "length_bucket": dict(length_bucket),
        "total_chars": total_chars,
        "char_length": {"avg": total_chars/n, "median": median, "p90": p90},
        "domain_terms": {
            term: {
                "total_occ": domain_term_counts[term],
                "doc_freq": domain_term_docs[term],
                "doc_ratio": domain_term_docs[term] / n,
            }
            for term in CORE_DOMAIN_TERMS
        },
        "slang": dict(slang_total),
        "formal_docs": formal_docs,
        "informal_docs": informal_docs,
        "formal_ratio": formal_docs / n,
        "informal_ratio": informal_docs / n,
        "sentence_endings": dict(sentence_endings),
    }
    with OUT.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[dimension_baseline] wrote {OUT.name}, total_docs={n}")


if __name__ == "__main__":
    main()
