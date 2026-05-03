"""Stage 0 — Build persona index.

Groups cpt_corpus.v3.jsonl by source_id, ranks by total post count,
flags potential 'veteran' authors (>=50 docs). Output: runs/audit/persona_index.jsonl.
"""
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "cpt_corpus.v3.jsonl"
OUT = REPO / "runs/audit/persona_index.jsonl"

VETERAN_THRESHOLD = 50  # min docs to be considered veteran

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    kind_counts: dict[str, Counter[str]] = {}
    total_chars: dict[str, int] = {}
    for ln in SRC.open():
        d = json.loads(ln)
        sid = str(d.get("source_id", ""))
        if not sid:
            continue
        counts[sid] += 1
        kind = d.get("kind", "?")
        kind_counts.setdefault(sid, Counter())[kind] += 1
        total_chars[sid] = total_chars.get(sid, 0) + len(d.get("text", ""))

    rows = []
    for sid, n in counts.most_common():
        rows.append({
            "source_id": sid,
            "doc_count": n,
            "kind_breakdown": dict(kind_counts[sid]),
            "total_chars": total_chars[sid],
            "avg_chars_per_doc": total_chars[sid] / max(n, 1),
            "veteran": n >= VETERAN_THRESHOLD,
        })

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_veteran = sum(1 for r in rows if r["veteran"])
    print(f"[persona_index] total_authors={len(rows)} veteran(>={VETERAN_THRESHOLD})={n_veteran}")
    print(f"[persona_index] top_5:")
    for r in rows[:5]:
        print(f"  {r['source_id']}: {r['doc_count']} docs, {r['total_chars']} chars")


if __name__ == "__main__":
    main()
