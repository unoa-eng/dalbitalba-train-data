"""Stage 1 Post-processing — Curate vocab candidates (v2 — aggressive).

Korean domain nouns (호빠 vocabulary) are almost exclusively 2-3 syllables.
The raw candidate list has many 4-5 syllable cross-word fragments due to
sparse spacing in the corpus. Filter aggressively:

1. Drop all candidates of length >= 5 (almost always cross-word fragments).
2. Drop length-4 candidates unless score >> threshold (rare compounds).
3. Apply BAD_START/BAD_END fragment filter (Korean particle/ending heuristic).
4. Substring dedup (length-aware).
5. Always include CORE_DOMAIN_TERMS verbatim.
6. Cap at TARGET_N.

Output: runs/audit/vocab_curated.json
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "runs/audit/vocab_candidates_top.json"
OUT = REPO / "runs/audit/vocab_curated.json"
TARGET_N = 200
LEN4_SCORE_THRESHOLD = 30.0  # only keep 4-syllable candidates with score above this

CORE_DOMAIN_TERMS = [
    "쩜오", "텐카", "호빠", "도파민", "하이퍼", "퍼펙트", "마담", "TC", "티씨",
    "빠꾸", "밀빵", "풀묶", "팁", "초이스", "케어", "갯수", "강남", "역삼", "선릉", "논현",
]

BAD_END = {"는","은","이","가","을","를","의","도","만","에","께","랑","와","과","면","서","로","고","며","다","지"}
BAD_START = {"은","는","을","를","이","가","의","에","도","고","며","지","다","서","받","주","되"}


def is_fragment(term: str) -> bool:
    L = len(term)
    if L <= 1:
        return True
    if L >= 5:
        return True   # cross-word fragments dominate this length class
    if term[0] in BAD_START:
        return True
    if term[-1] in BAD_END:
        return True
    return False


def main() -> None:
    data = json.loads(SRC.read_text())
    cands = data.get("candidates", [])
    n0 = len(cands)
    print(f"[curate-v2] input: {n0}")

    # 1) hard fragment filter
    keep1 = [c for c in cands if not is_fragment(c["term"])]
    print(f"[curate-v2] after hard filter (len>=5 OR fragment markers): {len(keep1)}")

    # 2) length-4 score gate
    keep2 = [c for c in keep1 if len(c["term"]) <= 3 or c["score"] >= LEN4_SCORE_THRESHOLD]
    print(f"[curate-v2] after len-4 score gate (>={LEN4_SCORE_THRESHOLD}): {len(keep2)}")

    # 3) substring dedup (longer wins if freq similar)
    keep2.sort(key=lambda c: (-len(c["term"]), -c["score"]))
    deduped = []
    for c in keep2:
        t = c["term"]
        f = c["freq"]
        drop = False
        for k in deduped:
            if t != k["term"] and t in k["term"] and f <= k["freq"] * 1.3:
                drop = True; break
        if not drop:
            deduped.append(c)
    print(f"[curate-v2] after substring dedup: {len(deduped)}")

    # 4) Score-sort + seed prepend
    deduped.sort(key=lambda c: -c["score"])
    seen = {c["term"] for c in deduped}
    head = []
    for t in CORE_DOMAIN_TERMS:
        if t in seen:
            for c in deduped:
                if c["term"] == t:
                    head.append({**c, "seed": True}); break
        else:
            head.append({"term": t, "freq": 0, "bpe_tokens": -1, "score": -1, "seed": True})
    rest = [c for c in deduped if c["term"] not in {h["term"] for h in head}]
    final = head + rest[: max(0, TARGET_N - len(head))]

    print(f"[curate-v2] final: {len(final)} (target={TARGET_N}, seeds={len(head)})")

    OUT.write_text(json.dumps({
        "n_input": n0,
        "n_after_hard_filter": len(keep1),
        "n_after_len4_gate": len(keep2),
        "n_after_dedup": len(deduped),
        "n_final": len(final),
        "target_n": TARGET_N,
        "len4_score_threshold": LEN4_SCORE_THRESHOLD,
        "candidates": final,
    }, ensure_ascii=False, indent=2))

    print("\n=== Top 40 curated ===")
    for c in final[:40]:
        seed = "★" if c.get("seed") else " "
        print(f"  {seed} {c['term']:8s} len={len(c['term'])} freq={c['freq']:>5d} bpe={c.get('bpe_tokens',0)}")


if __name__ == "__main__":
    main()
