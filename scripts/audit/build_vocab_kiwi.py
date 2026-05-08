"""Stage 1 — Kiwi 형태소 분석 기반 도메인 vocab 추출.

기존 char-n-gram 방식은 띄어쓰기 부재 코퍼스에서 cross-word fragment 양산.
Kiwi (한국어 형태소 분석기, 순수 Python)로 단어 경계 인식 후:
1. NNG/NNP/SL/SH (명사/외래어/한자) 토큰만 추출
2. freq >= MIN_FREQ
3. Qwen3 BPE가 분해(>= 2 토큰)하는 후보 우선
4. seed cooc 가중
5. top N 출력

Output: runs/audit/vocab_kiwi.json
"""
import json
import re
from collections import Counter
from math import log
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "cpt_corpus.v3.jsonl"
OUT = REPO / "runs/audit/vocab_kiwi.json"

MIN_FREQ = 20
TOP_N = 250
ALLOWED_TAGS = {"NNG", "NNP", "SL", "SH", "NR"}

CORE_DOMAIN_TERMS = [
    "쩜오", "텐카", "호빠", "도파민", "하이퍼", "퍼펙트", "마담", "TC", "티씨",
    "빠꾸", "밀빵", "풀묶", "팁", "초이스", "케어", "갯수", "강남", "역삼", "선릉", "논현",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("[1/4] loading Kiwi...")
    from kiwipiepy import Kiwi
    kiwi = Kiwi()

    print("[2/4] tokenizing corpus (this takes ~3-5 min)...")
    freq: Counter[str] = Counter()
    seed_cooc: Counter[str] = Counter()
    seed_pat = re.compile("|".join(re.escape(t) for t in CORE_DOMAIN_TERMS))
    n = 0
    with SRC.open() as f:
        for ln in f:
            n += 1
            if n % 5000 == 0:
                print(f"  ... {n} docs")
            d = json.loads(ln)
            text = d.get("text", "")
            in_seed_doc = bool(seed_pat.search(text))
            try:
                toks = kiwi.tokenize(text)
            except Exception:
                continue
            local = set()
            for t in toks:
                if t.tag in ALLOWED_TAGS and len(t.form) >= 2:
                    local.add(t.form)
            for w in local:
                freq[w] += 1
                if in_seed_doc:
                    seed_cooc[w] += 1
    print(f"  total_docs={n}, unique_nouns={len(freq)}")

    candidates = {w: f for w, f in freq.items() if f >= MIN_FREQ}
    print(f"[3/4] filter freq>={MIN_FREQ}: {len(candidates)}")

    print("[4/4] BPE check + scoring...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")

    rows = []
    for w, f in candidates.items():
        bpe = len(tok.encode(w, add_special_tokens=False))
        if bpe <= 1:
            continue   # already efficient — skip
        cooc = seed_cooc.get(w, 0)
        cooc_ratio = cooc / max(f, 1)
        score = log(1 + f) * (bpe - 1) * (1 + 2 * cooc_ratio)
        rows.append({
            "term": w, "freq": f, "bpe_tokens": bpe,
            "seed_cooc": cooc, "cooc_ratio": cooc_ratio, "score": score,
        })

    rows.sort(key=lambda r: -r["score"])
    print(f"  fragmented_candidates={len(rows)}")

    # seed prepend
    seen = {r["term"] for r in rows}
    head = []
    for t in CORE_DOMAIN_TERMS:
        if t in seen:
            for r in rows:
                if r["term"] == t:
                    head.append({**r, "seed": True}); break
        else:
            # CORE term이 코퍼스에 거의 없거나 Kiwi가 못 잡은 경우 — 강제 추가
            head.append({"term": t, "freq": 0, "bpe_tokens": -1, "score": -1, "seed": True})

    rest = [r for r in rows if r["term"] not in {h["term"] for h in head}]
    final = head + rest[: max(0, TOP_N - len(head))]

    OUT.write_text(json.dumps({
        "min_freq": MIN_FREQ,
        "target_n": TOP_N,
        "n_unique_nouns": len(freq),
        "n_after_freq_filter": len(candidates),
        "n_fragmented": len(rows),
        "n_final": len(final),
        "candidates": final,
    }, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT.name}, n={len(final)}")
    print("\n=== Top 40 ===")
    for c in final[:40]:
        seed = "★" if c.get("seed") else " "
        print(f"  {seed} {c['term']:10s} freq={c['freq']:>5d} bpe={c.get('bpe_tokens')}")


if __name__ == "__main__":
    main()
