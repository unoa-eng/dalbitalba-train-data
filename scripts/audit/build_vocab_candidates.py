"""Stage 1 (V4) — Identify domain-vocab candidates for tokenizer expansion.

Algorithm:
1. Extract Korean char n-grams (2-5) with freq >= MIN_FREQ from cpt_corpus.
2. For each candidate, tokenize with current Qwen3 tokenizer.
3. BPE inefficiency = (#BPE tokens) for the n-gram. Higher = better candidate.
4. Co-occurrence: how often the candidate appears in docs that also contain
   a CORE_DOMAIN_TERM seed.
5. Score = log(freq) * bpe_inefficiency * (1 + cooc_ratio)
6. Output top N candidates ranked.

Output: runs/audit/vocab_candidates.jsonl
        runs/audit/vocab_candidates_top.json (top 500)
"""
import json
import re
from collections import Counter, defaultdict
from math import log
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "cpt_corpus.v3.jsonl"
OUT_FULL = REPO / "runs/audit/vocab_candidates.jsonl"
OUT_TOP = REPO / "runs/audit/vocab_candidates_top.json"

MIN_FREQ = 50            # n-gram min frequency to consider
MIN_LEN = 2              # min n-gram length (Korean chars)
MAX_LEN = 5              # max n-gram length
TOP_N = 500              # final shortlist size
KOREAN = re.compile(r"[가-힣]+")  # Korean syllable blocks

CORE_DOMAIN_TERMS = [
    "쩜오", "텐카", "호빠", "도파민", "하이퍼", "퍼펙트", "마담", "TC", "티씨",
    "빠꾸", "밀빵", "풀묶", "팁", "초이스", "케어", "갯수", "강남", "역삼", "선릉", "논현",
]

# Common Korean words to exclude (to avoid polluting domain candidates).
COMMON_KO = {
    "그리고", "하지만", "그래서", "그래도", "근데", "오늘", "어제", "내일",
    "지금", "여기", "저기", "거기", "사람", "정말", "진짜", "그냥", "조금",
    "많이", "이번", "다음", "처음", "마지막", "혹시", "물론", "왜냐하면",
    "있어", "없어", "이야", "이지", "이지만", "그런", "이런", "저런",
    "그것", "이것", "저것", "그래", "안녕", "감사",
}


def main() -> None:
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] reading corpus...")
    docs = []
    seed_doc_ids = set()
    seed_pat = re.compile("|".join(re.escape(t) for t in CORE_DOMAIN_TERMS))
    for i, ln in enumerate(SRC.open()):
        d = json.loads(ln)
        text = d.get("text", "")
        docs.append(text)
        if seed_pat.search(text):
            seed_doc_ids.add(i)
    print(f"  total_docs={len(docs)}, seed_containing={len(seed_doc_ids)}")

    print("[2/4] extracting Korean n-grams (2-5)...")
    ngram_freq: Counter[str] = Counter()
    ngram_seed_cooc: Counter[str] = Counter()
    for i, text in enumerate(docs):
        # extract Korean substrings
        local = set()
        for run in KOREAN.findall(text):
            for n in range(MIN_LEN, MAX_LEN + 1):
                for j in range(0, len(run) - n + 1):
                    ng = run[j:j + n]
                    local.add(ng)
        for ng in local:
            ngram_freq[ng] += 1
            if i in seed_doc_ids:
                ngram_seed_cooc[ng] += 1
    # add raw seed terms (in case BPE-friendly)
    for t in CORE_DOMAIN_TERMS:
        ngram_freq[t] = max(ngram_freq.get(t, 0), 1)

    candidates = {ng: f for ng, f in ngram_freq.items()
                  if f >= MIN_FREQ and ng not in COMMON_KO}
    print(f"  raw_ngrams={len(ngram_freq)}, after_filter={len(candidates)}")

    print("[3/4] BPE-tokenizing candidates...")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
    except Exception as exc:
        print(f"  WARN tokenizer load failed: {exc}, falling back to len-based heuristic")
        tok = None

    print("[4/4] scoring + ranking...")
    rows = []
    for ng, freq in candidates.items():
        if tok:
            bpe_count = len(tok.encode(ng, add_special_tokens=False))
        else:
            bpe_count = max(1, len(ng) - 1)  # heuristic
        # only candidates that the BPE actually fragments are interesting
        if bpe_count <= 1:
            continue
        cooc = ngram_seed_cooc.get(ng, 0)
        cooc_ratio = cooc / max(freq, 1)
        score = log(1 + freq) * (bpe_count - 1) * (1 + 2 * cooc_ratio)
        rows.append({
            "term": ng,
            "freq": freq,
            "bpe_tokens": bpe_count,
            "seed_cooc": cooc,
            "cooc_ratio": cooc_ratio,
            "score": score,
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    with OUT_FULL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    top = rows[:TOP_N]
    OUT_TOP.write_text(json.dumps({
        "n_candidates_total": len(rows),
        "top_n": len(top),
        "min_freq_filter": MIN_FREQ,
        "scoring": "log(freq) * (bpe_tokens-1) * (1 + 2*cooc_ratio)",
        "candidates": top,
    }, ensure_ascii=False, indent=2))

    print(f"  total_scored={len(rows)}, top_{TOP_N} written to {OUT_TOP.name}")
    print("\nTop 20 vocab candidates:")
    for r in top[:20]:
        print(f"  {r['term']:8s} freq={r['freq']:>5d} bpe={r['bpe_tokens']} "
              f"cooc={r['cooc_ratio']:.2f} score={r['score']:.2f}")


if __name__ == "__main__":
    main()
