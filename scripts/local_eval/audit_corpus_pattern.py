"""Generic corpus pattern auditor.

Compares the frequency of arbitrary regex patterns across training corpora,
the validation set, and AI-generated samples. Used to investigate whether
suspicious patterns observed in model output (e.g. unexpected ad copy or
domain-foreign text) actually leak from training data, or are model
hallucinations.

USAGE
-----
1. Edit `EXAMPLE_PATTERNS` below for whatever you want to investigate.
2. Edit `FILES` below if your corpus paths differ.
3. Run from the repo root after activating .venv-local:
       python scripts/local_eval/audit_corpus_pattern.py

NOT a quality gate. NOT a training criterion. NOT an ongoing metric.
This is a one-off forensic tool — replace EXAMPLE_PATTERNS for each new
investigation. The current patterns reflect the 2026-05-03 academy-ad
hallucination investigation (sample 5 of D1.5/D2 surfaced 학년도/전원합격
copy that was traced to model hallucination, not data leak).
"""
import json
import re
from pathlib import Path

# Replace these for new investigations.
EXAMPLE_PATTERNS = {
    "학원광고": re.compile(
        r"(전원합격|내신\s*\+?\s*모의고사|학년도|특목고|영재학교|수능\s*\d|"
        r"등록금|모집공고|학원|입시\s*상담|동훈팀)"
    ),
    "공식어": re.compile(
        r"(이용약관|개인정보\s*처리|운영진\s*안내|가입\s*절차|회원가입\s*시)"
    ),
    "대량판매": re.compile(
        r"\d{2,3}만원[\s,~]\d{2,3}만원|VAT\s*별도|카드\s*결제|현금가|최저가\s*판매"
    ),
}

# Adjust paths/limits for your corpus layout.
FILES = [
    ("val_set", "/Users/unoa/projects/dalbitalba-train-data/val_set.v3.jsonl", None),
    ("cpt_corpus", "/Users/unoa/projects/dalbitalba-train-data/cpt_corpus.v3.jsonl", 50000),
    ("cpt_structured", "/Users/unoa/projects/dalbitalba-train-data/v3-data/cpt_structured_v3.jsonl", 50000),
    ("sft_5task", "/Users/unoa/projects/dalbitalba-train-data/v3-data/sft_5task_v3.jsonl", 50000),
    ("sft_thread", "/Users/unoa/projects/dalbitalba-train-data/sft_thread_conditioned.jsonl", 50000),
    ("ai_generated", "/tmp/ai_gen.jsonl", None),
]


def main() -> None:
    pattern_names = list(EXAMPLE_PATTERNS.keys())
    header = " ".join(f"{n:>10s}" for n in pattern_names)
    print(f"{'name':18s} {'lines':>8s} {header}")
    print("-" * (18 + 1 + 8 + 1 + 11 * len(pattern_names)))

    samples: dict[str, list[tuple[str, str]]] = {}
    for name, path, limit in FILES:
        p = Path(path)
        if not p.exists():
            continue
        n = 0
        counts = {k: 0 for k in EXAMPLE_PATTERNS}
        samples[name] = []
        with p.open() as fh:
            for ln in fh:
                n += 1
                if limit and n > limit:
                    break
                try:
                    d = json.loads(ln)
                    text = d.get("text", "") or json.dumps(d, ensure_ascii=False)
                except Exception:
                    text = ln
                for pname, pat in EXAMPLE_PATTERNS.items():
                    m = pat.search(text)
                    if m:
                        counts[pname] += 1
                        if pname == pattern_names[0] and len(samples[name]) < 2:
                            ctx = text[max(0, m.start() - 30): m.end() + 80].replace("\n", " ")
                            samples[name].append((pname, ctx))
        audited = min(n, limit) if limit else n
        cells = []
        for pname in pattern_names:
            c = counts[pname]
            pct = 100.0 * c / audited if audited else 0.0
            cells.append(f"{c:>4d}({pct:4.2f}%)")
        print(f"{name:18s} {audited:>8d} " + " ".join(cells))

    print(f"\n=== Sample matches for '{pattern_names[0]}' ===")
    for name, exs in samples.items():
        if exs:
            print(f"\n[{name}]")
            for tag, snippet in exs[:2]:
                print(f"  ...{snippet[:160]}...")


if __name__ == "__main__":
    main()
