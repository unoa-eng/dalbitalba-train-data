#!/usr/bin/env python3
"""
eval_domain_metrics_v3.py — 다층 도메인 평가 메트릭

TRAINING_DESIGN_V3.md Stage 3 구현:
- Level 1: 분포 메트릭 (Bigram JSD, Length KL)
- Level 2: 스타일 지문 (초성율, 웃음/울음 마커, 물음표, 존댓말, 은어 밀도 등)
- Level 3: Thread coherence (LLM-as-judge 옵션)

Usage:
    python scripts/eval_domain_metrics_v3.py \
        --reference cpt_corpus.v2.jsonl \
        --generated generated_samples.jsonl \
        --output eval_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


# ── Style fingerprint patterns ──────────────────────────────────────────
CHOSUNG_RE = re.compile(r"[ㄱ-ㅎ]{2,}")
LAUGH_RE = re.compile(r"ㅋ{2,}|ㅎ{2,}")
CRY_RE = re.compile(r"[ㅠㅜ]{2,}")
QUESTION_RE = re.compile(r"\?")
ELLIPSIS_RE = re.compile(r"\.{2,}")
EXCLAIM_RE = re.compile(r"!")
JONDAET_RE = re.compile(r"(습니다|입니다|세요|하세요|드려요|에요|는요|거든요|잖아요|인데요|나요|까요)")

DOMAIN_SLANG = [
    "하퍼", "초이스", "밀빵", "쩜오", "수위", "업진", "진상", "텐카",
    "셔츠", "상띠", "퍼블", "손놈", "보도", "가라", "노도", "하띠",
    "뺑이", "골타", "유흥", "실장", "마담", "부장", "TC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="도메인 메트릭 평가 v3")
    parser.add_argument("--reference", required=True, help="원천 데이터 JSONL")
    parser.add_argument("--generated", required=True, help="생성 데이터 JSONL")
    parser.add_argument("--output", help="결과 JSON 저장 경로")
    parser.add_argument("--text-field", default="text", help="텍스트 필드명")
    return parser.parse_args()


def load_texts(path: str, field: str) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(field, "") or row.get("output", "") or row.get("comment", "")
            if text:
                texts.append(text)
    return texts


def compute_style_fingerprint(texts: list[str]) -> dict:
    """텍스트 리스트에서 스타일 지문 메트릭을 계산."""
    n = len(texts)
    if n == 0:
        return {}

    chosung = sum(1 for t in texts if CHOSUNG_RE.search(t))
    laugh = sum(1 for t in texts if LAUGH_RE.search(t))
    cry = sum(1 for t in texts if CRY_RE.search(t))
    question = sum(1 for t in texts if QUESTION_RE.search(t))
    ellipsis = sum(1 for t in texts if ELLIPSIS_RE.search(t))
    exclaim = sum(1 for t in texts if EXCLAIM_RE.search(t))
    jondaet = sum(1 for t in texts if JONDAET_RE.search(t))

    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / n
    lengths.sort()
    median_len = lengths[n // 2]

    slang_counts = {}
    for term in DOMAIN_SLANG:
        cnt = sum(1 for t in texts if term in t)
        slang_counts[term] = round(cnt / n * 100, 2)

    return {
        "count": n,
        "chosung_rate": round(chosung / n * 100, 1),
        "laugh_rate": round(laugh / n * 100, 1),
        "cry_rate": round(cry / n * 100, 1),
        "question_rate": round(question / n * 100, 1),
        "ellipsis_rate": round(ellipsis / n * 100, 1),
        "exclaim_rate": round(exclaim / n * 100, 1),
        "jondaetmal_rate": round(jondaet / n * 100, 1),
        "avg_length": round(avg_len, 1),
        "median_length": median_len,
        "slang_density": slang_counts,
    }


def bigram_distribution(texts: list[str]) -> Counter:
    counter: Counter[str] = Counter()
    for text in texts:
        chars = list(text)
        for i in range(len(chars) - 1):
            counter[chars[i] + chars[i + 1]] += 1
    total = sum(counter.values())
    if total > 0:
        for k in counter:
            counter[k] /= total
    return counter


def jensen_shannon_divergence(p: Counter, q: Counter) -> float:
    all_keys = set(p.keys()) | set(q.keys())
    jsd = 0.0
    for k in all_keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        mk = (pk + qk) / 2
        if pk > 0 and mk > 0:
            jsd += pk * math.log2(pk / mk)
        if qk > 0 and mk > 0:
            jsd += qk * math.log2(qk / mk)
    return jsd / 2


def length_kl_divergence(ref_texts: list[str], gen_texts: list[str], bins: int = 20) -> float:
    ref_lens = [len(t) for t in ref_texts]
    gen_lens = [len(t) for t in gen_texts]
    if not ref_lens or not gen_lens:
        return float("inf")

    max_len = max(max(ref_lens), max(gen_lens))
    bin_size = max(max_len // bins, 1)

    ref_hist = Counter(l // bin_size for l in ref_lens)
    gen_hist = Counter(l // bin_size for l in gen_lens)

    ref_total = sum(ref_hist.values())
    gen_total = sum(gen_hist.values())
    all_bins = set(ref_hist.keys()) | set(gen_hist.keys())

    epsilon = 1e-10
    kl = 0.0
    for b in all_bins:
        p = ref_hist.get(b, 0) / ref_total + epsilon
        q = gen_hist.get(b, 0) / gen_total + epsilon
        kl += p * math.log(p / q)
    return kl


def compare_fingerprints(ref_fp: dict, gen_fp: dict) -> dict:
    """스타일 지문 비교. 각 메트릭의 원천 기준 대비 편차를 계산."""
    metrics = [
        "chosung_rate", "laugh_rate", "cry_rate", "question_rate",
        "ellipsis_rate", "exclaim_rate", "jondaetmal_rate",
        "avg_length", "median_length",
    ]
    comparison = {}
    for m in metrics:
        ref_val = ref_fp.get(m, 0)
        gen_val = gen_fp.get(m, 0)
        if ref_val > 0:
            deviation = round((gen_val - ref_val) / ref_val * 100, 1)
        else:
            deviation = 0.0
        comparison[m] = {
            "reference": ref_val,
            "generated": gen_val,
            "deviation_pct": deviation,
            "pass": abs(deviation) <= 30,  # ±30% 허용
        }

    # Slang density comparison
    ref_slang = ref_fp.get("slang_density", {})
    gen_slang = gen_fp.get("slang_density", {})
    slang_comparison = {}
    for term in DOMAIN_SLANG:
        rv = ref_slang.get(term, 0)
        gv = gen_slang.get(term, 0)
        if rv > 0.1:
            dev = round((gv - rv) / rv * 100, 1)
            slang_comparison[term] = {"ref": rv, "gen": gv, "dev": dev, "pass": abs(dev) <= 50}
    comparison["slang_density"] = slang_comparison

    return comparison


def main() -> None:
    args = parse_args()

    print("Loading texts...")
    ref_texts = load_texts(args.reference, args.text_field)
    gen_texts = load_texts(args.generated, args.text_field)

    print(f"Reference: {len(ref_texts)} texts")
    print(f"Generated: {len(gen_texts)} texts")

    # Level 1: Distribution metrics
    print("\nComputing distribution metrics...")
    ref_bigrams = bigram_distribution(ref_texts)
    gen_bigrams = bigram_distribution(gen_texts)
    jsd = jensen_shannon_divergence(ref_bigrams, gen_bigrams)
    len_kl = length_kl_divergence(ref_texts, gen_texts)

    # Level 2: Style fingerprint
    print("Computing style fingerprints...")
    ref_fp = compute_style_fingerprint(ref_texts)
    gen_fp = compute_style_fingerprint(gen_texts)
    comparison = compare_fingerprints(ref_fp, gen_fp)

    # Count passes
    metric_passes = sum(1 for v in comparison.values() if isinstance(v, dict) and v.get("pass", False))
    metric_total = sum(1 for v in comparison.values() if isinstance(v, dict) and "pass" in v)

    report = {
        "level_1_distribution": {
            "bigram_jsd": round(jsd, 6),
            "bigram_jsd_pass": jsd < 0.08,
            "length_kl": round(len_kl, 6),
            "length_kl_pass": len_kl < 0.01,
        },
        "level_2_style_fingerprint": {
            "reference": ref_fp,
            "generated": gen_fp,
            "comparison": comparison,
            "pass_rate": f"{metric_passes}/{metric_total}",
        },
        "overall_verdict": "PASS" if (jsd < 0.08 and len_kl < 0.01 and metric_passes >= metric_total * 0.8) else "NEEDS_IMPROVEMENT",
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Bigram JSD:  {jsd:.6f}  {'PASS' if jsd < 0.08 else 'FAIL'}")
    print(f"  Length KL:   {len_kl:.6f}  {'PASS' if len_kl < 0.01 else 'FAIL'}")
    print(f"  Style Match: {metric_passes}/{metric_total}")
    print(f"  Verdict:     {report['overall_verdict']}")
    print(f"{'='*60}")

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"\nReport saved to {args.output}")
    else:
        print(f"\n{payload}")


if __name__ == "__main__":
    main()
