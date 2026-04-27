#!/usr/bin/env python3
"""
Phase 0 — Calibration: measure raw-vs-raw baseline (achievability ceiling).

Splits the raw crawl into two disjoint halves by thread/author and computes
distributional distances between the halves. This is the upper bound on how
close an AI generation can ever be to the raw distribution: no model can
beat raw text's self-similarity.

Metrics (stdlib only; MAUVE is deferred to GPU pod):
  - char unigram JSD
  - char bigram JSD
  - length histogram KL divergence (xs/sm/md/lg/xl/xxl buckets)
  - digit / english / korean / whitespace density deltas
  - jamo-run + repeat-punct per record deltas

Output: .planning/calibration/raw-vs-raw.json

Usage:
  python3 scripts/phase0_calibration.py \\
      --raw-dir /mnt/c/Users/mapdr/Desktop/queenalba-crawler/crawled-data-v2 \\
      --out .planning/calibration/raw-vs-raw.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

HANGUL_RE = re.compile(r"[가-힣]")
JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
DIGIT_RE = re.compile(r"[0-9]")
ENGLISH_RE = re.compile(r"[A-Za-z]")
WS_RE = re.compile(r"\s")
JAMO_RUN_RE = re.compile(r"([ㄱ-ㅎㅏ-ㅣ])\1{1,}")
REPEAT_PUNCT_RE = re.compile(r"([?!.~])\1{1,}")

LENGTH_BUCKETS = [
    ("xs", 0, 19),
    ("sm", 20, 49),
    ("md", 50, 99),
    ("lg", 100, 199),
    ("xl", 200, 499),
    ("xxl", 500, 10**9),
]


def length_bucket(n: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return name
    return "unk"


def stable_hash(s: str) -> int:
    # deterministic, cross-platform. SHA-256 -> 64-bit int
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def iter_raw(raw_dir: str):
    """Yield dicts {text, kind, author_key, thread_key}."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        arr = obj if isinstance(obj, list) else (
            obj.get("posts") or obj.get("items") or []
        )
        if not isinstance(arr, list):
            continue
        for post in arr:
            if not isinstance(post, dict):
                continue
            post_id = str(post.get("id") or post.get("post_id") or "")
            author = str(post.get("author") or "").strip() or "anon"
            thread_key = post_id or stable_hash(
                (post.get("title") or "") + "|" + (post.get("content") or "")
            )
            title = (post.get("title") or "").strip()
            body = (post.get("content") or "").strip()
            combined = (title + "\n" + body).strip()
            if combined:
                yield {
                    "text": combined,
                    "kind": "post",
                    "author_key": author,
                    "thread_key": str(thread_key),
                }
            for comment in post.get("comments") or []:
                if not isinstance(comment, dict):
                    continue
                text = (comment.get("content") or "").strip()
                if not text:
                    continue
                c_author = str(comment.get("author") or "").strip() or "anon"
                yield {
                    "text": text,
                    "kind": "comment",
                    "author_key": c_author,
                    "thread_key": str(thread_key),
                }


def split_by_thread(records: list[dict], seed: int = 42):
    """Deterministic A/B split by thread_key hash. All records of the same
    thread go to the same half (no leakage)."""
    side_a: list[dict] = []
    side_b: list[dict] = []
    for rec in records:
        key = rec["thread_key"] + f":{seed}"
        if stable_hash(key) % 2 == 0:
            side_a.append(rec)
        else:
            side_b.append(rec)
    return side_a, side_b


def compute_profile(records: list[dict]) -> dict:
    """Compact distributional profile of a corpus half."""
    n = 0
    total_chars = 0
    length_hist: Counter = Counter()
    class_counts: Counter = Counter(
        {"hangul": 0, "jamo": 0, "digit": 0, "english": 0, "ws": 0, "other": 0}
    )
    jamo_runs_total = 0
    repeat_punct_total = 0
    char_unigram: Counter = Counter()
    char_bigram: Counter = Counter()

    for rec in records:
        text = rec["text"]
        if not text:
            continue
        n += 1
        total_chars += len(text)
        length_hist[length_bucket(len(text))] += 1
        hangul = len(HANGUL_RE.findall(text))
        jamo = len(JAMO_RE.findall(text))
        digit = len(DIGIT_RE.findall(text))
        english = len(ENGLISH_RE.findall(text))
        ws = len(WS_RE.findall(text))
        other = len(text) - (hangul + jamo + digit + english + ws)
        class_counts["hangul"] += hangul
        class_counts["jamo"] += jamo
        class_counts["digit"] += digit
        class_counts["english"] += english
        class_counts["ws"] += ws
        class_counts["other"] += other
        jamo_runs_total += len(JAMO_RUN_RE.findall(text))
        repeat_punct_total += len(REPEAT_PUNCT_RE.findall(text))
        for ch in text:
            char_unigram[ch] += 1
        for i in range(len(text) - 1):
            char_bigram[text[i] + text[i + 1]] += 1

    density = {
        k: (v / total_chars) if total_chars else 0.0 for k, v in class_counts.items()
    }
    length_hist_pct = {k: (v / n) if n else 0.0 for k, v in length_hist.items()}
    return {
        "n_records": n,
        "total_chars": total_chars,
        "avg_chars": (total_chars / n) if n else 0.0,
        "length_hist": length_hist_pct,
        "char_class_density": density,
        "jamo_runs_per_record": (jamo_runs_total / n) if n else 0.0,
        "repeat_punct_per_record": (repeat_punct_total / n) if n else 0.0,
        "unigram_total": sum(char_unigram.values()),
        "bigram_total": sum(char_bigram.values()),
        "_unigram": char_unigram,
        "_bigram": char_bigram,
    }


def js_div(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence in bits."""
    keys = set(p) | set(q)
    p_sum = sum(p.values()) or 1
    q_sum = sum(q.values()) or 1
    jsd = 0.0
    for k in keys:
        pk = p.get(k, 0) / p_sum
        qk = q.get(k, 0) / q_sum
        m = 0.5 * (pk + qk)
        if pk > 0:
            jsd += 0.5 * pk * math.log2(pk / m)
        if qk > 0:
            jsd += 0.5 * qk * math.log2(qk / m)
    return jsd


def kl_hist(p_hist: dict, q_hist: dict, eps: float = 1e-6) -> float:
    keys = set(p_hist) | set(q_hist)
    kl = 0.0
    for k in keys:
        pk = p_hist.get(k, 0.0) + eps
        qk = q_hist.get(k, 0.0) + eps
        kl += pk * math.log2(pk / qk)
    return kl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="/mnt/c/Users/mapdr/Desktop/queenalba-crawler/crawled-data-v2",
    )
    parser.add_argument(
        "--out", default=".planning/calibration/raw-vs-raw.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[phase0] reading raw crawl from: {args.raw_dir}")
    records = list(iter_raw(args.raw_dir))
    print(f"[phase0] total raw records: {len(records):,}")
    threads = {r["thread_key"] for r in records}
    authors = {r["author_key"] for r in records}
    print(f"[phase0] unique threads: {len(threads):,}")
    print(f"[phase0] unique authors: {len(authors):,}")

    print(f"[phase0] splitting by thread (seed={args.seed}) …")
    side_a, side_b = split_by_thread(records, seed=args.seed)
    print(f"[phase0]   side A: {len(side_a):,}")
    print(f"[phase0]   side B: {len(side_b):,}")

    print("[phase0] profiling side A …")
    prof_a = compute_profile(side_a)
    print("[phase0] profiling side B …")
    prof_b = compute_profile(side_b)

    uni_jsd = js_div(prof_a["_unigram"], prof_b["_unigram"])
    bi_jsd = js_div(prof_a["_bigram"], prof_b["_bigram"])
    len_kl_a_b = kl_hist(prof_a["length_hist"], prof_b["length_hist"])
    len_kl_b_a = kl_hist(prof_b["length_hist"], prof_a["length_hist"])

    deltas = {
        "avg_chars_delta": prof_a["avg_chars"] - prof_b["avg_chars"],
        "digit_density_delta": (
            prof_a["char_class_density"]["digit"]
            - prof_b["char_class_density"]["digit"]
        ),
        "english_density_delta": (
            prof_a["char_class_density"]["english"]
            - prof_b["char_class_density"]["english"]
        ),
        "hangul_density_delta": (
            prof_a["char_class_density"]["hangul"]
            - prof_b["char_class_density"]["hangul"]
        ),
        "jamo_runs_delta": (
            prof_a["jamo_runs_per_record"] - prof_b["jamo_runs_per_record"]
        ),
        "repeat_punct_delta": (
            prof_a["repeat_punct_per_record"] - prof_b["repeat_punct_per_record"]
        ),
    }

    # strip non-serializable Counters for output
    def public(p: dict) -> dict:
        return {k: v for k, v in p.items() if not k.startswith("_")}

    result = {
        "meta": {
            "raw_dir": args.raw_dir,
            "seed": args.seed,
            "total_records": len(records),
            "unique_threads": len(threads),
            "unique_authors": len(authors),
        },
        "side_a": public(prof_a),
        "side_b": public(prof_b),
        "baseline": {
            "char_unigram_jsd_bits": uni_jsd,
            "char_bigram_jsd_bits": bi_jsd,
            "length_hist_kl_a_b_bits": len_kl_a_b,
            "length_hist_kl_b_a_bits": len_kl_b_a,
            "length_hist_kl_symmetric": 0.5 * (len_kl_a_b + len_kl_b_a),
            "deltas": deltas,
        },
        "interpretation": {
            "note": (
                "These numbers are the achievability ceiling. "
                "No AI generator can beat raw-vs-raw self-similarity. "
                "Model pass targets should be defined as multiples of these baselines."
            ),
            "recommended_targets": {
                "ai_bigram_jsd_primary": max(0.08, bi_jsd * 1.5),
                "ai_bigram_jsd_stretch": max(0.05, bi_jsd * 1.2),
                "ai_unigram_jsd_primary": max(0.04, uni_jsd * 1.5),
                "ai_length_kl_primary": max(
                    0.10, 0.5 * (len_kl_a_b + len_kl_b_a) * 2.0
                ),
            },
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as h:
        json.dump(result, h, ensure_ascii=False, indent=2)

    print("")
    print("=" * 60)
    print("[phase0] BASELINE SUMMARY")
    print("=" * 60)
    print(f"  records       : {len(records):,}")
    print(f"  threads       : {len(threads):,}")
    print(f"  authors       : {len(authors):,}")
    print(f"  side A / B    : {len(side_a):,} / {len(side_b):,}")
    print(f"  unigram JSD   : {uni_jsd:.4f} bits")
    print(f"  bigram  JSD   : {bi_jsd:.4f} bits")
    print(
        f"  length KL sym : {0.5*(len_kl_a_b+len_kl_b_a):.4f} bits"
    )
    print(f"  avg chars Δ   : {deltas['avg_chars_delta']:+.2f}")
    print(f"  digit  Δ      : {deltas['digit_density_delta']:+.4f}")
    print(f"  english Δ     : {deltas['english_density_delta']:+.4f}")
    print(f"  hangul Δ      : {deltas['hangul_density_delta']:+.4f}")
    print("")
    print("[phase0] recommended AI-vs-raw targets:")
    for k, v in result["interpretation"]["recommended_targets"].items():
        print(f"    {k}: {v:.4f}")
    print("")
    print(f"[phase0] wrote {out_path}")


if __name__ == "__main__":
    main()
