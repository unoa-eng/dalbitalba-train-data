#!/usr/bin/env python3
"""
phase6_eval.py — Deterministic 5-metric evaluator for the autonomous loop.

Inputs:
  AI_JSONL   : {"text": "..."} per line — generated samples from current adapter
  RAW_JSONL  : {"text": "..."} per line — held-out raw crawl samples (val_set)

Output (stdout + --out path):
  JSON report with per-metric values and the gate verdict.

Metrics (all deterministic, no LLM judge):
  1. bigram_jsd       — char-level bigram distribution Jensen-Shannon divergence
  2. length_kl        — token-count histogram KL divergence (16 log-spaced bins)
  3. digit_density_Δ  — mean absolute delta in per-sample digit ratio
  4. english_density_Δ — mean absolute delta in per-sample english letter ratio
  5. mauve_score      — optional, requires `mauve-text` + klue/roberta encoder (GPU)

Gate thresholds (from .planning/calibration/raw-vs-raw.json):
  bigram_jsd      ≤ 0.15   (stretch 0.08, baseline 0.019)
  length_kl       ≤ 0.10
  digit_Δ         ≤ 0.03
  english_Δ       ≤ 0.02
  mauve           ≥ 0.80   (if available)

Exit codes:
  0 — PASS (all gates met)
  2 — FAIL (≥1 gate violated) — signals ORPO needed
  3 — ERROR (inputs missing or invalid)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def load_texts(path: str) -> list[str]:
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("text") or obj.get("generated") or obj.get("output") or "").strip()
            if text:
                texts.append(text)
    return texts


# ─── Metric 1: char bigram JSD ────────────────────────────────────────
def char_bigrams(text: str) -> Counter:
    return Counter(text[i : i + 2] for i in range(len(text) - 1))


def aggregate_bigrams(texts: list[str]) -> Counter:
    total: Counter = Counter()
    for t in texts:
        total.update(char_bigrams(t))
    return total


def normalize(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()} if total else {}


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for k in a:
            ak = a[k]
            bk = b.get(k, 0.0)
            if ak > 0 and bk > 0:
                total += ak * math.log2(ak / bk)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ─── Metric 2: length histogram KL ────────────────────────────────────
LENGTH_BINS = [0, 10, 20, 35, 60, 100, 160, 260, 400, 600, 900, 1400, 2200]


def length_histogram(texts: list[str]) -> list[float]:
    counts = [0] * (len(LENGTH_BINS) + 1)
    for t in texts:
        n = len(t)
        placed = False
        for i, upper in enumerate(LENGTH_BINS):
            if n <= upper:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * len(counts)


def kl_divergence(p: list[float], q: list[float]) -> float:
    eps = 1e-12
    return sum(
        pi * math.log2((pi + eps) / (qi + eps)) for pi, qi in zip(p, q) if pi > 0
    )


# ─── Metric 3/4: digit / english density delta ────────────────────────
def density(text: str, predicate) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if predicate(c)) / len(text)


def digit_density(text: str) -> float:
    return density(text, str.isdigit)


def english_density(text: str) -> float:
    return density(text, lambda c: ("a" <= c <= "z") or ("A" <= c <= "Z"))


def mean_abs_delta(texts_a: list[str], texts_b: list[str], fn) -> float:
    a_mean = sum(fn(t) for t in texts_a) / len(texts_a) if texts_a else 0.0
    b_mean = sum(fn(t) for t in texts_b) / len(texts_b) if texts_b else 0.0
    return abs(a_mean - b_mean)


# ─── Metric 5: MAUVE (optional, GPU) ──────────────────────────────────
def maybe_mauve(
    ai_texts: list[str], raw_texts: list[str], encoder: str = "klue/roberta-large"
) -> float | None:
    try:
        import mauve  # type: ignore
    except Exception:
        return None
    try:
        out = mauve.compute_mauve(
            p_text=raw_texts,
            q_text=ai_texts,
            featurize_model_name=encoder,
            device_id=0,
            max_text_length=512,
            verbose=False,
            batch_size=8,
        )
        return float(out.mauve)
    except Exception as exc:
        sys.stderr.write(f"[mauve] skip: {exc}\n")
        return None


# ─── Gate ─────────────────────────────────────────────────────────────
GATE = {
    "bigram_jsd": ("le", 0.15),
    "length_kl": ("le", 0.10),
    "digit_density_delta": ("le", 0.03),
    "english_density_delta": ("le", 0.02),
    "mauve_score": ("ge", 0.80),
}


def evaluate_gate(metrics: dict[str, float | None]) -> tuple[str, list[str]]:
    violations: list[str] = []
    for key, (op, threshold) in GATE.items():
        value = metrics.get(key)
        if value is None:
            continue
        if op == "le" and value > threshold:
            violations.append(f"{key}={value:.4f} > {threshold}")
        elif op == "ge" and value < threshold:
            violations.append(f"{key}={value:.4f} < {threshold}")
    return ("PASS" if not violations else "FAIL"), violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", required=True, help="AI-generated samples JSONL")
    parser.add_argument("--raw", required=True, help="Raw held-out samples JSONL")
    parser.add_argument("--out", help="Write JSON report to this path")
    parser.add_argument("--skip-mauve", action="store_true")
    args = parser.parse_args()

    ai = load_texts(args.ai)
    raw = load_texts(args.raw)
    if not ai or not raw:
        sys.stderr.write(
            f"[error] empty inputs: ai={len(ai)} raw={len(raw)}\n"
        )
        return 3

    p = normalize(aggregate_bigrams(ai))
    q = normalize(aggregate_bigrams(raw))
    bigram = jsd(p, q)

    len_kl = kl_divergence(length_histogram(ai), length_histogram(raw))
    digit_delta = mean_abs_delta(ai, raw, digit_density)
    english_delta = mean_abs_delta(ai, raw, english_density)

    mauve_score: float | None = None
    if not args.skip_mauve:
        mauve_score = maybe_mauve(ai, raw)

    metrics = {
        "n_ai": len(ai),
        "n_raw": len(raw),
        "bigram_jsd": bigram,
        "length_kl": len_kl,
        "digit_density_delta": digit_delta,
        "english_density_delta": english_delta,
        "mauve_score": mauve_score,
    }
    verdict, violations = evaluate_gate(metrics)
    report = {
        "metrics": metrics,
        "gate": {"verdict": verdict, "violations": violations},
        "thresholds": {k: {"op": op, "value": v} for k, (op, v) in GATE.items()},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
