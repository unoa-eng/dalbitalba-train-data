#!/usr/bin/env python3
"""
phase6_eval.py — Deterministic 7-metric evaluator for the autonomous loop.

Inputs:
  AI_JSONL   : {"text": "..."} per line — generated samples from current adapter
  RAW_JSONL  : {"text": "..."} per line — held-out raw crawl samples (val_set)

Output (stdout + --out path):
  JSON report with per-metric values and the gate verdict.
  If both inputs include `kind` metadata, the report also includes
  per-kind (`post` / `comment`) metric breakdowns.

Metrics (all deterministic, no LLM judge):
  1. bigram_jsd       — char-level bigram distribution Jensen-Shannon divergence
  2. length_kl        — token-count histogram KL divergence (16 log-spaced bins)
  3. digit_density_Δ  — mean absolute delta in per-sample digit ratio
  4. english_density_Δ — mean absolute delta in per-sample english letter ratio
  5. domain_keyword_alignment — minimum generated/raw sample-presence ratio across 20 core GAP terms
  6. tone_distribution_match — max absolute delta across 반말/존댓말 sample ratios
  7. mauve_score      — optional, requires `mauve-text` + klue/roberta encoder (GPU)

Gate thresholds (from .planning/calibration/raw-vs-raw.json):
  bigram_jsd      ≤ 0.08   (baseline 0.019)
  length_kl       ≤ 0.10
  digit_Δ         ≤ 0.03
  english_Δ       ≤ 0.02
  domain_keyword_alignment ≥ 0.50
  tone_distribution_match  ≤ 0.15
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
import re
import sys
from collections import Counter
from pathlib import Path


KIND_ORDER = ("post", "comment")


def normalize_kind(value: object) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw == "context_comment":
        return "comment"
    if raw in KIND_ORDER:
        return raw
    if raw in {"reply", "댓글", "답글"}:
        return "comment"
    if raw in {"본문", "원글"}:
        return "post"
    return None


def load_rows(path: str) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
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
                kind = normalize_kind(obj.get("kind") or obj.get("pair_type"))
                rows.append(
                    {
                        "text": text,
                        "kind": kind,
                    }
                )
    return rows


def texts_from_rows(rows: list[dict[str, str | None]]) -> list[str]:
    return [str(row["text"]) for row in rows if row.get("text")]


def texts_for_kind(rows: list[dict[str, str | None]], kind: str) -> list[str]:
    return [str(row["text"]) for row in rows if row.get("kind") == kind and row.get("text")]


def has_kind_metadata(rows: list[dict[str, str | None]]) -> bool:
    return any(row.get("kind") in KIND_ORDER for row in rows)


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


# ─── Metric 5: domain keyword alignment ───────────────────────────────
CORE_DOMAIN_TERMS = [
    "쩜오",
    "텐카",
    "호빠",
    "도파민",
    "하이퍼",
    "퍼펙트",
    "마담",
    "TC",
    "티씨",
    "빠꾸",
    "밀빵",
    "풀묶",
    "팁",
    "초이스",
    "케어",
    "갯수",
    "강남",
    "역삼",
    "선릉",
    "논현",
]


def sample_presence_ratio(texts: list[str], term: str) -> float:
    if not texts:
        return 0.0
    hits = sum(1 for text in texts if term in text)
    return hits / len(texts)


def domain_keyword_alignment(
    ai_texts: list[str], raw_texts: list[str]
) -> tuple[float, dict[str, object]]:
    per_term: dict[str, dict[str, object]] = {}
    ratios: list[float] = []
    aligned_terms = 0
    min_term = CORE_DOMAIN_TERMS[0]
    min_ratio = 1.0

    for term in CORE_DOMAIN_TERMS:
        raw_ratio = sample_presence_ratio(raw_texts, term)
        ai_ratio = sample_presence_ratio(ai_texts, term)
        if raw_ratio == 0.0:
            ratio = 1.0
            source_absent = True
        else:
            ratio = ai_ratio / raw_ratio
            source_absent = False
        ratios.append(ratio)
        if ratio >= 0.5:
            aligned_terms += 1
        if ratio < min_ratio:
            min_ratio = ratio
            min_term = term
        per_term[term] = {
            "generated_ratio": ai_ratio,
            "raw_ratio": raw_ratio,
            "ratio": ratio,
            "aligned": ratio >= 0.5,
            "source_absent": source_absent,
        }

    total_terms = len(CORE_DOMAIN_TERMS)
    return min(ratios) if ratios else 1.0, {
        "aligned_terms": aligned_terms,
        "total_terms": total_terms,
        "aligned_fraction": aligned_terms / total_terms if total_terms else 1.0,
        "minimum_ratio_term": min_term,
        "minimum_ratio": min_ratio,
        "per_term": per_term,
    }


# ─── Metric 6: tone distribution match ────────────────────────────────
_JONDAE_ENDINGS = re.compile(
    r"(합니다|합니까|하세요|하십시오|하시겠|입니다|습니다|습니까|세요|시오|겠습|에요|예요|해요|하죠|죠)\s*[.?!]?\s*$",
    re.MULTILINE,
)
_BANMAL_ENDINGS = re.compile(
    r"(해|했|한다|했다|하지|했지|함|했음|인데|인듯|거든|거임|임|음|ㅋ|ㅎ|ㅠ|ㅜ)\s*[.?!]?\s*$",
    re.MULTILINE,
)


def classify_tone(text: str) -> str:
    jondae = len(_JONDAE_ENDINGS.findall(text))
    banmal = len(_BANMAL_ENDINGS.findall(text))
    total = jondae + banmal
    if total == 0:
        return "혼합"
    if jondae / total > 0.7:
        return "존댓말"
    if banmal / total > 0.7:
        return "반말"
    return "혼합"


def tone_distribution(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {"반말": 0.0, "존댓말": 0.0, "혼합": 0.0}
    counts: Counter[str] = Counter(classify_tone(text) for text in texts)
    total = len(texts)
    return {
        "반말": counts.get("반말", 0) / total,
        "존댓말": counts.get("존댓말", 0) / total,
        "혼합": counts.get("혼합", 0) / total,
    }


def tone_distribution_match(
    ai_texts: list[str], raw_texts: list[str]
) -> tuple[float, dict[str, object]]:
    ai_dist = tone_distribution(ai_texts)
    raw_dist = tone_distribution(raw_texts)
    tracked_tones = ("반말", "존댓말")
    deltas = {
        tone: abs(ai_dist.get(tone, 0.0) - raw_dist.get(tone, 0.0))
        for tone in tracked_tones
    }
    max_delta = max(deltas.values()) if deltas else 0.0
    return max_delta, {
        "generated": ai_dist,
        "raw": raw_dist,
        "absolute_deltas": deltas,
        "within_threshold": all(delta <= 0.15 for delta in deltas.values()),
    }


# ─── Metric 7: MAUVE (optional, GPU) ──────────────────────────────────
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
    "bigram_jsd": ("le", 0.08),
    "length_kl": ("le", 0.10),
    "digit_density_delta": ("le", 0.03),
    "english_density_delta": ("le", 0.02),
    "domain_keyword_alignment": ("ge", 0.50),
    "tone_distribution_match": ("le", 0.15),
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


def compute_metric_bundle(
    ai_texts: list[str],
    raw_texts: list[str],
    *,
    include_mauve: bool,
) -> tuple[dict[str, float | None], dict[str, object]]:
    p = normalize(aggregate_bigrams(ai_texts))
    q = normalize(aggregate_bigrams(raw_texts))
    bigram = jsd(p, q)

    len_kl = kl_divergence(length_histogram(ai_texts), length_histogram(raw_texts))
    digit_delta = mean_abs_delta(ai_texts, raw_texts, digit_density)
    english_delta = mean_abs_delta(ai_texts, raw_texts, english_density)
    domain_alignment, domain_details = domain_keyword_alignment(ai_texts, raw_texts)
    tone_match, tone_details = tone_distribution_match(ai_texts, raw_texts)

    mauve_score: float | None = None
    if include_mauve:
        mauve_score = maybe_mauve(ai_texts, raw_texts)

    metrics = {
        "n_ai": len(ai_texts),
        "n_raw": len(raw_texts),
        "bigram_jsd": bigram,
        "length_kl": len_kl,
        "digit_density_delta": digit_delta,
        "english_density_delta": english_delta,
        "domain_keyword_alignment": domain_alignment,
        "tone_distribution_match": tone_match,
        "mauve_score": mauve_score,
    }
    details = {
        "domain_keyword_alignment": domain_details,
        "tone_distribution_match": tone_details,
    }
    return metrics, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", required=True, help="AI-generated samples JSONL")
    parser.add_argument("--raw", required=True, help="Raw held-out samples JSONL")
    parser.add_argument("--out", help="Write JSON report to this path")
    parser.add_argument("--skip-mauve", action="store_true")
    args = parser.parse_args()

    ai_rows = load_rows(args.ai)
    raw_rows = load_rows(args.raw)
    ai = texts_from_rows(ai_rows)
    raw = texts_from_rows(raw_rows)
    if not ai or not raw:
        sys.stderr.write(
            f"[error] empty inputs: ai={len(ai)} raw={len(raw)}\n"
        )
        return 3

    metrics, details = compute_metric_bundle(ai, raw, include_mauve=not args.skip_mauve)
    verdict, violations = evaluate_gate(metrics)

    kind_breakdown: dict[str, object] = {}
    kind_breakdown_mode = "unavailable"
    if has_kind_metadata(ai_rows) and has_kind_metadata(raw_rows):
        kind_breakdown_mode = "explicit_kind_field"
        for kind in KIND_ORDER:
            ai_kind = texts_for_kind(ai_rows, kind)
            raw_kind = texts_for_kind(raw_rows, kind)
            if not ai_kind or not raw_kind:
                continue
            kind_metrics, kind_details = compute_metric_bundle(
                ai_kind,
                raw_kind,
                include_mauve=False,
            )
            kind_verdict, kind_violations = evaluate_gate(kind_metrics)
            kind_breakdown[kind] = {
                "metrics": kind_metrics,
                "details": kind_details,
                "gate": {"verdict": kind_verdict, "violations": kind_violations},
            }

    details["kind_counts"] = {
        "generated": dict(
            Counter(str(row.get("kind") or "unknown") for row in ai_rows)
        ),
        "raw": dict(
            Counter(str(row.get("kind") or "unknown") for row in raw_rows)
        ),
    }
    details["kind_breakdown_mode"] = kind_breakdown_mode
    details["kind_breakdown"] = kind_breakdown

    report = {
        "metrics": metrics,
        "details": details,
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
