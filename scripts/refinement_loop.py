#!/usr/bin/env python3
"""
refinement_loop.py — Full diagnostic → fix → verify cycle for
source-vs-training GAP analysis in the dalbitalba fine-tuning pipeline.

Usage:
    python scripts/refinement_loop.py --source-dir /path/to/crawled-data-v2 [--fix] [--cycle N]

Stdlib-only. Designed for 16 GB Mac memory budget.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_cpt_path(cli_value: str | None = None) -> Path:
    candidates: list[Path] = []
    raw_candidates = []
    if cli_value:
        raw_candidates.append(cli_value)
    env_value = os.environ.get("TRAIN_CPT_JSONL", "").strip()
    if env_value:
        raw_candidates.append(env_value)
    raw_candidates.extend(
        [
            "cpt_context_stream.jsonl",
            "cpt_corpus.v3.jsonl",
            "cpt_corpus.v2.jsonl",
        ]
    )

    for raw in raw_candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidates.append(candidate)
        if candidate.exists():
            return candidate

    return candidates[0]

TRAIN_FILES = {
    "cpt": resolve_cpt_path(),
    "sft": PROJECT_ROOT / "sft_pairs.v2.jsonl",
}

KIND_ORDER = ("post", "comment")

DOMAIN_TERMS = {
    "업소유형": ["쩜오", "텐카", "호빠", "도파민", "하이퍼", "퍼펙트"],
    "직업용어": ["언니", "실장", "사장", "마담", "TC", "선수"],
    "금전": ["티씨", "빠꾸", "밀빵", "풀묶", "팁"],
    "행위": ["초이스", "케어", "방", "갯수"],
    "지역": ["강남", "역삼", "선릉", "논현"],
    "초성": ["ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅇㅈ", "ㄹㅇ", "ㅈㄹ", "ㅅㅂ"],
    "감정": ["ㅡㅡ", "^^", "ㄷㄷ", "ㅇㅇ"],
}

GAP_LOW = 0.5
GAP_HIGH = 2.0
STRUCTURE_SAMPLE_SIZE = 500

PUNCTUATION_METRICS = {
    "?": ("?", "？"),
    "!": ("!", "！"),
    "...": ("...", "…"),
}

EMOTIVE_MARKERS = {
    "ㅋㅋ": ("ㅋㅋ",),
    "ㅎㅎ": ("ㅎㅎ",),
    "ㅠㅠ": ("ㅠㅠ",),
    "^^": ("^^",),
    "ㅡㅡ": ("ㅡㅡ",),
}

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def count_term(text: str, term: str) -> int:
    """Count non-overlapping occurrences of *term* in *text*."""
    return text.count(term)


def per_million(count: int, total_chars: int) -> float:
    if total_chars == 0:
        return 0.0
    return count / total_chars * 1_000_000


def classify_severity(ratio: float) -> str:
    if ratio <= 0.1 or ratio >= 10.0:
        return "critical"
    if ratio < GAP_LOW or ratio > GAP_HIGH:
        return "moderate"
    return "ok"


def init_kind_buckets():
    return {
        kind: {"texts": [], "source_ids": set(), "text_count": 0}
        for kind in KIND_ORDER
    }


def add_corpus_text(
    *,
    texts: list[str],
    source_ids: set[str],
    kind_buckets: dict[str, dict],
    kind_counts: Counter,
    text: str,
    kind: str,
    source_id: str = "",
) -> None:
    if not text:
        return

    texts.append(text)
    if source_id:
        source_ids.add(source_id)

    if kind in kind_buckets:
        bucket = kind_buckets[kind]
        bucket["texts"].append(text)
        if source_id:
            bucket["source_ids"].add(source_id)
        bucket["text_count"] += 1
        kind_counts[kind] += 1


def build_corpus_bundle(
    texts: list[str],
    source_ids: set[str],
    kind_buckets: dict[str, dict],
    kind_counts: Counter,
) -> dict:
    return {
        "concat": "\n".join(texts),
        "texts": texts,
        "source_ids": source_ids,
        "kind_counts": dict(kind_counts),
        "by_kind": {
            kind: {
                "texts": kind_buckets[kind]["texts"],
                "concat": "\n".join(kind_buckets[kind]["texts"]),
                "source_ids": kind_buckets[kind]["source_ids"],
                "text_count": kind_buckets[kind]["text_count"],
            }
            for kind in KIND_ORDER
        },
    }


def count_marker_variants(text: str, variants: tuple[str, ...]) -> int:
    return sum(text.count(variant) for variant in variants)


# ──────────────────────────────────────────────────────────────
# Source data loader (streaming, memory-friendly)
# ──────────────────────────────────────────────────────────────

def iter_source_texts(source_dir: Path):
    """Yield (source_id, kind, text) tuples from crawled JSON files.

    Each file is a list of dicts.  We extract post content and individual
    comment texts separately so the downstream analysis mirrors how
    training data is structured (kind = post | comment).
    """
    for fpath in sorted(source_dir.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                records = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log(f"  WARN: skipping {fpath.name} ({exc})")
            continue

        for rec in records:
            if not isinstance(rec, dict):
                continue
            sid = str(rec.get("id", ""))
            content = rec.get("content", "")
            if content and isinstance(content, str) and len(content.strip()) > 0:
                yield (sid, "post", content)
            for cmt in rec.get("comments", []) or []:
                if not isinstance(cmt, dict):
                    continue
                ctext = cmt.get("content", "")
                if ctext and isinstance(ctext, str) and len(ctext.strip()) > 0:
                    yield (sid, "comment", ctext)


def load_source_concat(source_dir: Path):
    """Return a corpus bundle for raw source texts, stratified by kind."""
    texts = []
    source_ids = set()
    kind_buckets = init_kind_buckets()
    kind_counts: Counter[str] = Counter()

    for sid, kind, text in iter_source_texts(source_dir):
        add_corpus_text(
            texts=texts,
            source_ids=source_ids,
            kind_buckets=kind_buckets,
            kind_counts=kind_counts,
            text=text,
            kind=kind,
            source_id=sid,
        )

    return build_corpus_bundle(texts, source_ids, kind_buckets, kind_counts)


def load_train_concat():
    """Return a corpus bundle for training texts, stratified by kind.

    Reads live CPT v3 rows plus SFT pairs. Validation rows are explicitly
    excluded so diagnostics reflect train-only data.
    """
    texts = []
    source_ids = set()
    kind_buckets = init_kind_buckets()
    kind_counts: Counter[str] = Counter()

    # CPT corpus
    cpt_path = TRAIN_FILES["cpt"]
    if cpt_path.exists():
        with open(cpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                add_corpus_text(
                    texts=texts,
                    source_ids=source_ids,
                    kind_buckets=kind_buckets,
                    kind_counts=kind_counts,
                    text=str(rec.get("text", "") or "").strip(),
                    kind=str(rec.get("kind", "") or "").strip(),
                    source_id=str(rec.get("source_id", "") or "").strip(),
                )

    # SFT pairs
    sft_path = TRAIN_FILES["sft"]
    if sft_path.exists():
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = str(rec.get("source_id", "") or "").strip()
                add_corpus_text(
                    texts=texts,
                    source_ids=source_ids,
                    kind_buckets=kind_buckets,
                    kind_counts=kind_counts,
                    text=str(rec.get("post", "") or "").strip(),
                    kind="post",
                    source_id=sid,
                )
                add_corpus_text(
                    texts=texts,
                    source_ids=source_ids,
                    kind_buckets=kind_buckets,
                    kind_counts=kind_counts,
                    text=str(rec.get("comment", "") or "").strip(),
                    kind="comment",
                    source_id=sid,
                )

                # Some older SFT exports store a single text field plus a kind marker.
                fallback_kind = str(rec.get("kind") or rec.get("pair_type") or "").strip()
                if fallback_kind in KIND_ORDER and not rec.get(fallback_kind):
                    add_corpus_text(
                        texts=texts,
                        source_ids=source_ids,
                        kind_buckets=kind_buckets,
                        kind_counts=kind_counts,
                        text=str(rec.get("text", "") or "").strip(),
                        kind=fallback_kind,
                        source_id=sid,
                    )

    return build_corpus_bundle(texts, source_ids, kind_buckets, kind_counts)


# ──────────────────────────────────────────────────────────────
# Phase 1a: Term frequency comparison
# ──────────────────────────────────────────────────────────────

def build_frequency_entry(
    *,
    kind: str,
    category: str,
    term: str,
    source_count: int,
    train_count: int,
    source_chars: int,
    train_chars: int,
) -> dict:
    src_pm = per_million(source_count, source_chars)
    trn_pm = per_million(train_count, train_chars)
    if src_pm == 0:
        ratio = float("inf") if trn_pm > 0 else 1.0
    else:
        ratio = trn_pm / src_pm

    return {
        "kind": kind,
        "category": category,
        "term": term,
        "source_count": source_count,
        "train_count": train_count,
        "source_per_million": round(src_pm, 2),
        "train_per_million": round(trn_pm, 2),
        "ratio": round(ratio, 4) if ratio != float("inf") else "inf",
        "severity": classify_severity(ratio),
    }


def phase1a_term_freq(src_corpus: dict, trn_corpus: dict):
    """Compare normalized per-million-char term frequency by post/comment kind."""
    log("Phase 1a: Kind-stratified term / punctuation / emotive frequency comparison ...")

    results: dict[str, dict] = {}
    gaps = []

    for kind in KIND_ORDER:
        src_concat = src_corpus["by_kind"][kind]["concat"]
        trn_concat = trn_corpus["by_kind"][kind]["concat"]
        src_len = len(src_concat)
        trn_len = len(trn_concat)
        log(
            f"  {kind}: source chars={src_len:,} texts={len(src_corpus['by_kind'][kind]['texts']):,}  |  "
            f"train chars={trn_len:,} texts={len(trn_corpus['by_kind'][kind]['texts']):,}"
        )

        kind_results = {
            "source_chars": src_len,
            "train_chars": trn_len,
            "domain_terms": {},
            "punctuation": {},
            "emotive_markers": {},
        }

        for category, terms in DOMAIN_TERMS.items():
            for term in terms:
                entry = build_frequency_entry(
                    kind=kind,
                    category=category,
                    term=term,
                    source_count=count_term(src_concat, term),
                    train_count=count_term(trn_concat, term),
                    source_chars=src_len,
                    train_chars=trn_len,
                )
                kind_results["domain_terms"][term] = entry
                if entry["severity"] != "ok":
                    gaps.append(entry)
                    direction = "UNDER" if entry["ratio"] != "inf" and entry["ratio"] < 1 else "OVER"
                    log(
                        f"  GAP [{entry['severity']}] {kind}/{category}/{term}: "
                        f"ratio={entry['ratio']}  ({direction}-represented)"
                    )

        for term, variants in PUNCTUATION_METRICS.items():
            entry = build_frequency_entry(
                kind=kind,
                category="punctuation",
                term=term,
                source_count=count_marker_variants(src_concat, variants),
                train_count=count_marker_variants(trn_concat, variants),
                source_chars=src_len,
                train_chars=trn_len,
            )
            kind_results["punctuation"][term] = entry
            if entry["severity"] != "ok":
                gaps.append(entry)
                direction = "UNDER" if entry["ratio"] != "inf" and entry["ratio"] < 1 else "OVER"
                log(
                    f"  GAP [{entry['severity']}] {kind}/punctuation/{term}: "
                    f"ratio={entry['ratio']}  ({direction}-represented)"
                )

        for term, variants in EMOTIVE_MARKERS.items():
            entry = build_frequency_entry(
                kind=kind,
                category="emotive_marker",
                term=term,
                source_count=count_marker_variants(src_concat, variants),
                train_count=count_marker_variants(trn_concat, variants),
                source_chars=src_len,
                train_chars=trn_len,
            )
            kind_results["emotive_markers"][term] = entry
            if entry["severity"] != "ok":
                gaps.append(entry)
                direction = "UNDER" if entry["ratio"] != "inf" and entry["ratio"] < 1 else "OVER"
                log(
                    f"  GAP [{entry['severity']}] {kind}/emotive_marker/{term}: "
                    f"ratio={entry['ratio']}  ({direction}-represented)"
                )

        results[kind] = kind_results

    log(f"  Kind-stratified frequency gaps found: {len(gaps)}")
    return results, gaps


# ──────────────────────────────────────────────────────────────
# Phase 1b: Structure comparison
# ──────────────────────────────────────────────────────────────

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


def classify_text_type(text: str) -> str:
    q_marks = text.count("?") + text.count("？")
    if q_marks >= 2:
        return "질문형"
    review_keywords = ("후기", "솔직히", "다녀왔", "갔는데", "갔다", "해봤", "먹어봤")
    if any(kw in text for kw in review_keywords):
        return "후기형"
    return "정보형"


def structure_stats(texts: list[str], sample_n: int = STRUCTURE_SAMPLE_SIZE) -> dict:
    if len(texts) > sample_n:
        sampled = random.sample(texts, sample_n)
    else:
        sampled = texts

    tone_counter: Counter = Counter()
    type_counter: Counter = Counter()
    sentence_counts = []
    words_per_sentence_all = []

    for text in sampled:
        tone_counter[classify_tone(text)] += 1
        type_counter[classify_text_type(text)] += 1

        # Split sentences on common Korean sentence boundaries
        sentences = re.split(r"[.!?。？！\n]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_counts.append(len(sentences))

        for sent in sentences:
            word_count = len(re.findall(r"\S+", sent))
            if word_count > 0:
                words_per_sentence_all.append(word_count)

    n = len(sampled)
    if n == 0:
        return {
            "sample_size": 0,
            "tone_distribution": {"반말": 0.0, "존댓말": 0.0, "혼합": 0.0},
            "text_type_distribution": {"정보형": 0.0, "질문형": 0.0, "후기형": 0.0},
            "avg_sentences_per_text": 0.0,
            "avg_words_per_sentence": 0.0,
        }

    avg_sentences = sum(sentence_counts) / max(n, 1)
    avg_words_per_sent = (
        sum(words_per_sentence_all) / max(len(words_per_sentence_all), 1)
    )

    return {
        "sample_size": n,
        "tone_distribution": {
            k: round(v / n, 4) for k, v in tone_counter.most_common()
        },
        "text_type_distribution": {
            k: round(v / n, 4) for k, v in type_counter.most_common()
        },
        "avg_sentences_per_text": round(avg_sentences, 2),
        "avg_words_per_sentence": round(avg_words_per_sent, 2),
    }


def phase1b_structure(src_by_kind: dict, trn_by_kind: dict):
    log("Phase 1b: Kind-stratified structure comparison (500-sample each) ...")
    results = {}
    gaps = []

    for kind in KIND_ORDER:
        src_texts = src_by_kind[kind]["texts"]
        trn_texts = trn_by_kind[kind]["texts"]
        log(
            f"  Sampling {kind}: source={len(src_texts):,} texts  |  train={len(trn_texts):,} texts"
        )

        src_stats = structure_stats(src_texts)
        trn_stats = structure_stats(trn_texts)
        kind_gaps = []

        for tone in ("반말", "존댓말", "혼합"):
            src_v = src_stats["tone_distribution"].get(tone, 0)
            trn_v = trn_stats["tone_distribution"].get(tone, 0)
            diff = abs(src_v - trn_v)
            if diff > 0.15:
                severity = "critical" if diff > 0.3 else "moderate"
                gap = {
                    "kind": kind,
                    "metric": f"tone_{tone}",
                    "source": src_v,
                    "train": trn_v,
                    "diff": round(diff, 4),
                    "severity": severity,
                }
                kind_gaps.append(gap)
                gaps.append(gap)
                log(
                    f"  GAP [{severity}] {kind}/tone/{tone}: "
                    f"source={src_v:.2%} train={trn_v:.2%}"
                )

        for ttype in ("정보형", "질문형", "후기형"):
            src_v = src_stats["text_type_distribution"].get(ttype, 0)
            trn_v = trn_stats["text_type_distribution"].get(ttype, 0)
            diff = abs(src_v - trn_v)
            if diff > 0.15:
                severity = "critical" if diff > 0.3 else "moderate"
                gap = {
                    "kind": kind,
                    "metric": f"type_{ttype}",
                    "source": src_v,
                    "train": trn_v,
                    "diff": round(diff, 4),
                    "severity": severity,
                }
                kind_gaps.append(gap)
                gaps.append(gap)
                log(
                    f"  GAP [{severity}] {kind}/type/{ttype}: "
                    f"source={src_v:.2%} train={trn_v:.2%}"
                )

        s_sent = src_stats["avg_sentences_per_text"]
        t_sent = trn_stats["avg_sentences_per_text"]
        if s_sent > 0:
            sent_ratio = t_sent / s_sent
            if sent_ratio < GAP_LOW or sent_ratio > GAP_HIGH:
                severity = "critical" if sent_ratio < 0.25 or sent_ratio > 4.0 else "moderate"
                gap = {
                    "kind": kind,
                    "metric": "avg_sentences_per_text",
                    "source": s_sent,
                    "train": t_sent,
                    "ratio": round(sent_ratio, 4),
                    "severity": severity,
                }
                kind_gaps.append(gap)
                gaps.append(gap)
                log(
                    f"  GAP [{severity}] {kind}/avg_sentences: "
                    f"source={s_sent} train={t_sent}"
                )

        s_wps = src_stats["avg_words_per_sentence"]
        t_wps = trn_stats["avg_words_per_sentence"]
        if s_wps > 0:
            wps_ratio = t_wps / s_wps
            if wps_ratio < GAP_LOW or wps_ratio > GAP_HIGH:
                severity = "critical" if wps_ratio < 0.25 or wps_ratio > 4.0 else "moderate"
                gap = {
                    "kind": kind,
                    "metric": "avg_words_per_sentence",
                    "source": s_wps,
                    "train": t_wps,
                    "ratio": round(wps_ratio, 4),
                    "severity": severity,
                }
                kind_gaps.append(gap)
                gaps.append(gap)
                log(
                    f"  GAP [{severity}] {kind}/avg_words_per_sentence: "
                    f"source={s_wps} train={t_wps}"
                )

        results[kind] = {
            "source": src_stats,
            "train": trn_stats,
            "gaps": kind_gaps,
        }

    log(f"  Kind-stratified structure gaps found: {len(gaps)}")
    return {"by_kind": results, "gaps": gaps}


# ──────────────────────────────────────────────────────────────
# Phase 1c: Post coverage & length distribution
# ──────────────────────────────────────────────────────────────

def length_histogram(texts: list[str], bins=None) -> dict[str, int]:
    """Bucket texts by character length."""
    if bins is None:
        bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]
    counts: dict[str, int] = {}
    for i in range(len(bins)):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else math.inf
        label = f"{lo}-{hi}" if hi != math.inf else f"{lo}+"
        counts[label] = 0

    for t in texts:
        tlen = len(t)
        placed = False
        for i in range(len(bins)):
            lo = bins[i]
            hi = bins[i + 1] if i + 1 < len(bins) else math.inf
            label = f"{lo}-{hi}" if hi != math.inf else f"{lo}+"
            if lo <= tlen < hi:
                counts[label] += 1
                placed = True
                break
        if not placed:
            # Longer than last bin
            last_label = f"{bins[-1]}+"
            counts[last_label] = counts.get(last_label, 0) + 1

    return counts


def phase1c_coverage(
    src_ids: set[str],
    trn_ids: set[str],
    src_texts: list[str],
    trn_texts: list[str],
):
    log("Phase 1c: Post coverage & length distribution ...")
    covered = src_ids & trn_ids
    coverage_pct = len(covered) / max(len(src_ids), 1) * 100

    log(f"  Source posts: {len(src_ids):,}  |  Covered in train: {len(covered):,}  ({coverage_pct:.1f}%)")

    src_hist = length_histogram(src_texts)
    trn_hist = length_histogram(trn_texts)

    gaps = []
    # Flag bins where proportions differ significantly
    src_total = max(len(src_texts), 1)
    trn_total = max(len(trn_texts), 1)
    for bin_label in src_hist:
        src_pct = src_hist[bin_label] / src_total
        trn_pct = trn_hist.get(bin_label, 0) / trn_total
        diff = abs(src_pct - trn_pct)
        if diff > 0.1:
            severity = "critical" if diff > 0.2 else "moderate"
            gaps.append({
                "metric": f"length_bin_{bin_label}",
                "source_pct": round(src_pct, 4),
                "train_pct": round(trn_pct, 4),
                "diff": round(diff, 4),
                "severity": severity,
            })
            log(f"  GAP [{severity}] length bin {bin_label}: source={src_pct:.1%} train={trn_pct:.1%}")

    if coverage_pct < 50:
        gaps.append({
            "metric": "post_coverage",
            "value": round(coverage_pct, 2),
            "severity": "critical",
        })
        log(f"  GAP [critical] post coverage only {coverage_pct:.1f}%")
    elif coverage_pct < 80:
        gaps.append({
            "metric": "post_coverage",
            "value": round(coverage_pct, 2),
            "severity": "moderate",
        })
        log(f"  GAP [moderate] post coverage {coverage_pct:.1f}%")

    return {
        "source_post_count": len(src_ids),
        "train_source_ids": len(trn_ids),
        "covered": len(covered),
        "coverage_pct": round(coverage_pct, 2),
        "source_length_histogram": src_hist,
        "train_length_histogram": trn_hist,
        "gaps": gaps,
    }


# ──────────────────────────────────────────────────────────────
# Phase 2: Report generation
# ──────────────────────────────────────────────────────────────

def build_report(
    term_results: dict,
    term_gaps: list,
    structure: dict,
    coverage: dict,
    cycle: int,
) -> dict:
    all_gaps = []
    for g in term_gaps:
        all_gaps.append({
            **g,
            "phase": "1a_term_freq",
            "action": "manual_review" if g["severity"] == "critical" else "monitor",
        })
    for g in structure["gaps"]:
        all_gaps.append({
            **g,
            "phase": "1b_structure",
            "action": "report_only",
        })
    for g in coverage["gaps"]:
        all_gaps.append({
            **g,
            "phase": "1c_coverage",
            "action": "report_only",
        })

    summary = {
        "total_gaps": len(all_gaps),
        "critical": sum(1 for g in all_gaps if g.get("severity") == "critical"),
        "moderate": sum(1 for g in all_gaps if g.get("severity") == "moderate"),
        "minor": sum(1 for g in all_gaps if g.get("severity") == "minor"),
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "cycle": cycle,
        "summary": summary,
        "term_frequency": term_results,
        "structure_comparison": {
            kind: {
                "source": structure["by_kind"][kind]["source"],
                "train": structure["by_kind"][kind]["train"],
            }
            for kind in KIND_ORDER
        },
        "coverage": {
            k: v for k, v in coverage.items() if k != "gaps"
        },
        "gaps": all_gaps,
        "recommended_actions": _build_recommendations(all_gaps),
    }


def _build_recommendations(gaps: list) -> list[dict]:
    recs = []
    seen = set()

    for g in gaps:
        phase = g.get("phase", "")
        metric = g.get("metric", g.get("term", ""))
        kind = g.get("kind")
        kind_prefix = f"{kind} " if kind else ""
        key = f"{phase}:{kind or 'overall'}:{metric}"
        if key in seen:
            continue
        seen.add(key)

        if phase == "1a_term_freq":
            term = g.get("term", "")
            ratio = g.get("ratio", 1)
            if ratio == "inf" or (isinstance(ratio, (int, float)) and ratio > GAP_HIGH):
                recs.append({
                    "priority": g["severity"],
                    "target": f"term:{kind or 'overall'}:{term}",
                    "description": f"{kind_prefix}term '{term}' is OVER-represented in training vs source (ratio={ratio}). "
                                   "Check if training pipeline is duplicating or over-sampling texts containing this term.",
                    "auto_fixable": False,
                })
            else:
                recs.append({
                    "priority": g["severity"],
                    "target": f"term:{kind or 'overall'}:{term}",
                    "description": f"{kind_prefix}term '{term}' is UNDER-represented in training vs source (ratio={ratio}). "
                                   "Likely caused by promo/spam filter removing posts containing this term. "
                                   "Review filter rules.",
                    "auto_fixable": False,
                })
        elif phase == "1b_structure":
            recs.append({
                "priority": g["severity"],
                "target": f"structure:{kind or 'overall'}:{metric}",
                "description": f"{kind_prefix}structural gap in {metric}: source={g.get('source', '?')} vs train={g.get('train', '?')}. "
                               "Consider adjusting sampling strategy to match source distribution.",
                "auto_fixable": False,
            })
        elif phase == "1c_coverage":
            if metric == "post_coverage":
                recs.append({
                    "priority": g["severity"],
                    "target": "coverage",
                    "description": f"Only {g.get('value', '?')}% of source posts are represented in training data. "
                                   "Consider re-running the data pipeline with relaxed filters.",
                    "auto_fixable": False,
                })
            else:
                recs.append({
                    "priority": g["severity"],
                    "target": f"length_dist:{metric}",
                    "description": f"Length distribution mismatch in bin {metric}. "
                                   "Adjust length-based sampling or filtering.",
                    "auto_fixable": False,
                })

    # Sort by severity
    severity_order = {"critical": 0, "moderate": 1, "minor": 2, "ok": 3}
    recs.sort(key=lambda r: severity_order.get(r["priority"], 9))
    return recs


# ──────────────────────────────────────────────────────────────
# Phase 3: Auto-fix (stub — flags only)
# ──────────────────────────────────────────────────────────────

def phase3_autofix(report: dict, run_dir: Path) -> list[dict]:
    """Evaluate gaps and apply lightweight fixes where possible.

    Currently all detected gaps are flagged for manual review because:
    - Term underrepresentation is usually caused by the promo filter,
      which should not be loosened automatically.
    - Structural gaps require human judgment on sampling strategy.

    Returns a list of actions taken (or deferred).
    """
    log("Phase 3: Auto-fix evaluation ...")
    actions = []
    for gap in report.get("gaps", []):
        phase = gap.get("phase", "")
        severity = gap.get("severity", "ok")
        metric = gap.get("metric", gap.get("term", "unknown"))
        kind = gap.get("kind", "overall")

        if phase == "1a_term_freq" and severity in ("critical", "moderate"):
            actions.append({
                "gap": f"{phase}:{kind}:{metric}",
                "action": "DEFERRED — promo filter may be cause; manual review required",
                "applied": False,
            })
        elif phase == "1b_structure":
            actions.append({
                "gap": f"{phase}:{kind}:{metric}",
                "action": "REPORT_ONLY — structural gap, adjust sampling manually",
                "applied": False,
            })
        elif phase == "1c_coverage":
            actions.append({
                "gap": f"{phase}:{metric}",
                "action": "REPORT_ONLY — coverage/length gap, review pipeline filters",
                "applied": False,
            })

    # Write fix log
    fix_log_path = run_dir / "fix_log.json"
    with open(fix_log_path, "w", encoding="utf-8") as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)
    log(f"  Fix log written to {fix_log_path}")
    log(f"  Actions evaluated: {len(actions)}  |  Applied: {sum(1 for a in actions if a['applied'])}")
    return actions


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────

def run_cycle(source_dir: Path, run_dir: Path, cycle: int, do_fix: bool):
    log(f"{'=' * 60}")
    log(f"Cycle {cycle} starting")
    log(f"{'=' * 60}")

    # Load source data
    log("Loading source data ...")
    src_corpus = load_source_concat(source_dir)
    log(
        f"  Source: {len(src_corpus['texts']):,} texts "
        f"({src_corpus['kind_counts'].get('post', 0):,} posts, "
        f"{src_corpus['kind_counts'].get('comment', 0):,} comments) "
        f"from {len(src_corpus['source_ids']):,} unique posts"
    )

    # Load training data
    log("Loading training data ...")
    trn_corpus = load_train_concat()
    log(
        f"  Training: {len(trn_corpus['texts']):,} texts "
        f"from {len(trn_corpus['source_ids']):,} unique source IDs"
    )
    for k, v in trn_corpus["kind_counts"].items():
        log(f"    {k}: {v:,}")

    # Phase 1a
    term_results, term_gaps = phase1a_term_freq(src_corpus, trn_corpus)

    # Phase 1b
    structure = phase1b_structure(src_corpus["by_kind"], trn_corpus["by_kind"])

    # Phase 1c
    coverage = phase1c_coverage(
        src_corpus["source_ids"],
        trn_corpus["source_ids"],
        src_corpus["texts"],
        trn_corpus["texts"],
    )

    # Phase 2: Report
    log("Phase 2: Generating report ...")
    report = build_report(term_results, term_gaps, structure, coverage, cycle)

    report_path = run_dir / "diagnostic.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"  Report written to {report_path}")

    # Summary to stdout
    s = report["summary"]
    log(f"  SUMMARY: {s['total_gaps']} gaps "
        f"(critical={s['critical']}, moderate={s['moderate']}, minor={s['minor']})")

    if report["recommended_actions"]:
        log("  TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommended_actions"][:5], 1):
            log(f"    {i}. [{rec['priority']}] {rec['description'][:100]}")

    # Phase 3: Auto-fix
    if do_fix:
        phase3_autofix(report, run_dir)
    else:
        log("Phase 3: Skipped (use --fix to enable)")

    log(f"Cycle {cycle} complete.")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Refinement loop: diagnostic → fix → verify for source-vs-training GAP analysis",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to crawled source data directory (e.g. /Users/unoa/Downloads/crawled-data-v2)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        default=False,
        help="Enable auto-fix phase (applies lightweight fixes where possible)",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=1,
        help="Number of diagnostic cycles to run (default: 1)",
    )
    parser.add_argument(
        "--cpt-jsonl",
        type=str,
        default=None,
        help="Override CPT input JSONL for diagnostics (default: TRAIN_CPT_JSONL env, else cpt_context_stream.jsonl when present, else cpt_corpus.v3.jsonl, else cpt_corpus.v2.jsonl)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    TRAIN_FILES["cpt"] = resolve_cpt_path(args.cpt_jsonl)
    log(f"Using CPT corpus: {TRAIN_FILES['cpt']}")

    # Verify training files exist
    for name, path in TRAIN_FILES.items():
        if not path.exists():
            log(f"WARNING: Training file not found: {path}  (will skip {name})")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for cycle_num in range(1, args.cycle + 1):
        run_dir = PROJECT_ROOT / "runs" / f"refinement-{timestamp}" / f"cycle-{cycle_num}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"Run directory: {run_dir}")

        report = run_cycle(source_dir, run_dir, cycle_num, args.fix)

        if cycle_num < args.cycle:
            log(f"--- Pausing before next cycle ---")
            # In a real fix scenario, the next cycle would pick up
            # the modified training data and re-diagnose.
            if report["summary"]["total_gaps"] == 0:
                log("No gaps remaining. Stopping early.")
                break

    log("All cycles complete.")


if __name__ == "__main__":
    main()
