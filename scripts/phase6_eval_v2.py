#!/usr/bin/env python3
"""phase6_eval_v2.py — Round-2 extended evaluator.

Wraps the deterministic 7-metric evaluator from phase6_eval.py and adds:

  8. punct_ratio_match     — per-kind ratio of '?', '!', '...' (max abs delta)
  9. choseong_marker_match — per-kind ratio of ㅋㅋ/ㅠㅠ/ㄹㅇ/ㅇㅈ/ㅈㄴ/ㅅㅂ (max abs delta)
 10. reply_depth_kl        — comment thread depth distribution KL divergence
 11. persona_consistency   — fraction of generated samples whose 'persona' field
                             label is in the persona-30 list (when supplied)
 12. cross_machine_agreement — JSD between two evaluator runs
                             (e.g. mac-mini MPS vs windows-NPU local) — supplied
                             via --secondary-ai

The wrapper retains the original 7 gates and adds new gates per the
recommended-stack.md (round2-methodology). All thresholds are loaded from
.planning/calibration/raw-vs-raw-v2.json if present, else fall back to defaults
embedded below.

Inputs:
  --ai          AI-generated jsonl                  (one of: text/output/generated)
  --raw         Held-out raw jsonl                   (text)
  --persona-list runs/round2-obsidian-synthesis/persona-30-extracted.json
  --secondary-ai (optional) second AI jsonl for cross-machine agreement

CLI:
  python scripts/phase6_eval_v2.py --ai gen.jsonl --raw raw.jsonl \\
      --persona-list runs/round2-obsidian-synthesis/persona-30-extracted.json \\
      --out report.json --skip-mauve

Exit codes:
  0  PASS    (all gates met)
  2  FAIL    (>=1 gate violated)
  3  ERROR   (inputs missing / invalid)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

# Reuse the deterministic core from phase6_eval where reasonable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6_eval as base  # type: ignore  # noqa: E402


PUNCT_PATTERNS = {
    "q": re.compile(r"\?"),
    "ex": re.compile(r"!"),
    "ellipsis": re.compile(r"\.{2,}|…"),
}

CHOSEONG_PATTERNS = {
    "kkk": re.compile(r"ㅋ{2,}|크{2,}"),
    "huhu": re.compile(r"ㅎ{2,}"),
    "tt": re.compile(r"ㅠ{1,}|ㅜ{1,}"),
    "rieng": re.compile(r"ㄹㅇ"),
    "oj": re.compile(r"ㅇㅈ"),
    "jn": re.compile(r"ㅈㄴ"),
    "sb": re.compile(r"ㅅㅂ"),
}

REPLY_DEPTH = re.compile(r"^\s*\[(\d+(?:-\d+)*)\]")

# Round-2 specific gates — initial calibration; tighten in cycle 1
GATE_V2 = {
    "punct_ratio_match_max": ("le", 0.15),       # max abs delta any of q/ex/ellipsis
    "choseong_marker_match_max": ("le", 0.20),   # max abs delta any of 7 markers
    "reply_depth_kl": ("le", 0.20),
    "persona_consistency": ("ge", 0.85),
    "cross_machine_agreement": ("le", 0.10),     # JSD-like, smaller is better when supplied
}


def normalize_persona_tokens(value: object) -> set[object]:
    raw = str(value or "").strip()
    if not raw:
        return set()
    tokens: set[object] = {raw}
    lowered = raw.lower()
    if lowered.startswith("p-"):
        suffix = raw[2:]
        if suffix:
            tokens.add(suffix)
            tokens.add(suffix.lstrip("0") or "0")
            try:
                tokens.add(int(suffix))
            except ValueError:
                pass
    try:
        tokens.add(int(raw))
    except ValueError:
        pass
    return tokens


def load_full_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = (obj.get("text") or obj.get("generated") or obj.get("output") or "").strip()
            if not text:
                continue
            obj = dict(obj)
            obj["text"] = text
            obj["kind"] = base.normalize_kind(obj.get("kind") or obj.get("pair_type"))
            rows.append(obj)
    return rows


def punct_ratio(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {k: 0.0 for k in PUNCT_PATTERNS}
    return {
        name: sum(1 for t in texts if pat.search(t or "")) / len(texts)
        for name, pat in PUNCT_PATTERNS.items()
    }


def choseong_ratio(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {k: 0.0 for k in CHOSEONG_PATTERNS}
    return {
        name: sum(1 for t in texts if pat.search(t or "")) / len(texts)
        for name, pat in CHOSEONG_PATTERNS.items()
    }


def punct_ratio_match(ai_rows: list[dict], raw_rows: list[dict]) -> tuple[float, dict]:
    out: dict[str, dict] = {}
    overall_max = 0.0
    for kind in base.KIND_ORDER + ("any",):
        if kind == "any":
            ai_t = base.texts_from_rows(ai_rows)
            raw_t = base.texts_from_rows(raw_rows)
        else:
            ai_t = base.texts_for_kind(ai_rows, kind)
            raw_t = base.texts_for_kind(raw_rows, kind)
        if not ai_t or not raw_t:
            continue
        ar = punct_ratio(ai_t)
        rr = punct_ratio(raw_t)
        deltas = {k: abs(ar[k] - rr[k]) for k in PUNCT_PATTERNS}
        out[kind] = {"generated": ar, "raw": rr, "deltas": deltas}
        overall_max = max(overall_max, max(deltas.values()))
    return overall_max, out


def choseong_marker_match(ai_rows: list[dict], raw_rows: list[dict]) -> tuple[float, dict]:
    out: dict[str, dict] = {}
    overall_max = 0.0
    for kind in base.KIND_ORDER + ("any",):
        if kind == "any":
            ai_t = base.texts_from_rows(ai_rows)
            raw_t = base.texts_from_rows(raw_rows)
        else:
            ai_t = base.texts_for_kind(ai_rows, kind)
            raw_t = base.texts_for_kind(raw_rows, kind)
        if not ai_t or not raw_t:
            continue
        ar = choseong_ratio(ai_t)
        rr = choseong_ratio(raw_t)
        deltas = {k: abs(ar[k] - rr[k]) for k in CHOSEONG_PATTERNS}
        out[kind] = {"generated": ar, "raw": rr, "deltas": deltas}
        overall_max = max(overall_max, max(deltas.values()))
    return overall_max, out


def reply_depth_value(text: str) -> int:
    m = REPLY_DEPTH.match(text or "")
    return 0 if not m else m.group(1).count("-") + 1


def reply_depth_distribution(rows: list[dict]) -> list[float]:
    """Histogram of reply depths 0..4 over comment rows; 0 if no comments present."""
    counts = [0, 0, 0, 0, 0]  # bins: 0,1,2,3,4+
    n = 0
    for r in rows:
        if r.get("kind") != "comment":
            continue
        d = reply_depth_value(str(r.get("text") or ""))
        counts[min(d, 4)] += 1
        n += 1
    if not n:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [c / n for c in counts]


def reply_depth_kl(ai_rows: list[dict], raw_rows: list[dict]) -> tuple[float, dict]:
    p = reply_depth_distribution(ai_rows)
    q = reply_depth_distribution(raw_rows)
    if sum(q) == 0:
        return 0.0, {"generated": p, "raw": q, "skipped": "no raw comments"}
    return base.kl_divergence(p, q), {"generated": p, "raw": q}


def persona_consistency(ai_rows: list[dict], persona_list_path: Path | None) -> tuple[float, dict]:
    if not persona_list_path:
        return None, {"waived": "no persona-list provided"}  # type: ignore[return-value]
    if not persona_list_path.exists():
        return 0.0, {"error": "persona-list missing"}
    payload = json.loads(persona_list_path.read_text(encoding="utf-8"))
    accepted = payload.get("personas") or []
    accepted_ids = {p.get("id") for p in accepted}
    accepted_names = {p.get("name") for p in accepted if p.get("name")}
    accepted_tokens: set[object] = set()
    for pid in accepted_ids:
        accepted_tokens.update(normalize_persona_tokens(pid))
    matched = 0
    declared = 0
    total = len(ai_rows)
    for r in ai_rows:
        pid = r.get("persona_id") if isinstance(r, dict) else None
        pname = r.get("persona") if isinstance(r, dict) else None
        if pid is None and not pname:
            continue
        declared += 1
        if normalize_persona_tokens(pid) & accepted_tokens or pname in accepted_names:
            matched += 1
    if not total:
        return 0.0, {"error": "no generated rows"}
    if not declared:
        return 0.0, {"error": "no persona-tagged rows"}
    score = matched / total
    return score, {
        "total_rows": total,
        "declared": declared,
        "missing": total - declared,
        "matched": matched,
        "score": score,
        "accepted_ids": sorted(str(v) for v in accepted_ids if v is not None),
    }


def cross_machine_agreement(ai_rows: list[dict], secondary_rows: list[dict]) -> tuple[float, dict]:
    """JSD between two AI runs aggregated bigram distributions (smaller = more agreement)."""
    if not secondary_rows:
        return None, {"waived": "no secondary-ai provided"}  # type: ignore[return-value]
    a_t = base.texts_from_rows(ai_rows)
    b_t = base.texts_from_rows(secondary_rows)
    if not a_t or not b_t:
        return 0.0, {"skipped": "empty texts"}
    p = base.normalize(base.aggregate_bigrams(a_t))
    q = base.normalize(base.aggregate_bigrams(b_t))
    return base.jsd(p, q), {"n_primary": len(a_t), "n_secondary": len(b_t)}


def evaluate_v2_gate(
    metrics: dict[str, float | None],
    skipped_keys: frozenset[str] = frozenset(),
) -> tuple[str, list[str]]:
    # cycle-6 B5a parity: a v2 gate metric whose value is None means "we could
    # not compute it" — that must FAIL, not silently PASS. Mirrors the v1 gate
    # behaviour in phase6_eval.evaluate_gate. Callers pass `skipped_keys` for
    # metrics that are legitimately optional (e.g. cross_machine_agreement
    # when no --secondary-ai is provided).
    violations: list[str] = []
    for key, (op, threshold) in GATE_V2.items():
        if key in skipped_keys:
            continue
        value = metrics.get(key)
        if value is None:
            violations.append(f"{key}=value_unavailable reason=value_unavailable")
            continue
        if op == "le" and value > threshold:
            violations.append(f"{key}={value:.4f} > {threshold}")
        elif op == "ge" and value < threshold:
            violations.append(f"{key}={value:.4f} < {threshold}")
    return ("PASS" if not violations else "FAIL"), violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True, help="AI-generated samples JSONL")
    ap.add_argument("--raw", required=True, help="Raw held-out samples JSONL")
    ap.add_argument("--secondary-ai", help="Second AI run (e.g. mac-mini MPS) for agreement")
    ap.add_argument("--persona-list", help="persona-30-extracted.json")
    ap.add_argument("--out", help="Write JSON report to this path")
    ap.add_argument("--skip-mauve", action="store_true")
    ap.add_argument(
        "--min-rows",
        type=int,
        default=20,
        help="Minimum AI/raw rows required for a stable gate (default: 20)",
    )
    args = ap.parse_args()

    ai_rows = load_full_rows(args.ai)
    raw_rows = load_full_rows(args.raw)
    secondary_rows = load_full_rows(args.secondary_ai) if args.secondary_ai else []
    if not ai_rows or not raw_rows:
        sys.stderr.write(f"[error] empty inputs: ai={len(ai_rows)} raw={len(raw_rows)}\n")
        return 3

    ai_texts = base.texts_from_rows(ai_rows)
    raw_texts = base.texts_from_rows(raw_rows)
    base_skipped: frozenset[str] = frozenset({"mauve_score"}) if args.skip_mauve else frozenset()
    base_metrics, base_details = base.compute_metric_bundle(
        ai_texts, raw_texts, include_mauve=not args.skip_mauve
    )
    base_verdict, base_violations = base.evaluate_gate(base_metrics, skipped_keys=base_skipped)
    sample_violations: list[str] = []
    if len(ai_rows) < args.min_rows:
        sample_violations.append(f"ai_rows={len(ai_rows)} < min_rows={args.min_rows}")
    if len(raw_rows) < args.min_rows:
        sample_violations.append(f"raw_rows={len(raw_rows)} < min_rows={args.min_rows}")
    if len(ai_rows) != len(raw_rows):
        sample_violations.append(f"row_count_mismatch ai={len(ai_rows)} raw={len(raw_rows)}")

    # New v2 metrics
    p_ratio_max, p_ratio_detail = punct_ratio_match(ai_rows, raw_rows)
    c_marker_max, c_marker_detail = choseong_marker_match(ai_rows, raw_rows)
    rd_kl, rd_detail = reply_depth_kl(ai_rows, raw_rows)
    persona_score, persona_detail = persona_consistency(
        ai_rows, Path(args.persona_list) if args.persona_list else None
    )
    cma, cma_detail = cross_machine_agreement(ai_rows, secondary_rows)

    v2_metrics = {
        "punct_ratio_match_max": p_ratio_max,
        "choseong_marker_match_max": c_marker_max,
        "reply_depth_kl": rd_kl,
        "persona_consistency": persona_score,
        "cross_machine_agreement": cma,
    }
    # cross_machine_agreement is optional: when --secondary-ai is not provided
    # there are no secondary rows, so the metric is genuinely uncomputable and
    # must be skipped rather than failing the gate (B5a parity).
    v2_skipped: frozenset[str] = (
        frozenset({"cross_machine_agreement"}) if not secondary_rows else frozenset()
    )
    v2_verdict, v2_violations = evaluate_v2_gate(v2_metrics, skipped_keys=v2_skipped)

    overall_violations = list(sample_violations) + list(base_violations) + list(v2_violations)
    overall_verdict = "PASS" if not overall_violations else "FAIL"

    report = {
        "phase6_v2_version": "round2-1",
        "base": {
            "metrics": base_metrics,
            "details": base_details,
            "gate": {"verdict": base_verdict, "violations": base_violations},
        },
        "sample": {
            "ai_rows": len(ai_rows),
            "raw_rows": len(raw_rows),
            "secondary_rows": len(secondary_rows),
            "min_rows": args.min_rows,
            "violations": sample_violations,
        },
        "v2": {
            "metrics": v2_metrics,
            "details": {
                "punct_ratio": p_ratio_detail,
                "choseong_marker": c_marker_detail,
                "reply_depth": rd_detail,
                "persona_consistency": persona_detail,
                "cross_machine_agreement": cma_detail,
            },
            "gate": {"verdict": v2_verdict, "violations": v2_violations},
        },
        "overall": {"verdict": overall_verdict, "violations": overall_violations},
        "thresholds": {
            "base": {k: {"op": op, "value": v} for k, (op, v) in base.GATE.items()},
            "v2": {k: {"op": op, "value": v} for k, (op, v) in GATE_V2.items()},
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0 if overall_verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
