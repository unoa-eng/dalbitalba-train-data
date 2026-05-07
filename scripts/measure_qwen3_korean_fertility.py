#!/usr/bin/env python3
"""
measure_qwen3_korean_fertility.py — How well does the Qwen3-8B-Base BPE
tokenize dalbitalba's Korean nightlife/community speech?

Why this matters
----------------
Standard BPE on Korean (Thunder-Tok, arXiv:2506.15138) reaches ~1.51 tokens
per character; language-aware tokenizers reach ~1.37. The smaller chars/token
gets, the more compute and context budget the corpus eats, and the more
fragile rare slang ("ㅈㄴ", "ㄹㅇ", "쩜오", "텐카") becomes — those map to rare
token IDs whose embeddings barely got trained during the base model's
pre-training.

This script does a single GPU-free pass against val_set.v2.jsonl and emits a
JSON report so a regression in tokenization (e.g. after data regen, vocab
extension, or base-model swap) can be detected mechanically.

Output: .planning/calibration/qwen3_korean_fertility.json

Decision triggers (for recipe_mutator.py R7 / data regeneration):
- chars_per_token_p50 < 2.0  → Korean fertility too low, consider vocab
  expansion or a Korean-pretrained base (Bllossom-8B, Open-Ko-8B).
- slang_token_ratio > 1.5    → 초성/도메인 슬랭이 과도하게 분리되고 있음;
  검토 후 SFT 프롬프트에서 슬랭 일관성 보강 필요.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_SLANG_TERMS = [
    # 초성 (initial consonants used as full slang words)
    "ㅈㄴ", "ㅇㅈ", "ㄹㅇ", "ㅅㅂ", "ㄱㅊ", "ㄱㄱ", "ㄴㄴ",
    "ㅂㅅ", "ㄷㄷ", "ㄲㅃ", "ㅊㅇㅅ",
    # emotional markers
    "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ",
    # nightlife domain slang
    "하퍼", "초이스", "밀빵", "쩜오", "수위", "업진", "텐카", "셔츠",
    "상띠", "퍼블", "손놈", "보도", "가라", "노도", "하띠", "뺑이", "골타",
]


def _load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        sys.stderr.write(
            "[fertility] transformers not installed; "
            "run `pip install -U transformers tokenizers` first.\n"
        )
        raise SystemExit(2) from exc
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)


def measure(samples: list[str], tok, slang_terms: list[str]) -> dict:
    chars_per_tok: list[float] = []
    tokens_per_sample: list[int] = []
    for s in samples:
        if not s:
            continue
        ids = tok.encode(s, add_special_tokens=False)
        n_tok = max(1, len(ids))
        chars_per_tok.append(len(s) / n_tok)
        tokens_per_sample.append(n_tok)

    slang_breakdown = []
    multi_tok_slang = 0
    for term in slang_terms:
        ids = tok.encode(term, add_special_tokens=False)
        decoded = [tok.decode([i]) for i in ids]
        slang_breakdown.append(
            {"term": term, "n_tokens": len(ids), "decoded": decoded}
        )
        if len(ids) > 1:
            multi_tok_slang += 1

    def _q(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        idx = max(0, min(len(xs_sorted) - 1, int(round(q * (len(xs_sorted) - 1)))))
        return xs_sorted[idx]

    return {
        "samples_used": len(chars_per_tok),
        "chars_per_token": {
            "p50": _q(chars_per_tok, 0.5),
            "p10": _q(chars_per_tok, 0.1),
            "p90": _q(chars_per_tok, 0.9),
            "mean": statistics.fmean(chars_per_tok) if chars_per_tok else 0.0,
        },
        "tokens_per_sample": {
            "p50": _q(tokens_per_sample, 0.5),
            "p99": _q(tokens_per_sample, 0.99),
            "max": max(tokens_per_sample) if tokens_per_sample else 0,
        },
        "slang_breakdown": slang_breakdown,
        "slang_summary": {
            "n_terms": len(slang_terms),
            "n_terms_split_into_multiple_tokens": multi_tok_slang,
            "split_ratio": multi_tok_slang / max(1, len(slang_terms)),
        },
        "decision_triggers": {
            "low_korean_fertility": _q(chars_per_tok, 0.5) < 2.0,
            "high_slang_fragmentation": (
                multi_tok_slang / max(1, len(slang_terms))
            ) > 1.5,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    p.add_argument(
        "--samples-jsonl",
        default=str(REPO / "val_set.v2.jsonl"),
        help="JSONL file with a 'text' field per row",
    )
    p.add_argument("--field", default="text")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument(
        "--out",
        default=str(REPO / ".planning" / "calibration" / "qwen3_korean_fertility.json"),
    )
    args = p.parse_args()

    src = Path(args.samples_jsonl)
    if not src.exists():
        sys.stderr.write(f"[fertility] {src} not found\n")
        return 1
    samples: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = obj.get(args.field, "")
            if isinstance(txt, str):
                samples.append(txt)

    tok = _load_tokenizer(args.model)
    report = measure(samples, tok, DEFAULT_SLANG_TERMS)
    report["meta"] = {
        "model": args.model,
        "samples_jsonl": str(src),
        "field": args.field,
        "limit": args.limit,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[fertility] model={args.model} samples={report['samples_used']:,} "
        f"chars/tok p50={report['chars_per_token']['p50']:.3f} "
        f"slang split={report['slang_summary']['split_ratio']*100:.1f}% "
        f"-> {out}"
    )
    if report["decision_triggers"]["low_korean_fertility"]:
        print(
            "[fertility] WARN: chars/token p50 < 2.0 — vocab extension or "
            "Korean-pretrained base (Bllossom-8B / Open-Ko-8B) recommended.",
            file=sys.stderr,
        )
    if report["decision_triggers"]["high_slang_fragmentation"]:
        print(
            "[fertility] WARN: domain slang fragmenting into multiple tokens; "
            "downstream rare-token undertraining is likely.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
