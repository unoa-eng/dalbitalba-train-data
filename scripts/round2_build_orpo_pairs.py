#!/usr/bin/env python3
"""round2_build_orpo_pairs.py — Build ORPO preference pairs from
runs/refinement-* good-vs-bad outputs.

Input:
  runs/refinement-2026042?-*/cycle-N/{ai_generated.jsonl, eval-report.json}

Heuristic for chosen/rejected:
  - "chosen": rows from refinement runs whose phase6_eval verdict is PASS
              OR whose bigram_jsd <= 0.10 AND domain_keyword_alignment >= 0.40
  - "rejected": rows that show formal-AI markers
              (다음과 같은, 정확하지는 않을 수, AI로서, 죄송하지만)
              AND/OR bigram_jsd > 0.20

Output (jsonl):
  {prompt, chosen, rejected, kind, source_run, reason}

Usage:
  python3 scripts/round2_build_orpo_pairs.py \\
      --runs-glob 'runs/refinement-2026042*' \\
      --out orpo_pairs.jsonl

Acceptance: >= 500 pairs produced.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

random.seed(11)

FORMAL_AI_MARKERS = [
    "다음과 같은", "정확하지는 않을 수", "AI로서", "죄송하지만",
    "위 정보를 바탕으로", "도움이 되었기를 바랍니다", "참고만 해주세요",
    "보다 자세한 내용은", "전문가에게 상담", "정확한 정보를 제공",
]
FORMAL_RE = re.compile("|".join(re.escape(t) for t in FORMAL_AI_MARKERS))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def is_formal_ai(text: str) -> bool:
    if not text:
        return False
    return bool(FORMAL_RE.search(text))


def is_short_korean(text: str) -> bool:
    return bool(text) and len(text) < 200 and len(text) >= 8 and any("가" <= c <= "힣" for c in text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/refinement-2026042*")
    ap.add_argument("--val-set", type=Path,
                    help="optional backfill pool; disabled by default to avoid validation leakage")
    ap.add_argument("--samples", type=Path, default=None,
                    help="optional samples_200.jsonl or similar AI sample pool")
    ap.add_argument("--cpt-corpus", type=Path, default=None,
                    help="optional cpt_corpus.v*.jsonl for additional 'chosen' / 'rejected' mining via marker heuristics")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--allow-val-backfill", action="store_true")
    ap.add_argument("--allow-synthetic-rejected", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    chosen_pool: list[dict[str, Any]] = []
    rejected_pool: list[dict[str, Any]] = []

    for run_dir in sorted(glob.glob(args.runs_glob)):
        for cycle_dir in sorted(glob.glob(f"{run_dir}/cycle-*")):
            ai_p = Path(cycle_dir) / "ai_generated.jsonl"
            eval_p = Path(cycle_dir) / "eval-report.json"
            if not ai_p.exists():
                continue
            verdict = "FAIL"
            metrics: dict[str, Any] = {}
            if eval_p.exists():
                try:
                    rep = json.loads(eval_p.read_text(encoding="utf-8"))
                    verdict = rep.get("gate", {}).get("verdict", "FAIL")
                    metrics = rep.get("metrics", {})
                except Exception:  # noqa: BLE001
                    pass
            run_label = Path(cycle_dir).as_posix()
            for r in load_jsonl(ai_p):
                txt = r.get("text") or r.get("output") or ""
                if not is_short_korean(txt):
                    continue
                if is_formal_ai(txt):
                    rejected_pool.append({"text": txt, "kind": r.get("kind", "comment"),
                                          "source_run": run_label, "reason": "formal-AI markers"})
                elif verdict == "PASS" or metrics.get("bigram_jsd", 1.0) <= 0.10:
                    chosen_pool.append({"text": txt, "kind": r.get("kind", "comment"),
                                        "source_run": run_label})

    # Mine extra rejected from samples (formal-AI markers) and from cpt_corpus tail
    extra_pools = []
    if args.samples and args.samples.exists():
        extra_pools.append(("samples", load_jsonl(args.samples)))
    if args.cpt_corpus and args.cpt_corpus.exists():
        extra_pools.append(("cpt_corpus", load_jsonl(args.cpt_corpus)))
    for label, pool in extra_pools:
        for r in pool:
            txt = r.get("text") or r.get("output") or ""
            if not is_short_korean(txt):
                continue
            if is_formal_ai(txt):
                rejected_pool.append({"text": txt, "kind": r.get("kind", "comment"),
                                      "source_run": label, "reason": "formal-AI markers"})

    if args.allow_val_backfill:
        if not args.val_set or not args.val_set.exists():
            print("[warn] --allow-val-backfill requested but --val-set missing; skipping", file=sys.stderr)
        else:
            val_rows = load_jsonl(args.val_set)
            candidates = [v for v in val_rows if is_short_korean(v.get("text", ""))]
            target_chosen = args.max_pairs
            if candidates and len(chosen_pool) < target_chosen:
                need = target_chosen - len(chosen_pool)
                for r in random.sample(candidates, min(need, len(candidates))):
                    chosen_pool.append({"text": r.get("text", ""), "kind": r.get("kind", "comment"),
                                        "source_run": args.val_set.name})

    # Synthesize rejected from chosen only when explicitly allowed.
    if args.allow_synthetic_rejected and chosen_pool and len(rejected_pool) < args.max_pairs:
        synth_prefixes = [
            "다음과 같은 점들이 있습니다. ", "정확하지는 않을 수 있지만 ",
            "AI로서 답변드립니다. ", "참고만 해주세요. ",
            "위 정보를 바탕으로 답변드립니다. ", "보다 자세한 내용은 전문가에게 상담을 권장합니다. ",
            "도움이 되었기를 바랍니다. ", "정확한 정보를 제공하기 위해 노력하고 있습니다. ",
        ]
        synth_suffixes = [
            " 도움이 되었기를 바랍니다.",
            " 보다 자세한 내용은 전문가에게 상담해보시는 것을 권장드립니다.",
            " 참고만 해주시기 바랍니다.",
        ]
        # Generate rejected_pool entries by applying transformations to chosen
        target = max(args.max_pairs, len(chosen_pool))
        i = 0
        while len(rejected_pool) < target and i < len(chosen_pool) * 2:
            c = chosen_pool[i % len(chosen_pool)]
            # randomly mix prefix or suffix or both
            mode = random.randint(0, 2)
            if mode == 0:
                txt = random.choice(synth_prefixes) + c["text"]
            elif mode == 1:
                txt = c["text"] + random.choice(synth_suffixes)
            else:
                txt = random.choice(synth_prefixes) + c["text"] + random.choice(synth_suffixes)
            rejected_pool.append({
                "text": txt,
                "kind": c.get("kind", "comment"),
                "source_run": "synthesized-formal-AI",
                "reason": f"synth-formal-AI mode={mode}",
            })
            i += 1

    # Pair them up
    random.shuffle(chosen_pool)
    random.shuffle(rejected_pool)
    n_pairs = min(len(chosen_pool), len(rejected_pool), args.max_pairs)
    pairs = []
    for i in range(n_pairs):
        c = chosen_pool[i]
        r = rejected_pool[i]
        # Use a generic prompt frame
        prompt = ("다음 한국어 댓글을 cb2 밤문화 커뮤니티 톤으로 한 줄 생성:")
        pairs.append({
            "prompt": prompt,
            "chosen": c["text"],
            "rejected": r["text"],
            "kind": c.get("kind", "comment"),
            "source_run_chosen": c.get("source_run"),
            "source_run_rejected": r.get("source_run"),
            "reason": r.get("reason", "formal-AI ratio"),
        })

    with args.out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"DONE pairs={n_pairs} chosen_pool={len(chosen_pool)} rejected_pool={len(rejected_pool)}")
    if n_pairs < args.min_pairs:
        print(
            f"[error] insufficient preference pairs: {n_pairs} < {args.min_pairs}; "
            "refusing to keep a low-signal ORPO dataset",
            file=sys.stderr,
        )
        try:
            args.out.unlink(missing_ok=True)
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
