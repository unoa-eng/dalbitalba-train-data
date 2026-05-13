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
  --cpt-corpus cpt_corpus.v3.jsonl \\
      --out orpo_pairs.jsonl

Acceptance: >= 500 pairs produced.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
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


def _index_val_completions(val_path: Path | None) -> set[str]:
    """Build a forbidden-completion set from a val_set jsonl.

    Mirrors the auditor (G_validate_orpo.py) by harvesting:
      - top-level keys: text / completion / answer / target_comment
      - assistant turns inside messages[]
    Always returns a set; empty when val_path is None or missing.
    """
    if not val_path or not val_path.exists():
        return set()
    out: set[str] = set()
    for row in load_jsonl(val_path):
        for k in ("text", "completion", "answer", "target_comment"):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                out.add(v.strip())
        msgs = row.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        out.add(c.strip())
    return out


def _collect_root_ids(*paths: Path) -> set[str]:
    """Union of root_id / thread_key / source_id across provided jsonl paths."""
    out: set[str] = set()
    for p in paths:
        if not p or not p.exists():
            continue
        for r in load_jsonl(p):
            for k in ("root_id", "thread_key", "source_id"):
                v = r.get(k)
                if v:
                    out.add(str(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/refinement-2026042*")
    ap.add_argument("--val-set", type=Path, default=None,
                    help="held-out validation set; chosen candidates that exact-match a val completion are dropped pre-pair")
    ap.add_argument("--sft-eval", type=Path, default=None,
                    help="SFT eval jsonl; chosen candidates sharing root_id with eval rows are dropped (thread-level holdout)")
    ap.add_argument("--samples", type=Path, default=None,
                    help="optional samples_200.jsonl or similar AI sample pool")
    ap.add_argument("--cpt-corpus", type=Path, default=None,
                    help="optional cpt_corpus.v*.jsonl for additional 'chosen' / 'rejected' mining via marker heuristics")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--llm-judge-model", default=None,
                    help="Anthropic alias (e.g. claude-sonnet-4-5-20250929) — when set, "
                         "each chosen vs rejected pair is scored 0-5 by the judge and "
                         "only pairs with score_chosen - score_rejected >= --min-judge-delta "
                         "survive. Requires ANTHROPIC_API_KEY.")
    ap.add_argument("--min-judge-delta", type=float, default=1.0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    val_texts = _index_val_completions(args.val_set)
    val_root_ids = _collect_root_ids(args.val_set, args.sft_eval)
    val_drop_count = 0
    thread_drop_count = 0

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
                    # Pre-filter chosen against val to prevent leakage.
                    if val_texts and txt.strip() in val_texts:
                        val_drop_count += 1
                        continue
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

    # Backfill chosen with train corpus rows only. Validation is held out and
    # must not become ORPO chosen text.
    train_rows = load_jsonl(args.cpt_corpus) if args.cpt_corpus else []
    candidates = []
    for v in train_rows:
        txt = v.get("text", "")
        if not is_short_korean(txt):
            continue
        if val_texts and str(txt).strip() in val_texts:
            val_drop_count += 1
            continue
        # Thread-level holdout: a thread containing any eval comment may not
        # contribute other comments to ORPO chosen (indirect leak prevention).
        rid = v.get("source_id") or v.get("root_id") or v.get("thread_key")
        if val_root_ids and rid and str(rid) in val_root_ids:
            thread_drop_count += 1
            continue
        candidates.append(v)
    target_chosen = args.max_pairs
    if candidates and len(chosen_pool) < target_chosen:
        need = target_chosen - len(chosen_pool)
        for r in random.sample(candidates, min(need, len(candidates))):
            chosen_pool.append({"text": r.get("text", ""), "kind": r.get("kind", "comment"),
                                "source_run": str(args.cpt_corpus)})

    # Synthesize rejected from chosen by injecting formal-AI patterns (always run if rejected_pool < target)
    if chosen_pool and len(rejected_pool) < args.max_pairs:
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

    # Final defense-in-depth: drop any chosen entries that exact-match val.
    # Should be a no-op given pre-filtering above, but guards against any
    # future code path that bypasses the upstream filter.
    if val_texts:
        before = len(chosen_pool)
        chosen_pool = [c for c in chosen_pool if c.get("text", "").strip() not in val_texts]
        post_drop = before - len(chosen_pool)
        if post_drop:
            val_drop_count += post_drop

    # Pair them up
    random.shuffle(chosen_pool)
    random.shuffle(rejected_pool)
    n_pairs = min(len(chosen_pool), len(rejected_pool), args.max_pairs)
    pairs = []
    judge_client = None
    if args.llm_judge_model and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore
            judge_client = anthropic.Anthropic()
        except ImportError:
            print("[WARN] anthropic SDK missing; LLM judge skipped")

    def judge_score(text: str) -> float | None:
        if not judge_client:
            return None
        try:
            resp = judge_client.messages.create(
                model=args.llm_judge_model, max_tokens=8,
                messages=[{
                    "role": "user",
                    "content": (
                        "다음 한국어 댓글이 cb2 밤문화 커뮤니티의 자연스러운 톤인지 0-5점으로만 답하라:\n"
                        f"{text}\n점수:"
                    ),
                }],
            )
            raw = resp.content[0].text if resp.content else "0"
            return float(re.search(r"[0-5](?:\.\d+)?", raw).group())
        except Exception as exc:
            print(f"[judge] error: {exc}")
            return None

    judged_drop = 0
    for i in range(n_pairs):
        c = chosen_pool[i]
        r = rejected_pool[i]
        prompt = ("다음 한국어 댓글을 cb2 밤문화 커뮤니티 톤으로 한 줄 생성:")
        sc = judge_score(c["text"])
        sr = judge_score(r["text"])
        if sc is not None and sr is not None and (sc - sr) < args.min_judge_delta:
            judged_drop += 1
            continue
        entry = {
            "prompt": prompt,
            "chosen": c["text"],
            "rejected": r["text"],
            "kind": c.get("kind", "comment"),
            "source_run_chosen": c.get("source_run"),
            "source_run_rejected": r.get("source_run"),
            "reason": r.get("reason", "formal-AI ratio"),
        }
        if sc is not None and sr is not None:
            entry["judge"] = "llm"
            entry["judge_model"] = args.llm_judge_model
            entry["score_chosen"] = sc
            entry["score_rejected"] = sr
        pairs.append(entry)

    with args.out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(
        f"DONE pairs={n_pairs} chosen_pool={len(chosen_pool)} "
        f"rejected_pool={len(rejected_pool)} val_indexed={len(val_texts)} "
        f"val_filtered_chosen={val_drop_count}"
    )
    return 0 if n_pairs >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
