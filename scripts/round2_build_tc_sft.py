#!/usr/bin/env python3
"""round2_build_tc_sft.py — Reshape cpt_context_stream.jsonl into the
Thread-Conditioned SFT schema described in TRAINING_DESIGN.md §4.

Input:
  cpt_context_stream.jsonl rows:
    {text, kind: "context_comment"|"post", source_id, source_field,
     length_bucket, context_mode, comment_key, parent_comment_key}
  Plus the raw cb2 JSON batches (for parent context lookup) at source-dir.

Output (jsonl):
  {instruction, input, output, kind, depth, root_id, parent_id, persona_id, loss_weight}

  - instruction: "[POST-TITLE] {title}\n[POST-BODY] {body}"
  - input:       "[CONTEXT]\n{parent_chain}\n[REPLY-DEPTH={d}]\n[PERSONA: {pid} | {tone} | {mood}]"
  - output:      "{depth-d reply text}"
  - loss_weight: 1.5 if any cell contains >=2 distinct argot tokens, else 1.0

Usage:
  python3 scripts/round2_build_tc_sft.py \\
      --context-stream cpt_context_stream.jsonl \\
      --raw-source-dir source_db_cache \\
      --persona-list runs/round2-obsidian-synthesis/persona-30-extracted.json \\
      --out sft_thread_conditioned.jsonl

Acceptance: at least 8000 rows produced; argot-weighted rows >= 5%.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the global MinHash dedup helper (R2 follow-up #1 — closes the duality
# bug where chain_train_round2.sh Phase 3 ran without dedup while
# build_thread_aware_datasets.py applied it).
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from dedup_minhash import dedup_records  # noqa: E402

random.seed(7)

# ---------------------------------------------------------------------------
# Argot heuristic — env-driven so chain_train_round2.sh / round2_mutator.py
# escalations actually reach the live Phase-3 builder. Mirrors the contract
# documented in scripts/build_thread_aware_datasets.py:90-150.
#
# Env vars (opt-in override of defaults):
#   SFT_LOSS_WEIGHT_ARGOT      numeric weight applied when threshold met
#                              (default 1.5; matches chain_train_round2.sh)
#   SFT_LOSS_WEIGHT_THRESHOLD  minimum distinct argot tokens to trigger
#                              (default 2; legacy default)
#   SFT_LOSS_WEIGHT_TERMS      comma-separated override for the 20-term default
#                              keyword list (optional).
# ---------------------------------------------------------------------------
DEFAULT_ARGOT_TERMS = [
    "TC", "티씨", "밀빵", "쩜오", "텐카", "초이스", "케어", "갯수", "마담", "퍼블",
    "보도", "하퍼", "도파민", "셔츠룸", "빠꾸", "시다", "텐", "노도", "골타", "뺑이",
]


def _load_argot_config() -> tuple[re.Pattern, int, float, list[str]]:
    raw_terms = os.environ.get("SFT_LOSS_WEIGHT_TERMS", "").strip()
    if raw_terms:
        terms = [t.strip() for t in raw_terms.split(",") if t.strip()]
    else:
        terms = list(DEFAULT_ARGOT_TERMS)
    try:
        threshold = int(os.environ.get("SFT_LOSS_WEIGHT_THRESHOLD", "2"))
    except ValueError:
        threshold = 2
    try:
        weight = float(os.environ.get("SFT_LOSS_WEIGHT_ARGOT", "1.5"))
    except ValueError:
        weight = 1.5
    if threshold < 1:
        threshold = 1
    if weight < 1.0:
        weight = 1.0
    pattern = re.compile("|".join(re.escape(t) for t in terms))
    return pattern, threshold, weight, terms


ARGOT_RE, ARGOT_THRESHOLD, ARGOT_WEIGHT, ARGOT_TERMS = _load_argot_config()
REPLY_PREFIX = re.compile(r"^\s*\[(\d+(?:-\d+)*)\]\s*")


def reply_depth(text: str) -> int:
    m = REPLY_PREFIX.match(text or "")
    if not m:
        return 0
    return m.group(1).count("-") + 1  # depth-1 = direct reply, depth-2 = chain


def argot_count(text: str) -> int:
    return len(set(ARGOT_RE.findall(text or "")))


def compute_loss_weight(*cells: str | None) -> float:
    """Return ARGOT_WEIGHT iff distinct-argot count across cells >= threshold."""
    seen: set[str] = set()
    for cell in cells:
        if not cell:
            continue
        seen.update(ARGOT_RE.findall(cell))
    return ARGOT_WEIGHT if len(seen) >= ARGOT_THRESHOLD else 1.0


def load_personas(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("personas") or []


def index_raw(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """post_id -> {title, body, comments_by_key: {key: {text, parent_key}}}"""
    by_id: dict[str, dict[str, Any]] = {}
    for fp in sorted(raw_dir.glob("cb2_*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {fp.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            continue
        for post in data:
            if not isinstance(post, dict):
                continue
            pid = str(post.get("id") or "")
            title = (post.get("title") or "").strip()
            body = (post.get("content") or "").strip()
            comments_by_key: dict[str, dict[str, Any]] = {}
            for c in post.get("comments") or []:
                if not isinstance(c, dict):
                    continue
                ctext = (c.get("content") or "").strip()
                m = REPLY_PREFIX.match(ctext)
                key = m.group(1) if m else "0"
                parent_key = "-".join(key.split("-")[:-1]) if "-" in key else None
                comments_by_key[key] = {"text": ctext, "parent_key": parent_key}
            by_id[pid] = {"title": title, "body": body, "comments_by_key": comments_by_key}
    return by_id


def assign_persona(personas: list[dict[str, Any]], length: int, has_argot: bool) -> dict[str, str]:
    if not personas:
        return {
            "persona_id": "p-019",
            "persona": "default",
            "tone": "혼용",
            "mood": "seeking_empathy",
        }
    p = random.choice(personas)
    return {
        "persona_id": f"p-{p.get('id'):03d}" if isinstance(p.get("id"), int) else "p-019",
        "persona": str(p.get("name") or "default"),
        "tone": p.get("tone") or ("반말" if has_argot else "혼용"),
        "mood": p.get("mood") or "seeking_empathy",
    }


def build_row(post: dict[str, Any], comment_key: str, raw: dict[str, Any],
              personas: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw_post = raw.get(post.get("source_id") or "")
    if not raw_post:
        return None
    cmts = raw_post.get("comments_by_key", {})
    target = cmts.get(comment_key)
    if not target:
        return None
    output_text = target["text"]
    if not output_text:
        return None
    depth = reply_depth(output_text)

    parent_chain: list[str] = []
    cur_parent_key = target["parent_key"]
    while cur_parent_key:
        parent = cmts.get(cur_parent_key)
        if not parent:
            break
        parent_chain.append(parent["text"])
        cur_parent_key = parent["parent_key"]
    parent_chain.reverse()

    title = raw_post["title"]
    body = raw_post["body"]
    if not (title or body):
        return None

    instruction = f"[POST-TITLE] {title}\n[POST-BODY] {body}"
    context_block = "\n".join(parent_chain) if parent_chain else "(no parent)"
    persona = assign_persona(personas, len(output_text), bool(argot_count(output_text)))
    input_str = (
        f"[CONTEXT]\n{context_block}\n"
        f"[REPLY-DEPTH={depth}]\n"
        f"[PERSONA: {persona['persona_id']} | {persona['tone']} | {persona['mood']}]"
    )
    # Env-driven argot heuristic (R2 follow-up #1): use distinct argot count
    # across all four cells (post-title, post-body, parent-context, target reply)
    # so SFT_LOSS_WEIGHT_ARGOT/THRESHOLD/TERMS escalations from
    # chain_train_round2.sh actually take effect on the live Phase-3 corpus.
    loss_weight = compute_loss_weight(title, body, context_block, output_text)

    return {
        "instruction": instruction,
        "input": input_str,
        "output": output_text,
        "kind": "comment",
        "depth": depth,
        "root_id": post.get("source_id"),
        "parent_id": f"{post.get('source_id')}:[{target.get('parent_key') or 'root'}]",
        "persona_id": persona["persona_id"],
        "persona": persona["persona"],
        "loss_weight": loss_weight,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-stream", required=True, type=Path)
    ap.add_argument("--raw-source-dir", required=True, type=Path)
    ap.add_argument("--persona-list", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument(
        "--apply-dedup",
        dest="apply_dedup",
        action="store_true",
        default=True,
        help="apply MinHash near-dedup on target reply text (default ON)",
    )
    ap.add_argument(
        "--no-apply-dedup",
        dest="apply_dedup",
        action="store_false",
        help="disable MinHash near-dedup (debug / ablation)",
    )
    ap.add_argument(
        "--dedup-field",
        default="output",
        help="JSONL key whose text drives MinHash dedup (default: output)",
    )
    ap.add_argument("--dedup-num-perm", type=int, default=128)
    ap.add_argument("--dedup-bands", type=int, default=32)
    ap.add_argument("--dedup-shingle-n", type=int, default=5)
    ap.add_argument("--dedup-seed", type=int, default=42)
    args = ap.parse_args()

    if not args.context_stream.exists():
        print(f"context-stream missing: {args.context_stream}", file=sys.stderr)
        return 2
    if not args.raw_source_dir.is_dir():
        print(f"raw source-dir missing: {args.raw_source_dir}", file=sys.stderr)
        return 2

    print(
        f"argot config: weight={ARGOT_WEIGHT} threshold={ARGOT_THRESHOLD} "
        f"terms={len(ARGOT_TERMS)} (env-driven via SFT_LOSS_WEIGHT_*)",
        flush=True,
    )

    print("indexing raw cb2 JSON...", flush=True)
    raw_by_id = index_raw(args.raw_source_dir)
    print(f"  {len(raw_by_id)} posts indexed", flush=True)

    personas = load_personas(args.persona_list)
    print(f"  {len(personas)} personas loaded", flush=True)

    # Phase A: build all rows in memory so we can run global MinHash dedup.
    rows: list[dict[str, Any]] = []
    skipped = 0
    with args.context_stream.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if obj.get("kind") != "context_comment":
                continue  # only comments fit TC-SFT schema
            comment_key = obj.get("comment_key") or "0"
            row = build_row(obj, comment_key, raw_by_id, personas)
            if not row:
                skipped += 1
                continue
            rows.append(row)
            if args.max_rows and len(rows) >= args.max_rows:
                break

    pre_dedup = len(rows)
    dedup_stats: dict[str, Any] = {}
    if args.apply_dedup and rows:
        rows, dedup_stats = dedup_records(
            rows,
            field=args.dedup_field,
            num_perm=args.dedup_num_perm,
            bands=args.dedup_bands,
            shingle_n=args.dedup_shingle_n,
            seed=args.dedup_seed,
        )
        print(
            f"[minhash] field={args.dedup_field} input={pre_dedup:,} "
            f"kept={len(rows):,} dropped={pre_dedup - len(rows):,} "
            f"({(pre_dedup - len(rows)) / max(1, pre_dedup) * 100:.1f}%) "
            f"biggest_cluster={dedup_stats.get('biggest_cluster', 0)} "
            f"clusters_5plus={dedup_stats.get('clusters_5plus', 0)}",
            flush=True,
        )
    elif not args.apply_dedup:
        print("[minhash] skipped (--no-apply-dedup)", flush=True)

    # Phase B: emit kept rows.
    weighted = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            if float(row.get("loss_weight", 1.0) or 1.0) > 1.0:
                weighted += 1

    written = len(rows)
    print(
        f"DONE written={written} weighted={weighted} skipped={skipped} "
        f"pre_dedup={pre_dedup}"
    )
    print(f"weighted_ratio={weighted / max(written, 1):.3f}")
    return 0 if written >= 100 else 1


if __name__ == "__main__":
    sys.exit(main())
