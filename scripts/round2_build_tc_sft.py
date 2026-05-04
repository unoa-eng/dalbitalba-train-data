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
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

random.seed(7)

ARGOT_TERMS = [
    "TC", "티씨", "밀빵", "쩜오", "텐카", "초이스", "케어", "갯수", "마담", "퍼블",
    "보도", "하퍼", "도파민", "셔츠룸", "빠꾸", "시다", "텐", "노도", "골타", "뺑이",
]
ARGOT_RE = re.compile("|".join(re.escape(t) for t in ARGOT_TERMS))
REPLY_PREFIX = re.compile(r"^\s*\[(\d+(?:-\d+)*)\]\s*")


def reply_depth(text: str) -> int:
    m = REPLY_PREFIX.match(text or "")
    if not m:
        return 0
    return m.group(1).count("-") + 1  # depth-1 = direct reply, depth-2 = chain


def argot_count(text: str) -> int:
    return len(set(ARGOT_RE.findall(text or "")))


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
        return {"persona_id": "p-019", "tone": "혼용", "mood": "seeking_empathy"}
    p = random.choice(personas)
    return {
        "persona_id": f"p-{p.get('id'):03d}" if isinstance(p.get("id"), int) else "p-019",
        "tone": p.get("tone") or ("반말" if has_argot else "혼용"),
        "mood": p.get("mood") or "seeking_empathy",
    }


def build_row(
    post: dict[str, Any],
    comment_key: str,
    raw: dict[str, Any],
    personas: list[dict[str, Any]],
    *,
    argot_threshold: int,
    argot_weight: float,
) -> dict[str, Any] | None:
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
    cells = [instruction, input_str, output_text]
    has_argot = any(argot_count(c) >= argot_threshold for c in cells)
    loss_weight = argot_weight if has_argot else 1.0

    return {
        "instruction": instruction,
        "input": input_str,
        "output": output_text,
        "kind": "comment",
        "depth": depth,
        "root_id": post.get("source_id"),
        "parent_id": f"{post.get('source_id')}:[{target.get('parent_key') or 'root'}]",
        "persona_id": persona["persona_id"],
        "loss_weight": loss_weight,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-stream", required=True, type=Path)
    ap.add_argument("--raw-source-dir", required=True, type=Path)
    ap.add_argument("--persona-list", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--argot-threshold", type=int, default=2)
    ap.add_argument("--argot-weight", type=float, default=1.5)
    args = ap.parse_args()

    if not args.context_stream.exists():
        print(f"context-stream missing: {args.context_stream}", file=sys.stderr)
        return 2
    if not args.raw_source_dir.is_dir():
        print(f"raw source-dir missing: {args.raw_source_dir}", file=sys.stderr)
        return 2

    print("indexing raw cb2 JSON...", flush=True)
    raw_by_id = index_raw(args.raw_source_dir)
    print(f"  {len(raw_by_id)} posts indexed", flush=True)

    personas = load_personas(args.persona_list)
    print(f"  {len(personas)} personas loaded", flush=True)

    written = 0
    weighted = 0
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.context_stream.open("r", encoding="utf-8") as src, \
            args.out.open("w", encoding="utf-8") as dst:
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
            row = build_row(
                obj,
                comment_key,
                raw_by_id,
                personas,
                argot_threshold=args.argot_threshold,
                argot_weight=args.argot_weight,
            )
            if not row:
                skipped += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if row.get("loss_weight", 1.0) > 1.0:
                weighted += 1
            if args.max_rows and written >= args.max_rows:
                break

    print(f"DONE written={written} weighted={weighted} skipped={skipped}")
    print(f"weighted_ratio={weighted / max(written, 1):.3f}")
    return 0 if written >= 100 else 1


if __name__ == "__main__":
    sys.exit(main())
