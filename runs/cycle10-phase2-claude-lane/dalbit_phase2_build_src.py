#!/usr/bin/env python3
"""Phase 2: build sft-style source rows from threads_v3 seed corpus for boost-4 regen.

Each seed comment becomes one row: the boost-4 model regenerates the comment
conditioned on the seed thread (title/body) + persona. Personas are sampled
from the real sft_thread_conditioned.jsonl distribution, deterministically per
thread id so reruns are stable.
"""
import json
import random
import re
import sys
import zlib
from pathlib import Path

REPO = Path("/Users/unoa/dalbitalba-train-data")
SEED = REPO / "runs/cycle10-phase1-claude-direct/threads_v3.jsonl"
SFT = REPO / "sft_thread_conditioned.jsonl"

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/dalbit_phase2_src_smoke.jsonl")

# Real persona lines from training data, with their empirical frequency
persona_re = re.compile(r"\[PERSONA: ([^\]]+)\]")
personas = []
with SFT.open() as f:
    for line in f:
        row = json.loads(line)
        m = persona_re.search(row.get("input", ""))
        if m:
            personas.append(m.group(1))
print(f"persona pool: {len(personas)} rows, {len(set(personas))} unique")

rows_out = []
with SEED.open() as f:
    threads = [json.loads(l) for l in f]

for th in threads:
    comments = th.get("comments") or []
    if not comments:
        continue
    rng = random.Random(zlib.crc32(th["id"].encode()) ^ 0x9E2)
    for ci, c in enumerate(comments):
        persona = rng.choice(personas)
        instruction = f"[POST-TITLE] {th['title']}\n[POST-BODY] {th['content']}"
        input_text = f"[CONTEXT]\n(no parent)\n[REPLY-DEPTH=1]\n[PERSONA: {persona}]"
        rows_out.append({
            "instruction": instruction,
            "input": input_text,
            "output": c["content"],
            "kind": "comment",
            "depth": 1,
            "root_id": th["id"],
            "parent_id": f"{th['id']}:[root]",
            "persona_id": persona.split(" | ")[0],
            "comment_index": ci,
            "loss_weight": 1.0,
        })
        if len(rows_out) >= limit:
            break
        if limit <= 20:
            break  # smoke mode: spread across threads, 1 comment each
    if len(rows_out) >= limit:
        break

with out_path.open("w") as f:
    for r in rows_out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows_out)} rows -> {out_path}")
