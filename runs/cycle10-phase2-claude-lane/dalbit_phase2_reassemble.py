#!/usr/bin/env python3
"""Phase 2: reassemble threads with boost-4 regenerated comments.

Reads generation samples.jsonl (from cycle9_generation_audit.py), applies
cleanup v2 (winning recipe — no v3 additions), renumbers the [N] index prefix
to the comment's actual slot, and writes threads_v4.jsonl preserving all seed
metadata/timing. Falls back to the seed comment text when a generation failed
quality gates.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/unoa/dalbitalba-train-data")
SEED = REPO / "runs/cycle10-phase1-claude-direct/threads_v3.jsonl"

samples_path = Path(sys.argv[1])          # samples.jsonl (concatenated chunks)
src_path = Path(sys.argv[2])              # the exact --source-jsonl used for generation
out_path = Path(sys.argv[3])              # threads_v4.jsonl
report_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None

INDEX_RE = re.compile(r"^\[\d+(?:-\d+)?\]\s*")
NAMES = ["하유호", "장미", "성원이", " 성원", "haai0dan", "haai", "ha1922",
         "ha1", "onoo", "상띠", "아딜", "킬리"]

def cleanup_v2_comment(text: str) -> str:
    for marker in ["\n원글:", "\n부모댓글:", "원글:", "부모댓글:"]:
        if marker in text:
            text = text.split(marker)[0]
    for name in NAMES:
        text = text.replace(name, "")
    text = re.sub(r"ᄃᄎ", "", text)
    text = re.sub(r"ᄈ\d", "", text)
    text = re.sub(r"(ᄏ)\1{5,}", r"\1\1\1\1", text)
    text = re.sub(r"(ᅲ)\1{4,}", r"\1\1\1\1", text)
    text = re.sub(r"\+\+", "", text)
    text = re.sub(r"➖+", "", text)
    text = text.replace("�", "")
    # comments are single-block on site: collapse newlines to space
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()

def usable(gen_row: dict) -> bool:
    m = gen_row.get("metrics") or {}
    if m.get("returncode", 1) != 0 or m.get("empty"):
        return False
    for bad in ("control_token_leak", "invalid_replacement_char",
                "non_korean_cjk", "meta_artifact", "reply_context_leak"):
        if m.get(bad):
            return False
    if m.get("max_repeated_char_run", 0) >= 12:
        return False
    return True

threads = [json.loads(l) for l in SEED.open()]
# source_index in samples.jsonl is 1-based over the generation source jsonl;
# each source row carries root_id + comment_index from the builder.
keymap = {}
for i, line in enumerate(src_path.open(), start=1):
    row = json.loads(line)
    keymap[i] = (str(row["root_id"]), row["comment_index"])

regen = {}
stats = {"total": 0, "usable": 0, "too_short": 0, "fallback": 0, "key_miss": 0}
for line in samples_path.open():
    row = json.loads(line)
    stats["total"] += 1
    key = keymap.get(row.get("source_index"))
    if key is None:
        stats["key_miss"] += 1
        continue
    if key[0] != str(row.get("root_id")):
        stats["key_miss"] += 1
        continue
    text = (row.get("text") or "").strip()
    text = INDEX_RE.sub("", text)
    text = cleanup_v2_comment(text)
    if usable(row) and len(text) > 3:
        regen[key] = text
        stats["usable"] += 1
    elif len(text) <= 3:
        stats["too_short"] += 1

replaced = 0
for th in threads:
    for ci, c in enumerate(th.get("comments") or []):
        new = regen.get((th["id"], ci))
        if new:
            c["content"] = f"[{ci + 1}] {new}"
            replaced += 1
        else:
            stats["fallback"] += 1

with out_path.open("w") as f:
    for th in threads:
        f.write(json.dumps(th, ensure_ascii=False) + "\n")

stats["replaced"] = replaced
print(json.dumps(stats, ensure_ascii=False))
if report_path:
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
