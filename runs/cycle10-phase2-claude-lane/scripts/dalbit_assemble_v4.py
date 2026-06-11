#!/usr/bin/env python3
"""threads_v4 최종 조립: seed v3 + 게시글 rework + Claude 댓글(early+bulk) + 메타데이터 고도화.

순서:
1. seed threads_v3 로드
2. 게시글 rework (/tmp/dalbit_post_out/out*.jsonl) 적용 — title/content 교체
3. 댓글 교체: early(/tmp/dalbit_phase2_early_merged.jsonl) + bulk(/tmp/dalbit_bulk_out/out*.jsonl)
   → content = "[i+1] " + text (원천 인덱스 prefix 양식)
4. 메타데이터 고도화는 별도 단계(dalbit_metadata_refine.py)로 체인
출력: 중간본 (메타데이터 전) — refine까지 돌리려면:
  python3 dalbit_assemble_v4.py /tmp/dalbit_threads_v4_pre.jsonl
  python3 dalbit_metadata_refine.py /tmp/dalbit_threads_v4_pre.jsonl <final>
검증 리포트 출력: 교체율, 누락 슬롯, 제목/본문/댓글 uniqueness.
"""
import json
import glob
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/unoa/dalbitalba-train-data")
SEED = REPO / "runs/cycle10-phase1-claude-direct/threads_v3.jsonl"
out_path = Path(sys.argv[1])

threads = [json.loads(l) for l in SEED.open()]
by_id = {t["id"]: t for t in threads}

# 1. 게시글 rework 적용
post_n = 0
for f in sorted(glob.glob("/tmp/dalbit_post_out/out*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        t = by_id.get(str(r["id"]))
        if t is None:
            continue
        t["title"] = r["title"]
        t["content"] = r["content"]
        post_n += 1

# 2. 댓글 교체 (early + bulk)
cmt = {}
for f in ["/tmp/dalbit_phase2_early_merged.jsonl"] + sorted(glob.glob("/tmp/dalbit_bulk_out/out*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        cmt[(str(r["root_id"]), int(r["comment_index"]))] = r["content"].strip()

replaced = missing = 0
for t in threads:
    for ci, c in enumerate(t.get("comments") or []):
        new = cmt.get((t["id"], ci))
        if new:
            c["content"] = f"[{ci + 1}] {new}"
            replaced += 1
        else:
            missing += 1

with out_path.open("w") as f:
    for t in threads:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

tc = Counter(t["title"] for t in threads)
cc = Counter(t["content"] for t in threads)
mc = Counter(c["content"] for t in threads for c in t.get("comments") or [])
total_c = sum(mc.values())
print(json.dumps({
    "threads": len(threads),
    "posts_reworked": post_n,
    "comments_replaced": replaced,
    "comments_missing_slot": missing,
    "title_unique_pct": round(len(tc) / len(threads) * 100, 1),
    "content_unique_pct": round(len(cc) / len(threads) * 100, 1),
    "comment_unique_pct": round(len(mc) / max(1, total_c) * 100, 1),
}, ensure_ascii=False))
