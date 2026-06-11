#!/usr/bin/env python3
"""Phase 2 metadata 고도화: views/likes/댓글타이밍/crawledAt을 원천 분포에 맞춰 재생성.

- views: cpt_enriched.jsonl 실측 (comment_count 조건부 empirical) 리샘플
  + 사이트 성장 ramp (24-10 시작, day 285 peak 기준 audience factor)
- likes: views 대비 1-2% 수준 binomial-ish, 대부분 0-3
- 댓글 타이밍: 글 게시시각(저녁/심야 가중) + lognormal 딜레이 누적, 인덱스 순서 보장
- crawledAt: 2026-06-10 단일 크롤 스윕, 최신글부터 역순 페이지네이션 시뮬
- commentCount: len(comments)로 재계산
모든 난수는 thread id 시드 → 재현 가능. 형식(문자열 숫자, 날짜 포맷)은 seed 그대로 유지.
"""
import json
import math
import random
import sys
import zlib
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path("/Users/unoa/dalbitalba-train-data")
SITE_START = datetime(2024, 10, 1)
CRAWL_BASE = datetime(2026, 6, 11, 0, 30, 0)

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

# --- source empirical views by comment-count bucket ---
buckets = defaultdict(list)
for line in (REPO / "cpt_enriched.jsonl").open():
    r = json.loads(line)
    if r.get("kind") == "post" and isinstance(r.get("views"), int) and isinstance(r.get("comment_count"), int):
        buckets[min(r["comment_count"], 10)].append(r["views"])
for b in buckets:
    buckets[b].sort()

def sample_views(rng, cc, post_dt):
    pool = buckets.get(min(cc, 10)) or buckets[4]
    v = pool[rng.randrange(len(pool))]
    # site-growth audience factor: 작은 신생 사이트는 초기 조회수 낮음
    days = (post_dt - SITE_START).days
    factor = min(1.0, 0.25 + 0.75 * days / 285.0)
    # 오래된 글일수록 누적 조회 약간 가산 (크롤 시점까지 노출 기간)
    age_days = (CRAWL_BASE - post_dt).days
    accum = 1.0 + min(0.25, age_days / 2000.0)
    v = max(2, int(round(v * factor * accum * rng.uniform(0.8, 1.25))))
    return v

def sample_likes(rng, views):
    # 대부분 0~3, views 비례 약한 상관
    p = rng.choice([0.0, 0.0, 0.005, 0.01, 0.015, 0.02, 0.04])
    likes = 0
    if p > 0:
        likes = sum(1 for _ in range(views) if rng.random() < p)
    return min(likes, max(0, views // 4))

def post_time(rng, date_str):
    # 커뮤니티 특성: 저녁~심야 가중 게시
    hour = rng.choices(
        population=[1, 2, 3, 4, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0],
        weights=[6, 5, 4, 3, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 7],
    )[0]
    minute = rng.randrange(60)
    sec = rng.randrange(60)
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.replace(hour=hour, minute=minute, second=sec)

def comment_times(rng, post_dt, n, crawl_dt):
    """첫 댓글 lognormal(median~1.2h), 이후 gap은 점점 길어지는 tail. 정렬 보장."""
    times = []
    t = post_dt
    for i in range(n):
        if i == 0:
            delay_h = math.exp(rng.gauss(0.0, 1.2))       # median ~1h, tail 수일
        else:
            delay_h = math.exp(rng.gauss(-1.2, 1.4)) * (1 + i * 0.2)
        t = t + timedelta(hours=min(delay_h, 24 * 4))
        times.append(t)
    times.sort()
    # 크롤 시점 이후로 넘어간 댓글은 글~크롤 사이로 클램프 (순서 유지)
    span = (crawl_dt - post_dt).total_seconds()
    fixed = []
    prev = post_dt
    for x in times:
        if x >= crawl_dt:
            x = post_dt + timedelta(seconds=span * rng.uniform(0.5, 0.98))
        if x <= prev:
            x = prev + timedelta(seconds=rng.randrange(30, 1800))
        if x >= crawl_dt:
            x = crawl_dt - timedelta(seconds=rng.randrange(60, 600))
            if x <= prev:
                x = prev + timedelta(seconds=30)
        fixed.append(x)
        prev = x
    # 최종 보장: crawl 이전 + 순증가 (좁은 span 엣지케이스)
    prev = post_dt
    for i, x in enumerate(fixed):
        if x >= crawl_dt or x <= prev:
            x = prev + (crawl_dt - prev) / 2
        fixed[i] = x
        prev = x
    return fixed

threads = [json.loads(l) for l in in_path.open()]
# crawledAt: 최신글부터 역순 스윕 (board pagination), 글당 2-6초
threads_sorted = sorted(threads, key=lambda t: (t["date"], int(t["id"])), reverse=True)
crawl_at = {}
cur = CRAWL_BASE
for t in threads_sorted:
    rng = random.Random(zlib.crc32(t["id"].encode()) ^ 0xCA)
    cur = cur + timedelta(seconds=rng.uniform(2, 6))
    crawl_at[t["id"]] = cur

for t in threads:
    rng = random.Random(zlib.crc32(t["id"].encode()) ^ 0x9D7)
    comments = t.get("comments") or []
    cc = len(comments)
    pdt = post_time(rng, t["date"])
    cdt = crawl_at[t["id"]]
    views = sample_views(rng, cc, pdt)
    likes = sample_likes(rng, views)
    t["views"] = str(views)
    t["likes"] = str(likes)
    t["commentCount"] = str(cc)
    t["crawledAt"] = cdt.strftime("%Y-%m-%dT%H:%M:%S")
    for c, ct in zip(comments, comment_times(rng, pdt, cc, cdt)):
        c["date"] = ct.strftime("%Y-%m-%d %H:%M:%S")

with out_path.open("w") as f:
    for t in threads:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

import statistics
vs = [int(t["views"]) for t in threads]
ls = [int(t["likes"]) for t in threads]
print(f"refined {len(threads)} threads -> {out_path}")
print("views: mean", round(statistics.mean(vs), 1), "median", statistics.median(vs),
      "p90", sorted(vs)[len(vs) * 9 // 10], "max", max(vs))
print("likes: mean", round(statistics.mean(ls), 2), "median", statistics.median(ls), "max", max(ls))
