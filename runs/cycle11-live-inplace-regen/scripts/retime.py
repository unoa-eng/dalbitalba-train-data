#!/usr/bin/env python3
"""Re-time created_at to a 2024-Q4 -> 2026-06 EXPONENTIAL growth curve.

- posts: density grows exponentially toward present (24Q4 start, exp ramp).
- KST evening/late-night hour weighting preserved.
- comments: created_at = post_t + lognormal delay (median ~3h), >= post, <= now.
- deterministic (seeded by id). Structure read from batch_in (no extra DB fetch).

Usage: retime.py --dry   (preview monthly/hour distribution, no writes)
       retime.py          (live PATCH created_at via PostgREST, threaded+resilient)
"""
import json, re, sys, glob, math, random, zlib, datetime as dt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

ENV = "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"); KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Prefer": "return=minimal"}
BASE = Path("/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen")

START = dt.datetime(2024, 10, 1, tzinfo=dt.timezone.utc)      # 24 Q4 start
NOW   = dt.datetime(2026, 6, 12, 9, 0, tzinfo=dt.timezone.utc) # present cap (~today)
SPAN  = (NOW - START).total_seconds()
LAMBDA = 3.6   # exponential steepness (higher -> more recent-heavy)
# KST hour weights (evening/late-night heavy); index = KST hour
KSTW = [8,7,5,3,2,1,1,1,2,3,4,4,4,4,4,4,5,6,7,9,11,12,12,10]

def seed(s): return random.Random(zlib.crc32(s.encode()) ^ 0x5C11)

def post_time(pid):
    rng = seed(pid + ":t")
    u = rng.random()
    s = math.log(1 + u * (math.exp(LAMBDA) - 1)) / LAMBDA   # inverse-CDF of exp growth
    base = START + dt.timedelta(seconds=s * SPAN)
    kh = rng.choices(range(24), weights=KSTW)[0]            # KST hour
    uh = (kh - 9) % 24                                      # -> UTC hour
    t = base.replace(hour=uh, minute=rng.randint(0, 59), second=rng.randint(0, 59))
    if t > NOW: t = NOW - dt.timedelta(minutes=rng.randint(5, 600))
    return t

def cmt_time(cid, pt):
    rng = seed(cid + ":c")
    delay_h = min(math.exp(rng.gauss(math.log(3), 1.0)), 24 * 12)  # lognormal, median 3h, cap 12d
    t = pt + dt.timedelta(hours=delay_h)
    if t > NOW: t = pt + dt.timedelta(hours=rng.uniform(0.1, 6))
    if t > NOW: t = NOW
    return t

def load_structure():
    posts = {}
    for f in glob.glob(str(BASE / "batch_in/batch_*.json")):
        for p in json.load(open(f)):
            posts[p["id"]] = [c["id"] for c in p["comments"]]
    return posts

def patch(table, idv, iso):
    req = urllib.request.Request(f"{URL}/rest/v1/{table}?id=eq.{idv}",
        data=json.dumps({"created_at": iso}).encode(), headers=HDR, method="PATCH")
    urllib.request.urlopen(req, timeout=60).read()

def main():
    dry = "--dry" in sys.argv
    posts = load_structure()
    print(f"posts {len(posts)} comments {sum(len(v) for v in posts.values())}")
    tasks = []; months = Counter(); kst_hours = Counter()
    for pid, cids in posts.items():
        pt = post_time(pid)
        months[pt.strftime("%Y-%m")] += 1; kst_hours[(pt.hour + 9) % 24] += 1
        tasks.append(("community_posts", pid, pt.isoformat()))
        for cid in cids:
            tasks.append(("community_comments", cid, cmt_time(cid, pt).isoformat()))
    print("monthly distribution (24Q4 -> now):")
    for m in sorted(months): print(f"  {m}: {months[m]}")
    night = sum(kst_hours[h] for h in list(range(19,24))+list(range(0,3)))
    print(f"evening/night KST(19-03) share: {night/max(1,len(posts)):.3f}")
    if dry:
        print(f"DRY — {len(tasks)} rows would be patched."); return
    done=[0]; fails=[]
    def run(t):
        try: patch(*t); done[0]+=1
        except Exception as e: fails.append(str(e)[:30])
        if done[0] % 5000 == 0: print(f"  patched {done[0]}/{len(tasks)}")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(run, tasks))
    print(f"RETIME DONE: patched {done[0]}/{len(tasks)} | fails {len(fails)} {fails[:3]}")

if __name__ == "__main__":
    main()
