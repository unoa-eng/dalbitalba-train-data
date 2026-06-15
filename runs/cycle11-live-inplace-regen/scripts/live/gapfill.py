#!/usr/bin/env python3
"""One-time gap backfill: fill the quiet window [last_post .. now) on the growth curve so the
timeline is continuous before the live daemon takes over. Uses the SAME producer (Codex) and
KST hour weighting, but BACKDATES created_at across the gap (not real-time).

Idempotent via marker file .gapfilled. Run once: gapfill.py
"""
import json, re, os, sys, time, math, random, subprocess, urllib.request, datetime as dt
from pathlib import Path

HERE = Path(__file__).parent
ENV = "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"); KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
TENANT = d.get("NEXT_PUBLIC_TENANT_ID") or "dalbitalba"
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
BUFFER = HERE / "gap_buffer.jsonl"; MARKER = HERE / ".gapfilled"
os.environ["PRODUCE_BUFFER"] = str(BUFFER)  # isolate from live daemon's buffer

POSTS_PER_DAY = 75
KSTW = [8,7,5,3,2,1,1,1,2,3,4,4,4,4,4,4,5,6,7,9,11,12,12,10]

def now(): return dt.datetime.now(dt.timezone.utc)
def post_req(table, rows):
    h = dict(HDR); h["Prefer"] = "return=representation"
    req = urllib.request.Request(f"{URL}/rest/v1/{table}", data=json.dumps(rows).encode(), headers=h, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read() or "[]")
def get_req(p):
    req = urllib.request.Request(f"{URL}/rest/v1/{p}", headers={"apikey":KEY,"Authorization":f"Bearer {KEY}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read() or "[]")

def hour_weighted_times(start, end, n, rng):
    """sample n timestamps in [start,end] with KST hour weighting."""
    out = []
    span = (end - start).total_seconds()
    while len(out) < n:
        t = start + dt.timedelta(seconds=rng.random()*span)
        kh = (t.hour + 9) % 24
        if rng.random() < KSTW[kh]/max(KSTW):
            out.append(t)
    return sorted(out)

def buffer_lines():
    return BUFFER.read_text().splitlines() if BUFFER.exists() else []

def ensure_buffer(min_n):
    while len(buffer_lines()) < min_n:
        subprocess.run([sys.executable, str(HERE/"produce.py"), "14", str(time.time_ns())], timeout=900, capture_output=True)

def main():
    if MARKER.exists():
        print("gap already filled; skip"); return
    # anchor on the pre-live cluster: latest post OLDER than 2h (ignore live-daemon posts)
    from urllib.parse import quote
    cutoff = quote((now() - dt.timedelta(hours=2)).isoformat(), safe="")
    latest = get_req(f"community_posts?created_at=lt.{cutoff}&select=created_at&order=created_at.desc&limit=1")
    last_t = dt.datetime.fromisoformat(latest[0]["created_at"].replace("Z","+00:00")) if latest else now()-dt.timedelta(days=3)
    end = now() - dt.timedelta(hours=2)
    gap_days = (end - last_t).total_seconds()/86400
    if gap_days < 0.3:
        print(f"gap only {gap_days:.2f}d; nothing to fill"); MARKER.write_text("noop"); return
    n = int(gap_days * POSTS_PER_DAY)
    print(f"gap {gap_days:.2f}d -> backfilling ~{n} posts [{last_t} .. {end}]")
    rng = random.Random()
    times = hour_weighted_times(last_t, end, n, rng)
    filled = 0
    for i, t in enumerate(times):
        ensure_buffer(1)
        lines = buffer_lines()
        if not lines: print("buffer empty, stop"); break
        p = json.loads(lines[0]); BUFFER.write_text("\n".join(lines[1:]) + ("\n" if len(lines)>1 else ""))
        age_days = (now()-t).total_seconds()/86400
        views = max(1, int(rng.gauss(1822, 700) * min(1.0, age_days/5)))  # older gap posts have more views
        row = {"tenant_id":TENANT,"user_id":None,"category":p.get("category","FREE"),"title":p["title"],"body":p["body"],
               "is_anon":True,"source_author":None,"view_count":views,"like_count":rng.choice([0,0,0,1]),
               "created_at":t.isoformat(),"updated_at":t.isoformat()}
        try:
            pid = post_req("community_posts",[row])[0]["id"]
        except Exception as e:
            print(f"  post fail: {str(e)[:50]}"); continue
        # comments backdated after post, lognormal delay
        crows=[]; rel={}
        for idx,c in enumerate(p.get("comments") or []):
            dly=min(math.exp(rng.gauss(math.log(3),1.0)),24*8)  # hours, median 3h
            ct=t+dt.timedelta(hours=dly)
            if ct>=now(): ct=now()-dt.timedelta(minutes=rng.randint(1,120))
            pidx=c.get("parent_index"); parent=rel.get(pidx)
            r={"tenant_id":TENANT,"post_id":pid,"user_id":None,"parent_id":parent,"body":c["body"],
               "is_anon":True,"source_author":None,"created_at":ct.isoformat()}
            try:
                cid=post_req("community_comments",[r])[0]["id"]; rel[idx]=cid
            except Exception: pass
        filled += 1
        if filled % 20 == 0: print(f"  backfilled {filled}/{n}")
    MARKER.write_text(f"filled {filled} at {now().isoformat()}")
    print(f"GAPFILL DONE: {filled} posts backfilled")

if __name__ == "__main__":
    main()
