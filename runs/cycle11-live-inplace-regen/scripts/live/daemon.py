#!/usr/bin/env python3
"""Live operations daemon: continuously publish fresh posts + comments in REAL TIME
with randomized, KST-hour-weighted spacing (Poisson), matching the meta-graph density.

- posts: ~POSTS_PER_DAY/day, exponential inter-arrival modulated by hour weight (random, not fixed).
- comments: each post's planned comments released later via lognormal delay (real-time trickle).
- created_at = now() at insertion (genuine real-time). views grow via aging bumps.
- buffer auto-refilled by produce.py (Codex). State persisted -> restart-safe.

Run: daemon.py   (loops forever; launch under nohup). Ctrl-C / SIGTERM safe.
"""
import json, re, os, sys, time, math, random, subprocess, urllib.request, datetime as dt
from pathlib import Path

HERE = Path(__file__).parent
# persistent env (survives reboot) preferred; /tmp fallback for first run
ENV = str(HERE / ".env.prod") if (HERE / ".env.prod").exists() else "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"); KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
TENANT = d.get("NEXT_PUBLIC_TENANT_ID") or "dalbitalba"
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

BUFFER = HERE / "buffer.jsonl"
CQUEUE = HERE / "comment_queue.json"
RELEASED = HERE / "released_map.json"
GAP_STATE = HERE / "gap_state.json"
LOG = HERE / "daemon.log"
GAP_PER_TICK = 2   # backdated gap posts published per live tick (fills 3-day gap in ~1.5d, single codex)

POSTS_PER_DAY = 16
KSTW = [8,7,5,3,2,1,1,1,2,3,4,4,4,4,4,4,5,6,7,9,11,12,12,10]  # KST hour weights
AVGW = sum(KSTW)/24
BUFFER_LOW = 8
CMT_DELAY_MEDIAN_MIN = 28      # lognormal median for comment trickle (minutes)

def log(m):
    ts = dt.datetime.now().strftime("%m-%d %H:%M:%S")
    line = f"[{ts}] {m}"
    print(line, flush=True)
    with LOG.open("a") as f: f.write(line+"\n")

def now(): return dt.datetime.now(dt.timezone.utc)

def post_req(table, rows, prefer="return=representation"):
    h = dict(HDR); h["Prefer"] = prefer
    req = urllib.request.Request(f"{URL}/rest/v1/{table}", data=json.dumps(rows).encode(), headers=h, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read() or "[]")

def patch_req(table, idv, payload):
    h = dict(HDR); h["Prefer"] = "return=minimal"
    req = urllib.request.Request(f"{URL}/rest/v1/{table}?id=eq.{idv}", data=json.dumps(payload).encode(), headers=h, method="PATCH")
    urllib.request.urlopen(req, timeout=60).read()

def get_req(path):
    req = urllib.request.Request(f"{URL}/rest/v1/{path}", headers={"apikey":KEY,"Authorization":f"Bearer {KEY}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read() or "[]")

# ---------- buffer ----------
def buffer_count():
    if not BUFFER.exists(): return 0
    return sum(1 for _ in BUFFER.open())

def pop_post():
    lines = BUFFER.read_text().splitlines() if BUFFER.exists() else []
    if not lines: return None
    p = json.loads(lines[0])
    BUFFER.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""))
    return p

def refill_if_low():
    if buffer_count() < BUFFER_LOW:
        log(f"buffer low ({buffer_count()}) -> producing batch")
        subprocess.run([sys.executable, str(HERE/"produce.py"), "14", str(time.time_ns())],
                       timeout=900, capture_output=True)
        log(f"buffer now {buffer_count()}")

# ---------- comment queue ----------
def load_q():
    return json.loads(CQUEUE.read_text()) if CQUEUE.exists() else []
def save_q(q): CQUEUE.write_text(json.dumps(q, ensure_ascii=False))
def load_rel():
    return json.loads(RELEASED.read_text()) if RELEASED.exists() else {}
def save_rel(r): RELEASED.write_text(json.dumps(r, ensure_ascii=False))

# ---------- publish ----------
def publish_post(rng):
    p = pop_post()
    if not p: return None
    t = now()
    row = {"tenant_id": TENANT, "user_id": None, "category": p.get("category","FREE"),
           "title": p["title"], "body": p["body"], "is_anon": True, "source_author": None,
           "view_count": rng.randint(1, 22), "like_count": 0,
           "created_at": t.isoformat(), "updated_at": t.isoformat()}
    res = post_req("community_posts", [row])
    pid = res[0]["id"]
    # schedule planned comments with lognormal real-time delays, parents before children
    q = load_q()
    base = time.time()
    for idx, c in enumerate(p.get("comments") or []):
        delay_min = min(math.exp(rng.gauss(math.log(CMT_DELAY_MEDIAN_MIN), 0.9)), 60*40)
        q.append({"post_id": pid, "idx": idx, "parent_index": c.get("parent_index"),
                  "body": c["body"], "release": base + delay_min*60 + idx*30})
    save_q(q)
    log(f"POST  {pid[:8]} [{row['category']}] +{len(p.get('comments') or [])}cmt  «{p['title'][:30]}»")
    return pid

def release_due_comments():
    q = load_q()
    if not q: return 0
    rel = load_rel(); nowts = time.time(); keep = []; n = 0
    for item in sorted(q, key=lambda x: x["release"]):
        if item["release"] > nowts: keep.append(item); continue
        pid = item["post_id"]; pmap = rel.get(pid, {})
        parent_id = None
        if item.get("parent_index") is not None:
            parent_id = pmap.get(str(item["parent_index"]))
        t = now()
        row = {"tenant_id": TENANT, "post_id": pid, "user_id": None, "parent_id": parent_id,
               "body": item["body"], "is_anon": True, "source_author": None, "created_at": t.isoformat()}
        try:
            res = post_req("community_comments", [row])
            cid = res[0]["id"]
            pmap[str(item["idx"])] = cid; rel[pid] = pmap; n += 1
        except Exception as e:
            log(f"  cmt release fail {pid[:8]}: {str(e)[:40]}")
    save_q(keep); save_rel(rel)
    if n: log(f"  released {n} comment(s)")
    return n

# ---------- gap backfill (folded in: single process, no codex contention) ----------
def init_gap():
    """On first start, compute backdated hour-weighted timestamps for the quiet window
    [latest-pre-live-post .. now-2h] and persist. Ascending (fill chronologically)."""
    if GAP_STATE.exists():
        return
    from urllib.parse import quote
    cutoff = quote((now() - dt.timedelta(hours=2)).isoformat(), safe="")
    latest = get_req(f"community_posts?created_at=lt.{cutoff}&select=created_at&order=created_at.desc&limit=1")
    if not latest:
        GAP_STATE.write_text("[]"); return
    last_t = dt.datetime.fromisoformat(latest[0]["created_at"].replace("Z","+00:00"))
    end = now() - dt.timedelta(hours=2)
    gap_days = (end - last_t).total_seconds()/86400
    if gap_days < 0.3:
        GAP_STATE.write_text("[]"); log(f"gap {gap_days:.2f}d — nothing to backfill"); return
    n = int(gap_days * POSTS_PER_DAY)
    rng = random.Random(); span = (end - last_t).total_seconds(); ts = []
    while len(ts) < n:
        t = last_t + dt.timedelta(seconds=rng.random()*span)
        if rng.random() < KSTW[(t.hour+9)%24]/max(KSTW): ts.append(t)
    ts.sort()
    GAP_STATE.write_text(json.dumps([t.isoformat() for t in ts]))
    log(f"gap backfill queued: {len(ts)} posts across {gap_days:.2f}d [{last_t:%m-%d} .. {end:%m-%d}]")

def publish_gap(rng, k):
    if not GAP_STATE.exists(): return 0
    ts = json.loads(GAP_STATE.read_text())
    if not ts: return 0
    done = 0
    for _ in range(min(k, len(ts))):
        if not ts: break
        iso = ts.pop(0); t = dt.datetime.fromisoformat(iso)
        p = pop_post()
        if not p: break
        age_days = (now()-t).total_seconds()/86400
        views = max(1, int(rng.gauss(1822, 700) * min(1.0, age_days/5)))
        row = {"tenant_id": TENANT, "user_id": None, "category": p.get("category","FREE"),
               "title": p["title"], "body": p["body"], "is_anon": True, "source_author": None,
               "view_count": views, "like_count": rng.choice([0,0,0,1]),
               "created_at": t.isoformat(), "updated_at": t.isoformat()}
        try:
            pid = post_req("community_posts", [row])[0]["id"]
        except Exception as e:
            log(f"  gap post fail: {str(e)[:40]}"); continue
        rel = {}
        for idx, c in enumerate(p.get("comments") or []):
            dly = min(math.exp(rng.gauss(math.log(3), 1.0)), 24*8)
            ct = t + dt.timedelta(hours=dly)
            if ct >= now(): ct = now() - dt.timedelta(minutes=rng.randint(1,120))
            parent = rel.get(c.get("parent_index"))
            try:
                cid = post_req("community_comments", [{"tenant_id":TENANT,"post_id":pid,"user_id":None,
                    "parent_id":parent,"body":c["body"],"is_anon":True,"source_author":None,"created_at":ct.isoformat()}])[0]["id"]
                rel[idx] = cid
            except Exception: pass
        done += 1
    GAP_STATE.write_text(json.dumps(ts))
    if done: log(f"  GAP backfill +{done} ({len(ts)} remaining)")
    return done

# ---------- view aging ----------
def age_views(rng):
    # bump views on recent posts (age < 8d) so fresh posts climb toward the corpus median
    from urllib.parse import quote
    since = quote((now() - dt.timedelta(days=8)).isoformat(), safe="")
    rows = get_req(f"community_posts?created_at=gte.{since}&select=id,view_count,created_at,like_count&order=created_at.desc&limit=60")
    bumped = 0
    for r in rows:
        age_h = (now() - dt.datetime.fromisoformat(r["created_at"].replace("Z","+00:00"))).total_seconds()/3600
        rate = max(1, int(40 * math.exp(-age_h/120)))      # heavier when fresh, decays
        inc = rng.randint(0, rate)
        if inc:
            patch_req("community_posts", r["id"], {"view_count": int(r["view_count"]) + inc})
            bumped += 1
        if rng.random() < 0.03:  # occasional like
            patch_req("community_posts", r["id"], {"like_count": int(r["like_count"]) + 1})
    return bumped

# ---------- scheduling ----------
def next_interval(rng):
    h = (now().hour + 9) % 24                 # KST hour
    mean = (86400.0 / POSTS_PER_DAY) * (AVGW / KSTW[h])
    return rng.expovariate(1.0/mean)

def main():
    rng = random.Random()
    log(f"daemon start | tenant={TENANT} target={POSTS_PER_DAY}/day buffer={buffer_count()}")
    try: init_gap()
    except Exception as e: log(f"init_gap fail: {str(e)[:50]}")
    last_age = 0
    while True:
        refill_if_low()
        iv = next_interval(rng)
        # during the wait, release due comments every ~30s and age views periodically
        waited = 0
        while waited < iv:
            chunk = min(30, iv - waited)
            time.sleep(chunk); waited += chunk
            release_due_comments()
            if time.time() - last_age > 600:
                try: age_views(rng)
                except Exception as e: log(f"  age fail {str(e)[:40]}")
                last_age = time.time()
        try:
            publish_post(rng)                  # real-time live post
            publish_gap(rng, GAP_PER_TICK)     # backdated gap catch-up (until queue empty)
        except Exception as e:
            log(f"publish fail: {str(e)[:60]}")
            time.sleep(15)

if __name__ == "__main__":
    main()
