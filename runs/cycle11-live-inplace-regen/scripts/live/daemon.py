#!/usr/bin/env python3
"""Live operations daemon: continuously publish fresh posts + comments in REAL TIME
with randomized, KST-hour-weighted spacing (Poisson), matching the meta-graph density.

- posts: ~POSTS_PER_DAY/day, exponential inter-arrival modulated by hour weight (random, not fixed).
- comments: each post's planned comments released later via lognormal delay (real-time trickle).
- created_at = now() at insertion (genuine real-time). views grow via aging bumps.
- buffer auto-refilled by produce.py (Codex). State persisted -> restart-safe.

Run: daemon.py   (loops forever; launch under nohup). Ctrl-C / SIGTERM safe.
"""
import json, re, os, sys, time, math, random, zlib, subprocess, urllib.request, datetime as dt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verbatim_gate as VG  # 발행 직전 원천 verbatim 검수 게이트
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
    # 발행 직전 원천 verbatim 최종 검수 — 통과 못하면 폐기하고 다음 후보로
    ok, why = VG.review_post(p)
    if not ok:
        log(f"  PREPUB REJECT [{why}] «{(p.get('title') or '')[:24]}» -> 폐기")
        return publish_post(rng)
    t = now()
    row = {"tenant_id": TENANT, "user_id": None, "category": p.get("category","FREE"),
           "title": p["title"], "body": p["body"], "is_anon": True, "source_author": None,
           "origin": "synthetic",
           "view_count": rng.randint(1, 22), "like_count": 0, "synthetic_like_count": 0,
           "created_at": t.isoformat(), "updated_at": t.isoformat()}
    res = post_req("community_posts", [row])
    pid = res[0]["id"]
    # schedule planned comments with lognormal real-time delays, parents before children
    q = load_q()
    base = time.time()
    rel_times = []
    for idx, c in enumerate(p.get("comments") or []):
        delay_min = min(math.exp(rng.gauss(math.log(CMT_DELAY_MEDIAN_MIN), 0.9)), 60*40)
        own = base + delay_min*60 + idx*30
        pi = c.get("parent_index")
        if pi is not None and isinstance(pi, int) and 0 <= pi < idx:
            release = max(own, rel_times[pi] + rng.uniform(60, 600))
        else:
            release = own
        rel_times.append(release)
        q.append({"post_id": pid, "idx": idx, "parent_index": c.get("parent_index"),
                  "anon_seq": c.get("anon_seq"), "body": c["body"], "release": release})
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
               "body": item["body"], "is_anon": True, "source_author": None, "origin": "synthetic", "anon_seq": item.get("anon_seq"), "created_at": t.isoformat()}
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
        ok, why = VG.review_post(p)  # 백필 글도 원천 verbatim 검수
        if not ok:
            log(f"  GAP PREPUB REJECT [{why}] «{(p.get('title') or '')[:24]}» -> 폐기"); continue
        # 현실적 조회수: 글별 상한의 일부(가시 직후~성숙). 천단위 기본 금지.
        cap = view_ceiling("gap:"+iso)
        views = max(3, int(cap * (0.5 + rng.random()*0.5)))
        likes = min(like_ceiling(views), int(rng.expovariate(1/1.5)))  # 대부분 0~2, 난수
        row = {"tenant_id": TENANT, "user_id": None, "category": p.get("category","FREE"),
               "title": p["title"], "body": p["body"], "is_anon": True, "source_author": None,
               "origin": "synthetic",
               "view_count": views, "like_count": likes, "synthetic_like_count": likes,
               "created_at": t.isoformat(), "updated_at": t.isoformat()}
        try:
            pid = post_req("community_posts", [row])[0]["id"]
        except Exception as e:
            log(f"  gap post fail: {str(e)[:40]}"); continue
        rel = {}
        ct_times = []
        for idx, c in enumerate(p.get("comments") or []):
            dly = min(math.exp(rng.gauss(math.log(3), 1.0)), 24*8)
            ct = t + dt.timedelta(hours=dly)
            nowt = now()
            pi = c.get("parent_index")
            if pi is not None and isinstance(pi, int) and 0 <= pi < idx:
                # 답글은 반드시 (부모 ct, 현재) 사이에 위치 — 경계 클램프로도 역전 불가.
                # 부모는 귀납적으로 항상 현재 이전이므로 span>0 보장.
                lo = ct_times[pi]
                if ct <= lo or ct >= nowt:
                    span = (nowt - lo).total_seconds()
                    ct = lo + dt.timedelta(seconds=span * rng.uniform(0.1, 0.9))
            elif ct >= nowt:
                ct = nowt - dt.timedelta(seconds=rng.randint(30, 600))
            ct_times.append(ct)
            parent = rel.get(c.get("parent_index"))
            try:
                cid = post_req("community_comments", [{"tenant_id":TENANT,"post_id":pid,"user_id":None,
                    "parent_id":parent,"body":c["body"],"is_anon":True,"source_author":None,"origin":"synthetic","anon_seq":c.get("anon_seq"),"created_at":ct.isoformat()}])[0]["id"]
                rel[idx] = cid
            except Exception: pass
        done += 1
    GAP_STATE.write_text(json.dumps(ts))
    if done: log(f"  GAP backfill +{done} ({len(ts)} remaining)")
    return done

# ---------- realistic meta model (현실적 메타: 조회수 대부분 100언더, 인기글만 높게) ----------
def view_ceiling(post_id):
    """글별 결정론적 조회수 상한. 대부분 40~150(100 전후), 일부 수백, 드물게 천단위.
    aging은 이 상한까지만 천천히 증가 → median ~60, 천단위는 극소수."""
    r = (zlib.crc32(("vc:"+str(post_id)).encode()) % 100000) / 100000.0
    if   r < 0.78:  return 20 + int(r/0.78 * 100)          # 20~120 (다수, 100 전후)
    elif r < 0.95:  return 120 + int((r-0.78)/0.17 * 330)  # 120~450
    elif r < 0.995: return 450 + int((r-0.95)/0.045 * 650) # 450~1100 (희소)
    else:           return 1100 + int((r-0.995)/0.005 * 1300) # 1100~2400 (극소수)

def like_ceiling(view_count):
    """조회수 비례 좋아요 소프트상한 — 대부분 한 자리수, 인기글만 두 자리(난수). 하드캡 아님."""
    return max(1, view_count // 60)   # 60뷰당 ~1 좋아요 상한 (300뷰=5, 600=10, 1500=25)

# ---------- view aging ----------
def age_views(rng):
    # 최근글(8d 이내) 조회수를 글별 상한까지 천천히 증가. 상한 도달 시 멈춤(천단위 인플레 방지).
    from urllib.parse import quote
    since = quote((now() - dt.timedelta(days=8)).isoformat(), safe="")
    rows = get_req(f"community_posts?created_at=gte.{since}&select=id,view_count,created_at,like_count,synthetic_like_count&order=created_at.desc&limit=60")
    bumped = 0
    for r in rows:
        vc = int(r["view_count"]); cap = view_ceiling(r["id"])
        if vc < cap:
            inc = min(rng.randint(0, 4), cap - vc)   # 작은 증분, 상한 초과 금지
            if inc:
                patch_req("community_posts", r["id"], {"view_count": vc + inc}); vc += inc; bumped += 1
        # 좋아요: 조회수 비례 소프트상한 미만일 때만 가끔 +1 (난수, 하드캡 아님)
        if rng.random() < 0.04 and int(r["like_count"]) < like_ceiling(vc):
            patch_req("community_posts", r["id"], {"like_count": int(r["like_count"]) + 1, "synthetic_like_count": int(r.get("synthetic_like_count", 0)) + 1})
    return bumped

# ---------- scheduling ----------
def day_factor():
    """Per-KST-day random multiplier on the daily rate — gives organic 'busy/quiet day'
    overdispersion on top of the Poisson noise. Stable within a day (seeded by the date),
    varies day to day. ~N(1, 0.16) clamped to [0.65, 1.45]."""
    kday = (now() + dt.timedelta(hours=9)).strftime("%Y-%m-%d")
    r = random.Random(zlib.crc32(("dayfac:" + kday).encode()))
    return max(0.65, min(1.45, r.gauss(1.0, 0.16)))

def next_interval(rng):
    h = (now().hour + 9) % 24                       # KST hour
    eff_per_day = POSTS_PER_DAY * day_factor()      # today's randomized target (mean = POSTS_PER_DAY)
    mean = (86400.0 / eff_per_day) * (AVGW / KSTW[h])
    return rng.expovariate(1.0/mean)                # + Poisson noise on arrivals

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
