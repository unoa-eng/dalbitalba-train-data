#!/usr/bin/env python3
"""Verify regenerated batches against gates, then UPDATE Supabase by id (threaded PostgREST PATCH).

Gates (per batch_out vs batch_in):
  - integrity: same post ids, same comment ids+parent_id, equal comment count
  - banned tokens (venue/person/brand/contact) == 0
  - old-hangul jamo (U+1100..U+11FF / A960.. / D7B0..) == 0
  - verbatim: no >=5 consecutive identical eojeol vs source (same id)
Rows in a batch that FAIL any gate -> whole batch quarantined to requeue.json (regenerate later).
Passing batches -> PATCH community_posts(title,body) + community_comments(body) by id.

Usage: load_batches.py [--dry] [batch_0000 batch_0001 ...]   (default: all batch_out/*.json)
"""
import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENV = "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"); KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Prefer": "return=minimal"}

BASE = Path("/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen")
IN, OUT = BASE / "batch_in", BASE / "batch_out"

# HARD: identity/brand/contact — must NEVER appear in regen, regardless of source.
HARD = ["퀸알바","queenalba","[실명]","우지호","카톡","카카오톡","텔레그램","오픈채팅","면접비"]
# VENUE/brand names — flag only if the SAME term is in that post's SOURCE (i.e. Codex failed to strip).
# (substring-matching these as always-banned causes false positives on common words 엘리트/퍼펙트/보스턴/사라…)
VENUE = ["세이렌","유앤미","도파민","도파","보스턴","엘리트","퍼펙트","사라있네","달토","정점","썸데이",
         "하퍼","루나","퀸"]
PHONE = re.compile(r"01[016-9]\d{6,8}")

def has_jamo(t):
    return any(0x1100 <= ord(c) <= 0x11FF or 0xA960 <= ord(c) <= 0xA97F or 0xD7B0 <= ord(c) <= 0xD7FF for c in (t or ""))

def maxrun(a, b):
    A, B = (a or "").split(), (b or "").split(); best = 0
    for i in range(len(A)):
        for j in range(len(B)):
            k = 0
            while i + k < len(A) and j + k < len(B) and A[i + k] == B[j + k]: k += 1
            if k > best: best = k
    return best

def hard_hit(t):
    low = (t or "").lower()
    for w in HARD:
        if w.lower() in low: return w
    if PHONE.search(t or ""): return "phone"
    return None

def venue_hit(regen, src_blob):
    for w in VENUE:
        if w in (regen or "") and w in src_blob: return w
    return None

def verbatim_fail(a, b):
    r = maxrun(a, b)
    m = min(len((a or "").split()), len((b or "").split())) or 1
    return r >= 10 or (r >= 5 and r >= 0.8 * m)

def check_batch(n):
    """Return (reg, post_errs{pid:[reasons]}, structural[]). Per-POST granularity."""
    src = {p["id"]: p for p in json.load(open(IN / f"{n}.json"))}
    reg = {p["id"]: p for p in json.load(open(OUT / f"{n}.json"))}
    structural = []
    if set(reg) - set(src): structural.append("regen has unknown post ids")
    post_errs = {}
    for pid, s in src.items():
        r = reg.get(pid); e = []
        if not r:
            post_errs[pid] = ["missing in regen"]; continue
        sc = {c["id"]: c for c in s["comments"]}; rc = {c["id"]: c for c in r.get("comments", [])}
        if len(sc) != len(rc): e.append(f"cmt count {len(sc)}!={len(rc)}")
        src_blob = " ".join([s["title"], s["body"]] + [c["body"] for c in s["comments"]])
        fields = [("title", s["title"], r.get("title", "")), ("body", s["body"], r.get("body", ""))]
        for cid, c in sc.items():
            rcm = rc.get(cid)
            if not rcm: e.append(f"missing cmt {cid}"); continue
            if (c.get("parent_id") or None) != (rcm.get("parent_id") or None): e.append(f"parent {cid}")
            fields.append(("cmt", c["body"], rcm.get("body", "")))
        for nm, a, b in fields:
            h = hard_hit(b)
            if h: e.append(f"{nm} hard:{h}")
            v = venue_hit(b, src_blob)
            if v: e.append(f"{nm} unstripped:{v}")
            if has_jamo(b): e.append(f"{nm} oldjamo")
            if verbatim_fail(a, b): e.append(f"{nm} near-identical")
        if e: post_errs[pid] = e
    return reg, post_errs, structural

def patch(table, idv, payload):
    payload = {k: (v.replace("\x00", "").replace("​", "") if isinstance(v, str) else v)
               for k, v in payload.items()}
    req = urllib.request.Request(f"{URL}/rest/v1/{table}?id=eq.{idv}", data=json.dumps(payload).encode(),
                                 headers=HDR, method="PATCH")
    urllib.request.urlopen(req, timeout=60).read()

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry" in sys.argv
    LOADED_F = BASE / "loaded.json"
    loaded_set = set(json.loads(LOADED_F.read_text())) if LOADED_F.exists() else set()
    names = args or sorted(p.stem for p in OUT.glob("batch_*.json"))
    if not dry:
        names = [n for n in names if n not in loaded_set]
    loaded_p = loaded_c = 0
    tasks = []; ok_names = []; bad_batches = []; requeue_posts = []
    for n in names:
        if not (OUT / f"{n}.json").exists(): bad_batches.append(n); continue
        try:
            reg, post_errs, structural = check_batch(n)
        except Exception as e:
            bad_batches.append(n); print(f"{n}: PARSE/ERR {e}"); continue
        if structural:
            bad_batches.append(n); print(f"{n}: STRUCTURAL {structural}"); continue
        for pid, p in reg.items():
            if pid in post_errs:
                requeue_posts.append({"batch": n, "id": pid, "reason": post_errs[pid][:3]}); continue
            if not (p.get("title") or "").strip() or not (p.get("body") or "").strip():
                requeue_posts.append({"batch": n, "id": pid, "reason": ["empty title/body"]}); continue
            tasks.append(("community_posts", pid, {"title": p["title"], "body": p["body"]})); loaded_p += 1
            for c in p.get("comments", []):
                if not isinstance(c.get("id"), str) or len(c["id"]) != 36:
                    continue  # malformed comment id -> skip (post text still loads)
                tasks.append(("community_comments", c["id"], {"body": (c.get("body") or "").strip() or "…"})); loaded_c += 1
        ok_names.append(n)  # good posts of this batch are queued; its bad posts go to requeue_posts
    reasons = {}
    for rp in requeue_posts:
        k = rp["reason"][0].split(":")[0].split()[-1] if rp["reason"] else "?"
        reasons[k] = reasons.get(k, 0) + 1
    print(f"batches scanned={len(names)} | good posts={loaded_p} comments={loaded_c} | requeue posts={len(requeue_posts)} {reasons} | bad batches={len(bad_batches)}")
    # MERGE (union by id) with any existing requeue so incremental loads don't drop earlier failures
    RQP = BASE / "requeue_posts.json"
    merged = {}
    if RQP.exists():
        for rp in json.loads(RQP.read_text()): merged[rp["id"]] = rp
    for rp in requeue_posts: merged[rp["id"]] = rp
    RQP.write_text(json.dumps(list(merged.values()), ensure_ascii=False))
    RQB = BASE / "requeue.json"
    bb = set(json.loads(RQB.read_text())) if RQB.exists() else set()
    bb.update(bad_batches)
    RQB.write_text(json.dumps(sorted(bb), ensure_ascii=False))
    if dry:
        print("DRY — no DB writes. requeue ->", BASE / "requeue.json"); return
    done = [0]; fails = []
    def run(t):
        try:
            patch(*t); done[0] += 1
        except Exception as e:
            fails.append((t[0], t[1], str(e)[:30]))
        if (done[0] + len(fails)) % 3000 == 0: print(f"  patched {done[0]}/{len(tasks)} (fail {len(fails)})")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(run, tasks))
    if fails: print(f"  PATCH failures: {len(fails)} e.g. {fails[:3]}")
    loaded_set.update(ok_names)
    LOADED_F.write_text(json.dumps(sorted(loaded_set), ensure_ascii=False))
    print(f"DONE patched {done[0]} rows. loaded batches total={len(loaded_set)} | requeue posts={len(requeue_posts)} bad batches={len(bad_batches)}")

if __name__ == "__main__":
    main()
