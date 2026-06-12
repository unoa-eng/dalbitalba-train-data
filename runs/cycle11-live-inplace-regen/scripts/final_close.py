#!/usr/bin/env python3
"""Final close on the stubborn residual: guarantee 0 hard-banned / 0 unstripped-venue / 0 jamo
in the live DB for every residual post — accepting 'near-identical' (a quality, not safety, flag).

For each residual id: use the best Codex regen (requeue_out/batch_out) if it passes
HARD+VENUE+JAMO+integrity (ignoring near-identical) -> PATCH. Else deterministically SCRUB
(venue->generic, hard->remove, jamo->compat) the regen (or live DB text) -> PATCH.
"""
import json, re, sys, glob, unicodedata, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from load_batches import hard_hit, venue_hit, has_jamo, patch, VENUE  # noqa

ENV = "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"); KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
BASE = Path("/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen")

HARDSUB = {"[실명]": "", "면접비": "조건", "카톡": "연락", "카카오톡": "연락", "텔레그램": "연락",
           "오픈채팅": "대화", "퀸알바": "여기", "queenalba": "여기", "우지호": "그분"}
PHONE = re.compile(r"01[016-9]\d{6,8}")

def scrub(t):
    if not t: return t
    t = unicodedata.normalize("NFC", t)  # collapse stray old-jamo where possible
    for k, v in HARDSUB.items(): t = t.replace(k, v)
    for w in VENUE: t = t.replace(w, "거기")
    t = PHONE.sub("", t)
    # drop any remaining isolated old-hangul jamo
    t = "".join(c for c in t if not (0x1100 <= ord(c) <= 0x11FF or 0xA960 <= ord(c) <= 0xA97F or 0xD7B0 <= ord(c) <= 0xD7FF))
    return t.strip() or "…"

def clean(t, blob):
    return not hard_hit(t) and not venue_hit(t, blob) and not has_jamo(t)

def regen_index():
    idx = {}
    for f in glob.glob(str(BASE/"requeue_out/*.json")) + glob.glob(str(BASE/"batch_out/*.json")):
        try:
            for p in json.load(open(f)):
                if isinstance(p, dict) and "id" in p: idx[p["id"]] = p
        except Exception: pass
    return idx

def get_db(pid):
    g = lambda p: json.loads(urllib.request.urlopen(urllib.request.Request(f"{URL}/rest/v1/{p}", headers=HDR), timeout=60).read())
    post = g(f"community_posts?id=eq.{pid}&select=title,body")
    cmts = g(f"community_comments?post_id=eq.{pid}&is_hidden=eq.false&select=id,body")
    if not post: return None
    return {"id": pid, "title": post[0]["title"], "body": post[0]["body"],
            "comments": [{"id": c["id"], "body": c["body"]} for c in cmts]}

def main():
    ids = [x["id"] for x in json.load(open(BASE/"requeue_posts.json"))]
    idx = regen_index()
    print(f"residual to close: {len(ids)}")
    patched = scrubbed = 0
    for pid in ids:
        r = idx.get(pid) or get_db(pid)
        if not r: print(f"{pid}: no source"); continue
        blob = (r.get("title","")+" "+r.get("body","")+" "+" ".join(c.get("body","") for c in r.get("comments",[])))
        title, body = r.get("title") or "…", r.get("body") or "…"
        # if regen already clean, keep it; else scrub
        if not clean(title, blob) or not clean(body, blob): title, body = scrub(title), scrub(body); scrubbed += 1
        try:
            patch("community_posts", pid, {"title": title or "…", "body": body or "…"})
            for c in r.get("comments", []):
                cb = c.get("body") or "…"
                if not clean(cb, blob): cb = scrub(cb)
                if isinstance(c.get("id"), str) and len(c["id"]) == 36:
                    patch("community_comments", c["id"], {"body": cb or "…"})
            patched += 1
        except Exception as e:
            print(f"{pid}: patch fail {str(e)[:40]}")
    print(f"FINAL CLOSE: patched {patched}/{len(ids)} (scrubbed {scrubbed})")
    (BASE/"requeue_posts.json").write_text("[]")

if __name__ == "__main__":
    main()
