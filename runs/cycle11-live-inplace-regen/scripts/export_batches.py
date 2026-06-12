#!/usr/bin/env python3
"""Export visible community posts+comments from Supabase (PostgREST) into batch files.

Public surface = community_posts.is_hidden = false (matches queries.ts .eq('is_hidden',false)).
Each batch = 40 posts + their visible comments -> batch_in/batch_NNNN.json
Runs script-side with the service-role key (no Claude-token passthrough).
"""
import json, os, re, sys, urllib.request, urllib.parse
from pathlib import Path

ENV = "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Accept": "application/json"}

BASE = Path("/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen")
IN = BASE / "batch_in"; IN.mkdir(exist_ok=True)
BATCH = 40

def get(path):
    req = urllib.request.Request(f"{URL}/rest/v1/{path}", headers=HDR)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

def fetch_posts():
    out = []; off = 0
    while True:
        rows = get(f"community_posts?is_hidden=eq.false&select=id,category,title,body,comment_count,view_count,like_count,created_at&order=id.asc&limit=1000&offset={off}")
        if not rows: break
        out += rows; off += 1000
        print(f"  posts fetched {len(out)}", file=sys.stderr)
        if len(rows) < 1000: break
    return out

def fetch_comments(post_ids):
    ids = ",".join(post_ids)
    rows = []
    # chunk the in() list to keep URL length sane
    CH = 50
    for i in range(0, len(post_ids), CH):
        sub = ",".join(post_ids[i:i+CH])
        rows += get(f"community_comments?is_hidden=eq.false&post_id=in.({sub})&select=id,post_id,parent_id,body,created_at&order=created_at.asc")
    return rows

def main():
    posts = fetch_posts()
    print(f"visible posts: {len(posts)}")
    nb = 0; ncmt = 0
    for i in range(0, len(posts), BATCH):
        chunk = posts[i:i+BATCH]
        pids = [p["id"] for p in chunk]
        cmts = fetch_comments(pids)
        by = {}
        for c in cmts:
            by.setdefault(c["post_id"], []).append(c)
            ncmt += 1
        batch = []
        for p in chunk:
            batch.append({
                "id": p["id"], "category": p["category"], "title": p["title"], "body": p["body"],
                "comment_count": p["comment_count"], "created_at": p["created_at"],
                "comments": [{"id": c["id"], "parent_id": c["parent_id"], "body": c["body"]}
                             for c in by.get(p["id"], [])],
            })
        (IN / f"batch_{nb:04d}.json").write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
        nb += 1
    print(f"wrote {nb} batches, {len(posts)} posts, {ncmt} comments -> {IN}")

if __name__ == "__main__":
    main()
