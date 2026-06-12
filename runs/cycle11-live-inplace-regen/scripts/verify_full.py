#!/usr/bin/env python3
"""Comprehensive final verification of the live community board against ALL original goals.

Covers: (A) content/privacy gates, (B) schema integrity, (C) metadata composition,
(D) temporal flow (24Q4-start exponential growth, hour weighting, comment-after-post,
no future ts), (E) domain-language register (반말율/은어 밀도/길이) — regen vs original.

Reads live DB via PostgREST (service-role) + batch_in originals. Script-side only.
"""
import json, re, sys, glob, statistics, urllib.request, datetime as dt
from collections import Counter
from pathlib import Path

ENV = "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"); KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Accept": "application/json"}
BASE = Path("/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen")

def get(path):
    return json.loads(urllib.request.urlopen(urllib.request.Request(f"{URL}/rest/v1/{path}", headers=HDR), timeout=120).read())

def page(table, select, extra=""):
    out, off = [], 0
    while True:
        rows = get(f"{table}?is_hidden=eq.false&select={select}{extra}&order=id.asc&limit=1000&offset={off}")
        out += rows; off += 1000
        if len(rows) < 1000: break
    return out

HARD = ["퀸알바","queenalba","[실명]","우지호","카톡","카카오톡","텔레그램","오픈채팅","면접비"]
PHONE = re.compile(r"01[016-9]\d{6,8}")
SLANG = ["떼초","초이스","케어","티씨","수위","1부","2부","개수","지명","실장","부장","마담",
         "하띠","정쩜","본가게","젊손","연장","초이","날개","상띠"]
HON = re.compile(r"(요|니다|세요|네요|아요|어요|죠|습니다)[.!?~\s]*$")

def has_oldjamo(t): return any(0x1100<=ord(c)<=0x11FF or 0xA960<=ord(c)<=0xA97F or 0xD7B0<=ord(c)<=0xD7FF for c in t)
def hon_ratio(texts):
    n=sum(1 for t in texts if HON.search((t or '').strip())); return round(n/max(1,len(texts)),3)
def slang_count(texts):
    blob="\n".join(texts); return {s:blob.count(s) for s in SLANG}

def main():
    print("fetching posts/comments metadata + text ...", file=sys.stderr)
    posts = page("community_posts", "id,title,body,view_count,like_count,comment_count,created_at")
    cmts  = page("community_comments", "id,post_id,body,created_at")
    P = {p["id"]: p for p in posts}
    now = dt.datetime.now(dt.timezone.utc)
    def parse(t): return dt.datetime.fromisoformat(t.replace("Z","+00:00"))

    R = {}
    # ---------- A. content / privacy gates ----------
    hard_hits=[]; jamo=0
    alltext=[(p["id"],"post",p["title"]+" "+p["body"]) for p in posts]+[(c["id"],"cmt",c["body"]) for c in cmts]
    for i,k,t in alltext:
        low=(t or "").lower()
        for w in HARD:
            if w.lower() in low: hard_hits.append((k,i,w)); break
        else:
            if PHONE.search(t or ""): hard_hits.append((k,i,"phone"))
        if has_oldjamo(t or ""): jamo+=1
    R["A_gates"]={"hard_banned_hits":len(hard_hits),"samples":hard_hits[:6],"oldhangul_jamo_rows":jamo}

    # ---------- B. schema integrity ----------
    cc=Counter(c["post_id"] for c in cmts)
    mismatch=sum(1 for p in posts if int(p["comment_count"])!=cc.get(p["id"],0))
    empty=sum(1 for p in posts if not (p["title"] or "").strip() or not (p["body"] or "").strip())
    R["B_schema"]={"posts":len(posts),"comments":len(cmts),
        "comment_count_vs_actual_mismatch":mismatch,"empty_title_or_body":empty}

    # ---------- C. metadata composition ----------
    vc=sorted(int(p["view_count"]) for p in posts); lc=[int(p["like_count"]) for p in posts]
    likes_gt_views=sum(1 for p in posts if int(p["like_count"])>int(p["view_count"]))
    def pct(a,q): return a[min(len(a)-1,int(len(a)*q))]
    R["C_metadata"]={"views_median":pct(vc,.5),"views_p90":pct(vc,.9),"views_max":vc[-1],
        "likes_median":statistics.median(lc),"likes_max":max(lc),"likes_gt_views":likes_gt_views,
        "comment_count_median":statistics.median([int(p["comment_count"]) for p in posts])}

    # ---------- D. temporal flow ----------
    months=Counter(parse(p["created_at"]).strftime("%Y-%m") for p in posts)
    hours=Counter(parse(p["created_at"]).hour for p in posts)  # UTC; KST=+9
    future=sum(1 for p in posts if parse(p["created_at"])>now)+sum(1 for c in cmts if parse(c["created_at"])>now)
    cbefore=sum(1 for c in cmts if c["post_id"] in P and parse(c["created_at"])<parse(P[c["post_id"]]["created_at"]))
    delays=[ (parse(c["created_at"])-parse(P[c["post_id"]]["created_at"])).total_seconds()/3600
             for c in cmts if c["post_id"] in P]
    delays=[x for x in delays if x>=0]
    kst_hours=Counter((h+9)%24 for h in hours.elements())
    even_night=sum(kst_hours[h] for h in list(range(19,24))+list(range(0,3)))
    R["D_temporal"]={"first":min(months),"last":max(months),"months_span":len(months),
        "monthly_growth_head":dict(sorted(months.items())[:4]),"monthly_growth_tail":dict(sorted(months.items())[-4:]),
        "future_timestamps":future,"comment_before_post":cbefore,
        "comment_delay_h_median":round(statistics.median(delays),1) if delays else None,
        "evening_night_post_share_KST(19-03h)":round(even_night/max(1,len(posts)),3)}

    # ---------- E. domain register (regen vs original) ----------
    src={}
    for f in glob.glob(str(BASE/"batch_in/batch_*.json")):
        for p in json.load(open(f)): src[p["id"]]=p
    o_titles=[s["title"] for s in src.values()]; o_bodies=[s["body"] for s in src.values()]
    o_cmts=[c["body"] for s in src.values() for c in s["comments"]]
    r_titles=[p["title"] for p in posts]; r_bodies=[p["body"] for p in posts]; r_cmts=[c["body"] for c in cmts]
    R["E_register"]={
        "honorific_ratio_cmt": {"original":hon_ratio(o_cmts),"regen":hon_ratio(r_cmts)},
        "mean_len": {"title":[round(statistics.mean(map(len,o_titles)),1),round(statistics.mean(map(len,r_titles)),1)],
                     "body":[round(statistics.mean(map(len,o_bodies)),1),round(statistics.mean(map(len,r_bodies)),1)],
                     "cmt":[round(statistics.mean(map(len,o_cmts)),1),round(statistics.mean(map(len,r_cmts)),1)]},
        "slang_total": {"original":sum(slang_count(o_titles+o_bodies+o_cmts).values()),
                        "regen":sum(slang_count(r_titles+r_bodies+r_cmts).values())},
        "slang_top_regen": dict(sorted(slang_count(r_titles+r_bodies+r_cmts).items(),key=lambda x:-x[1])[:10]),
    }
    print(json.dumps(R, ensure_ascii=False, indent=1))
    (BASE/"verify_full_report.json").write_text(json.dumps(R,ensure_ascii=False,indent=1))

if __name__=="__main__":
    main()
