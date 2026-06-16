#!/usr/bin/env python3
"""Daily publishing report for the dalbitalba live ops daemon.

Queries live DB (PostgREST) for the last 24h activity + checks daemon/launchd health,
writes reports/daily-YYYY-MM-DD.md, prints to stdout, and (if a notifier is configured)
delivers it. Designed to be run by a daily launchd job.
"""
import json, re, os, subprocess, urllib.request, datetime as dt
from pathlib import Path

HERE = Path(__file__).parent
ENV = str(HERE / ".env.prod") if (HERE / ".env.prod").exists() else "/tmp/dalbit.env.prod"
d = dict(re.findall(r'^(\w+)=\"?([^\"\n]*)', open(ENV).read(), re.M))
URL = d["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"); KEY = d["SUPABASE_SERVICE_ROLE_KEY"]
HDR = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
REPORTS = HERE / "reports"; REPORTS.mkdir(exist_ok=True)
NOTIFY = HERE / ".notify.json"   # optional {"telegram":{"token":..,"chat_id":..}}

def now(): return dt.datetime.now(dt.timezone.utc)
def kst(t): return t.astimezone(dt.timezone(dt.timedelta(hours=9)))
def get(path):
    from urllib.parse import quote
    req = urllib.request.Request(f"{URL}/rest/v1/{path}", headers={**HDR, "Prefer": "count=exact"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("content-range", "")
        total = cr.split("/")[-1] if "/" in cr else None
        return json.loads(r.read() or "[]"), total
def count(table, filt=""):
    from urllib.parse import quote
    req = urllib.request.Request(f"{URL}/rest/v1/{table}?{filt}&select=id&limit=1", headers={**HDR, "Prefer": "count=exact"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("content-range", "0-0/0"); return int(cr.split("/")[-1])

def q(ts):
    from urllib.parse import quote
    return quote(ts, safe="")

def daemon_health():
    try:
        pid = subprocess.run(["pgrep","-f","live/daemon.py"], capture_output=True, text=True).stdout.strip()
    except Exception: pid = ""
    la = ""
    try:
        uid = os.getuid()
        out = subprocess.run(["launchctl","print",f"gui/{uid}/com.dalbit.live-daemon"],
                             capture_output=True, text=True).stdout
        m = re.search(r"state = (\w+)", out); runs = re.search(r"runs = (\d+)", out)
        la = f"{m.group(1) if m else '?'} (runs={runs.group(1) if runs else '?'})"
    except Exception: la = "unknown"
    return pid, la

def build():
    n = now(); since24 = n - dt.timedelta(hours=24); today_kst = kst(n).date()
    s24 = q(since24.isoformat())
    posts_24 = count("community_posts", f"created_at=gte.{s24}")
    cmts_24  = count("community_comments", f"created_at=gte.{s24}")
    total_p  = count("community_posts", "is_hidden=eq.false")
    total_c  = count("community_comments", "is_hidden=eq.false")
    # hourly KST distribution of last-24h posts
    rows,_ = get(f"community_posts?created_at=gte.{s24}&select=created_at,view_count,title&order=created_at.desc&limit=400")
    from collections import Counter
    hrs = Counter(kst(dt.datetime.fromisoformat(r["created_at"].replace("Z","+00:00"))).hour for r in rows)
    spark = "".join("▁▂▃▄▅▆▇█"[min(7, hrs.get(h,0)*7//max(1,max(hrs.values()) if hrs else 1))] for h in range(24))
    # gap remaining
    gap = 0
    gs = HERE / "gap_state.json"
    if gs.exists():
        try: gap = len(json.loads(gs.read_text()))
        except Exception: gap = 0
    pid, la = daemon_health()
    # last few published
    last5 = [f"  · [{r['created_at'][11:16]}Z] {r.get('title','')[:32]}" for r in rows[:5]] if rows else []
    health = "🟢 정상" if (pid and posts_24 > 0) else ("🟡 글 0/24h — 점검필요" if pid else "🔴 데몬 미가동")
    lines = [
        f"📊 달빛알바 발행현황 — {today_kst} (KST)",
        f"",
        f"상태: {health}  | 데몬 pid {pid or '없음'} · launchd {la}",
        f"",
        f"최근 24h: 글 {posts_24} · 댓글 {cmts_24}  (목표 ~16글/일)",
        f"누적(공개): 글 {total_p:,} · 댓글 {total_c:,}",
        f"시간대(KST 0→23h): {spark}",
        f"갭 백필 잔여: {gap} 글",
    ]
    if last5:
        lines += ["", "최근 발행:"] + last5
    return "\n".join(lines), today_kst

def deliver(text):
    # optional telegram via .notify.json
    if NOTIFY.exists():
        try:
            cfg = json.loads(NOTIFY.read_text()).get("telegram", {})
            tok, chat = cfg.get("token"), cfg.get("chat_id")
            if tok and chat:
                data = json.dumps({"chat_id": chat, "text": text}).encode()
                req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage",
                                             data=data, headers={"Content-Type":"application/json"})
                urllib.request.urlopen(req, timeout=20).read()
                print("[delivered via telegram]")
        except Exception as e:
            print(f"[telegram deliver fail: {str(e)[:50]}]")
    # macOS notification (local)
    try:
        subprocess.run(["osascript","-e",
            'display notification "발행현황 리포트 생성됨" with title "달빛알바 데몬" sound name "Glass"'],
            capture_output=True)
    except Exception: pass

def main():
    text, day = build()
    (REPORTS / f"daily-{day}.md").write_text(text)
    print(text)
    deliver(text)

if __name__ == "__main__":
    main()
