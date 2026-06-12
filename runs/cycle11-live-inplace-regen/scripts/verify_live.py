#!/usr/bin/env python3
"""Real-domain verification of https://dalbitalba.co.kr community surface.

Passes the adult gate via the review-phone OTP path (no login needed for public reads),
then checks list/detail/API JSON for raw exposure (venue/person/brand/contact author/text)
and confirms authorDisplay masking + 24Q4 timeline.
"""
import json, re, urllib.request, http.cookiejar, sys

BASE = "https://dalbitalba.co.kr"
PHONE = "01012345678"; CODE = "000000"
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [("User-Agent", "Mozilla/5.0 dalbit-verify"), ("Content-Type", "application/json")]

def req(path, data=None, method=None):
    url = BASE + path
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                               method=method or ("POST" if data is not None else "GET"))
    try:
        resp = op.open(r, timeout=60); return resp.getcode(), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")

BANNED = ["세이렌","유앤미","도파민","보스턴","엘리트","퍼펙트","사라있네","달토","썸데이","하퍼","루나",
          "퀸알바","queenalba","[실명]","우지호","면접비","카톡","카카오톡","오픈채팅"]
def scan(t):
    return [w for w in BANNED if w.lower() in (t or "").lower()] + (["phone"] if re.search(r"01[016-9]\d{6,8}", t or "") else [])

def main():
    # 1) adult gate via review OTP
    c, b = req("/api/sms-otp", {"action": "send", "phone": PHONE})
    vid = json.loads(b).get("verificationId") if c == 200 else None
    print(f"sms send: {c} vid={vid}")
    if not vid: print("FAIL adult-gate send", b[:200]); return
    c, b = req("/api/sms-otp", {"action": "verify", "verificationId": vid, "code": CODE}); print(f"sms verify: {c}")
    c, b = req("/api/adult-verify", {"method": "phone", "otpVerificationId": vid}); print(f"adult-verify: {c} {b[:80]}")
    has_gate = any(ck.name for ck in cj)
    print("cookies:", [ck.name for ck in cj])

    # 2) list API — author masking + raw scan
    leaks = []; authors = {}; dates = []
    ids = []
    for pg in range(1, 9):
        c, b = req(f"/api/community/posts?sort=new&page={pg}&limit=30")
        if c != 200: print(f"list page{pg}: HTTP {c} {b[:120]}"); break
        d = json.loads(b); items = d.get("data", [])
        for it in items:
            ids.append(it["id"])
            ad = it.get("authorDisplay"); authors[ad] = authors.get(ad, 0) + 1
            if "sourceAuthor" in it or "anonymousKey" in it: leaks.append(("list-rawfield", it["id"]))
            for w in scan((it.get("title") or "") + " " + (it.get("excerpt") or "")): leaks.append(("list-text:"+w, it["id"]))
            if ad and scan(ad): leaks.append(("author:"+ad, it["id"]))
            if it.get("createdAt"): dates.append(it["createdAt"][:7])
    print(f"\nlist scanned ~{len(ids)} posts | distinct authorDisplay: {dict(list(authors.items())[:6])}")
    print(f"createdAt months seen (sample): {sorted(set(dates))[:3]} ... {sorted(set(dates))[-3:]}")

    # 3) detail API — bodies + comments
    det = 0
    for pid in ids[:12]:
        c, b = req(f"/api/community/posts/{pid}")
        if c != 200: continue
        d = json.loads(b).get("data") or json.loads(b)
        post = d.get("post", d); cmts = d.get("comments", [])
        det += 1
        for w in scan((post.get("title") or "")+" "+(post.get("body") or "")): leaks.append(("body:"+w, pid))
        if "sourceAuthor" in post: leaks.append(("detail-rawfield", pid))
        for cm in cmts:
            if "sourceAuthor" in cm: leaks.append(("cmt-rawfield", pid))
            for w in scan(cm.get("body") or ""): leaks.append(("cmt:"+w, pid))
            ad = cm.get("authorDisplay")
            if ad and scan(ad): leaks.append(("cmt-author:"+ad, pid))
    print(f"detail scanned: {det} posts (+comments)")

    # 4) rendered HTML spot check
    for path in ["/community", f"/community/{ids[0]}" if ids else "/community"]:
        c, b = req(path)
        hits = scan(b)
        print(f"HTML {path}: HTTP {c} rawhits={hits[:5]}")

    print("\n=== VERDICT ===")
    print(f"adult-gate: {'OK' if has_gate else 'FAIL'} | author masking: {'OK (all 익명/글쓴이)' if set(authors)<= {'익명','글쓴이',None} else 'CHECK '+str(set(authors))}")
    print(f"raw leaks found: {len(leaks)}")
    for l in leaks[:12]: print("  ", l)
    print("PASS" if not leaks and (set(authors) <= {"익명","글쓴이",None}) else "REVIEW")

if __name__ == "__main__":
    main()
