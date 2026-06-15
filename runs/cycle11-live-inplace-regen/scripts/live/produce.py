#!/usr/bin/env python3
"""Producer: generate a batch of FRESH net-new posts (+planned comments) via Codex,
gate them (HARD/VENUE/jamo only — net-new so no verbatim/integrity), append to buffer.jsonl.

Run standalone: produce.py [N]   (default 12 posts). The daemon calls this when buffer is low.
"""
import json, re, sys, subprocess, random, zlib, time
from pathlib import Path

import os
HERE = Path(__file__).parent
BASE = HERE.parent.parent  # cycle11-live-inplace-regen
RECIPE = (HERE / "recipe_gen.md").read_text()
BUFFER = Path(os.environ.get("PRODUCE_BUFFER") or (HERE / "buffer.jsonl"))

# domain topic seeds — diverse angles; producer expands each into an original post
SEEDS = [
    "오늘 첫출근 가게 사이즈 고민", "담당이 케어 안해줘서 서운함", "진상 손님 방 후기",
    "수위 얘기 또 나옴", "티씨 페이 너무 짜다", "초이스 안돼서 멘탈 나감", "1부 2부 출근 타이밍 질문",
    "성형(코/눈/필러) 비용 후기", "본가게 옮길지 고민", "다른 지역 가게 정보 교환", "꽁 받는 기준",
    "젊손 취향 푸념", "실장 멘트 레퍼토리 짜증", "오늘 일 너무 없음", "텐카 할배 손님 힘듦",
    "다이어트/관리 정보", "쉬는날 뭐하지 일상", "마담 픽 vs 부장 픽", "초보인데 입문 가게 질문",
    "방에서 무한초 당한 썰", "연장 안되는 손님 스트레스", "동료 언니 뒷담/갈등", "월세 급해서 출근",
    "노도 vs 대형 비교", "담당 바꾸고 싶음", "의상/패션 뭐 입지", "손님 매너 좋았던 후기",
    "현금 수금 늦는 가게 불만", "멘탈 관리 어떻게들 하세요", "가게 분위기/조명 질문",
]

def jamo(t): return any(0x1100<=ord(c)<=0x11FF or 0xA960<=ord(c)<=0xA97F or 0xD7B0<=ord(c)<=0xD7FF for c in (t or ""))
HARD = ["퀸알바","queenalba","[실명]","우지호","카톡","카카오","텔레","오픈채팅","면접비","문의 주세요"]
VENUE = ["세이렌","유앤미","도파민","도파","보스턴","엘리트","퍼펙트","사라있네","달토","정점","썸데이","하퍼","루나","퀸"]
PHONE = re.compile(r"01[016-9]\d{6,8}")
def dirty(t):
    low=(t or "").lower()
    if any(w.lower() in low for w in HARD): return True
    if any(w in (t or "") for w in VENUE): return True
    if PHONE.search(t or "") or jamo(t or ""): return True
    return False

def gen_batch(n, salt):
    rng = random.Random(zlib.crc32(salt.encode()))
    # distinct seeds per batch (no back-to-back topic repeats); sample without replacement
    pool = SEEDS[:]; rng.shuffle(pool)
    seeds = (pool * ((n // len(pool)) + 1))[:n]
    cats = rng.choices(["FREE","QNA","TIP","NEWS"], weights=[88,9,2,1], k=n)
    specs = [{"topic_seed": s, "category": c, "planned_comments": rng.choices([0,1,2,3,4,5,7],weights=[8,14,20,20,16,12,10])[0]}
             for s, c in zip(seeds, cats)]
    prompt = RECIPE + f"""

작업: 아래 {n}개 사양으로 서로 다른 새 글을 창작하라. 각 사양: topic_seed(출발 소재), category, planned_comments(만들 댓글 수).
출력은 **유효한 JSON 배열 하나만**(코드펜스/설명 금지):
[{{"category":"FREE","title":"...","body":"...","comments":[{{"parent_index":null,"body":"..."}}]}}]
- comments 길이 = 해당 사양 planned_comments. parent_index 는 같은 글 안 다른 댓글의 0-based 인덱스(답글) 또는 null(평면).
- id 는 넣지 마라(서버가 부여). 식별정보 절대 금지 규칙 엄수.

사양:
{json.dumps(specs, ensure_ascii=False)}"""
    try:
        raw = subprocess.run(["codex","exec","--skip-git-repo-check","--dangerously-bypass-approvals-and-sandbox",prompt],
                             capture_output=True, text=True, timeout=600).stdout
    except Exception as e:
        sys.stderr.write(f"codex fail: {e}\n"); return []
    raw = raw.replace("```json","").replace("```","")
    i,j = raw.find("["), raw.rfind("]")
    if i<0 or j<=i: return []
    try: data = json.loads(raw[i:j+1])
    except Exception: return []
    out = []
    for p in data:
        if not isinstance(p, dict): continue
        title=(p.get("title") or "").strip(); body=(p.get("body") or "").strip()
        if not title or not body or dirty(title) or dirty(body): continue
        cmts=[]
        for c in p.get("comments") or []:
            b=(c.get("body") or "").strip()
            if not b or dirty(b): continue
            cmts.append({"parent_index": c.get("parent_index"), "body": b})
        out.append({"category": p.get("category","FREE") if p.get("category") in ("FREE","QNA","TIP","NEWS") else "FREE",
                    "title": title, "body": body, "comments": cmts})
    return out

def main():
    n = int(sys.argv[1]) if len(sys.argv)>1 else 12
    salt = sys.argv[2] if len(sys.argv)>2 else str(time.time_ns())
    batch = gen_batch(n, salt)
    with BUFFER.open("a") as f:
        for p in batch: f.write(json.dumps(p, ensure_ascii=False)+"\n")
    print(f"produced {len(batch)}/{n} -> {BUFFER}")

if __name__ == "__main__":
    main()
