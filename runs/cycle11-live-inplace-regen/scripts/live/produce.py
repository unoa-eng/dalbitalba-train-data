#!/usr/bin/env python3
"""Producer: generate a batch of FRESH net-new posts (+planned comments) via Codex,
gate them (HARD/VENUE/jamo only — net-new so no verbatim/integrity), append to buffer.jsonl.

Run standalone: produce.py [N]   (default 12 posts). The daemon calls this when buffer is low.
"""
import json, re, sys, subprocess, random, zlib, time, tempfile
from pathlib import Path

import os, shutil
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import verbatim_gate as VG  # 원천 verbatim 검수 게이트
BASE = HERE.parent.parent  # cycle11-live-inplace-regen
RECIPE = (HERE / "recipe_gen.md").read_text()
BUFFER = Path(os.environ.get("PRODUCE_BUFFER") or (HERE / "buffer.jsonl"))
# pin to a codex new enough for the current default model. /opt/homebrew/bin/codex is an
# OLD 0.118 that 400s on gpt-5.5; the nvm build (0.134+) works headless. Prefer explicit.
_CODEX_CANDS = [os.environ.get("CODEX_BIN"),
                "/Users/unoa/.nvm/versions/node/v24.14.1/bin/codex",
                shutil.which("codex")]
CODEX = next((c for c in _CODEX_CANDS if c and os.path.exists(c)), "codex")
# claude CLI — codex usage-limit/실패 시 자동 폴백. Anthropic 한도는 OpenAI(codex)와 별개 풀이라
# 한쪽이 막혀도 다른 쪽이 생성 지속. shell function 아닌 실제 바이너리 경로 필요.
_CLAUDE_CANDS = [os.environ.get("CLAUDE_BIN"),
                 "/Users/unoa/.local/bin/claude",
                 shutil.which("claude")]
CLAUDE = next((c for c in _CLAUDE_CANDS if c and os.path.exists(c)), "claude")

def _run_codex(prompt):
    # stdin=DEVNULL: codex 0.141은 stdin 안 닫히면 "Reading additional input from stdin..." 대기.
    # -o/--output-last-message: 0.141 exec 는 입력 프롬프트를 stdout에 에코한다. 프롬프트 안에
    # JSON 예시/specs 의 '['가 있어 stdout 첫'['~마지막']' 추출이 깨진다(파싱 0개). 최종 메시지를
    # 파일로 받아 클린 JSON만 파싱. 파일이 비면(usage-limit/오류) stdout+stderr 로 폴백(why 감지용).
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as _f:
        outpath = _f.name
    try:
        p = subprocess.run([CODEX,"exec","--skip-git-repo-check","--dangerously-bypass-approvals-and-sandbox",
                            "-o",outpath,prompt],
                           capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
        msg = ""
        try: msg = Path(outpath).read_text(encoding="utf-8")
        except Exception: msg = ""
        if msg.strip():
            return msg
        return (p.stdout or "") + "\n" + (p.stderr or "")
    finally:
        try: os.unlink(outpath)
        except Exception: pass

def _run_claude(prompt):
    # -p: 비대화형 print. </dev/null 로 stdin 닫아 블록 방지.
    p = subprocess.run([CLAUDE,"-p",prompt],
                       capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    return (p.stdout or "") + "\n" + (p.stderr or "")

# 폴백 순서(앞→뒤). PRODUCE_ENGINES=claude,codex 로 우선순위 뒤집기 가능.
_ENGINE_MAP = {"codex": _run_codex, "claude": _run_claude}
_ORDER = [e.strip() for e in (os.environ.get("PRODUCE_ENGINES") or "codex,claude").split(",") if e.strip() in _ENGINE_MAP]
ENGINES = [(name, _ENGINE_MAP[name]) for name in _ORDER] or [("codex",_run_codex),("claude",_run_claude)]

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
    # 자격지심·자기비하
    "남들 다 잘나가는데 나만 도태된 느낌", "초이스 안 돼서 내 탓 같고 위축됨",
    "동기보다 못 버는 것 같아 불안", "나이 많아서 안 뽑힐까 봐 자신감 바닥",
    "얼굴·몸매 비교돼서 출근이 무섭다", "나만 단골 없는 것 같아 자존감 떨어짐",
    # 은근한 자랑(험블브래그 — 푸념·고민으로 포장)
    "페이 너무 많이 들어와서 세금 걱정", "단골이 너무 붙어서 스케줄 빡세 힘들다",
    "초이스 계속 돼서 화장실 갈 틈도 없네 ㅠ", "지명 몰려서 몸이 못 버티겠다는 푸념",
    "성형 안 했는데 자연미인 소리 들어 부담", "어린 나이에 너무 잘 풀려서 고민",
    # 밸런스게임·실없는 잡담(가벼운 재미 — 댓글로 편 갈려 투표/티키타카)
    "차은우 사귀는대신 20억 갚아주기 vs 그냥 살기",
    "평생 진상만 받고 페이 3배 vs 매너손님만 받고 페이 반토막",
    "10억 받고 이 바닥 은퇴 vs 지금처럼 계속 벌기",
    "얼굴 그대로 30억 vs 연예인 얼굴에 무일푼",
    "단골 100명인데 다 텐카 vs 단골 3명인데 다 영앤리치",
    "평생 초이스 무조건 됨 근데 페이 최저 vs 초이스 복불복 근데 대박가게",
    "일주일 풀근무하고 한달 쉬기 vs 매일 조금씩 나가기",
    "가고싶은 데 다 갈 수 있는데 담당이 최악 vs 가게는 별론데 담당이 천사",
    "잠 안자도 안 피곤한 몸 vs 하루 12시간 자야되는데 개꿀피부",
    "전남친이 손님으로 옴 vs 아는 오빠가 손님으로 옴",
    "출근길 로또 1등인데 가게 사장이 앎 vs 아무도 모르는데 꽝",
    "MBTI별 진상 유형 얘기", "요즘 꽂힌 노래/드라마 잡담", "쉬는날 하고싶은 거 버킷리스트",
    "복권 되면 제일 먼저 할 일", "다시 태어나면 이 일 할까 말까",
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

# 자기감사 게이트: 익명 게시판에서 '평면' 댓글이 원글/팁/정보에 고마워하면 글쓴이 자화자찬으로 읽힌다.
# 답글(parent_index 지정)이면 다른 댓글에 대한 감사이므로 허용. 평면+감사+정보지시어 → 드롭.
_THANKS = re.compile(r"(고마|감사|ㄱㅅ|땡큐)")
_OPOBJ  = re.compile(r"(팁|꿀팁|정보|정리글|올려줘|올려주|적어줘|적어주|알려줘|알려주|(?<!댓)글 고마|(?<!댓)글 감사|공유)")
def _self_thanks_flat(parent_index, body):
    if parent_index is not None: return False
    b = body or ""
    return bool(_THANKS.search(b) and _OPOBJ.search(b))

def _norm_anon_seq(author, parent_index):
    """생성 모델의 author 값을 저장용 anon_seq 로 정규화.
    0=글쓴이(OP) — 답글일 때만 유효(평면이면 무효 None). 1..K=스레드 내 익명 페르소나.
    미지정/무효 → None(표시단 폴백; 평면은 절대 OP 아님)."""
    if isinstance(author, str):
        a = author.strip().lower()
        if a in ("op", "글쓴이"):
            return 0 if parent_index is not None else None
        try: author = int(a)
        except (TypeError, ValueError): return None
    if author == 0:
        return 0 if parent_index is not None else None
    try:
        v = int(author)
        return v if v >= 1 else None
    except (TypeError, ValueError):
        return None

def _parse_and_gate(raw):
    """엔진 출력(raw, stderr 섞여도 무방)에서 JSON 배열 추출 + HARD/VENUE/jamo/verbatim 게이트."""
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
        if VG.is_verbatim(title) or VG.is_verbatim(body): continue
        cmts=[]
        for c in p.get("comments") or []:
            b=(c.get("body") or "").strip()
            pi=c.get("parent_index")
            if not b or dirty(b) or VG.is_verbatim(b): continue
            if _self_thanks_flat(pi, b): continue
            cmts.append({"parent_index": pi, "body": b,
                         "anon_seq": _norm_anon_seq(c.get("author"), pi)})
        out.append({"category": p.get("category","FREE") if p.get("category") in ("FREE","QNA","TIP","NEWS") else "FREE",
                    "title": title, "body": body, "comments": cmts})
    return out

# ── 루틴화 방지: 글마다 형식/감정/길이/문체 축을 랜덤 부여 → 같은 topic도 매번 다른 글이 되게 ──
# form: 글의 뼈대(질문/하소연만 반복되는 걸 깨는 게 핵심)
FORMS = [
    ("hasoyeon", "하소연·푸념(답 안 구하고 그냥 털어놓기, 끝에 질문 없이 마무리해도 됨)"),
    ("question", "질문(구체적으로 뭐 하나 물어봄)"),
    ("review", "후기·경험담(있었던 일 서술 위주, 질문 없이)"),
    ("info", "정보·꿀팁 공유(내가 아는 거 알려주기)"),
    ("banzzak", "짧은 한 줄 잡담·드립(1~2문장 툭 던지기)"),
    ("balance", "밸런스게임·투표 유도(짧게 던지고 댓글로 편 갈리게)"),
    ("brag", "은근한 자랑/기쁜 일(자랑인데 티 안 나게 or 대놓고 신남)"),
    ("rant", "화남·빡침(진상·실장·담당한테 열받은 걸 쏟아냄)"),
    ("tmi", "TMI·일상(오늘 뭐 먹었네, 이런 시시콜콜)"),
    ("ask_opinion", "의견 물음(A vs B 뭐가 나아? 다들 어떻게 생각?)"),
]
TONES = ["담담함", "예민·짜증", "울적·자조", "신남·들뜸", "웃김·드립", "무덤덤 시크", "따뜻·다정", "냉소·시니컬"]
# length: 글자 수 대략 타깃(좁게 몰리지 않게 넓게)
LENGTHS = [("아주짧게","15~35자, 한 줄"), ("짧게","35~70자"), ("보통","70~130자"), ("길게","130~240자, 서너 문장")]
# 시간대 감성: 발행 시각(KST hour)에 맞춰 글의 결이 달라지게(새벽 퇴근감성 vs 낮 일상). 없으면 무관.
def _timeflavor(kst_hour):
    if kst_hour is None: return None
    if 0 <= kst_hour < 6:   return "새벽(퇴근 직후/취기/피곤·울적, 담배·집 가는 길·잠 안 옴 같은 결). 낮 얘기하듯 쓰지 말 것"
    if 6 <= kst_hour < 11:  return "아침·오전(자다 깸/전날 여파/해장, 늘어진 톤). 출근 각오는 아직 이름"
    if 11 <= kst_hour < 17: return "낮(오프·일상·준비, 병원·쇼핑·집안일 등 일 밖 얘기 어울림)"
    if 17 <= kst_hour < 20: return "초저녁(출근 준비/각오/긴장, 오늘 자리 어떨까)"
    return "밤(일하는 중 잠깐 짬/대기실에서/방금 있었던 일 실시간 느낌)"

def gen_batch(n, salt):
    rng = random.Random(zlib.crc32(salt.encode()))
    # distinct seeds per batch (no back-to-back topic repeats); sample without replacement
    pool = SEEDS[:]; rng.shuffle(pool)
    seeds = (pool * ((n // len(pool)) + 1))[:n]
    cats = rng.choices(["FREE","QNA","TIP","NEWS"], weights=[88,9,2,1], k=n)
    # form 은 편향 완화를 위해 셔플·순환(질문/하소연 쏠림 방지). tone/length 는 독립 랜덤.
    fpool = FORMS[:]; rng.shuffle(fpool)
    # 발행 시각대 가중치(데몬 KSTW와 정합) — 버퍼 글이 반나절 걸쳐 나가므로 시각 감성도 이 분포로.
    KSTW = [8,7,5,3,2,1,1,1,2,3,4,4,4,4,4,4,5,6,7,9,11,12,12,10]
    specs = []
    for k, (s, c) in enumerate(zip(seeds, cats)):
        form_key, form_desc = fpool[k % len(fpool)]
        tone = rng.choice(TONES)
        _, len_desc = rng.choice(LENGTHS)
        kst_h = rng.choices(range(24), weights=KSTW, k=1)[0]
        spec = {"topic_seed": s, "category": c,
                "planned_comments": rng.choices([0,1,2,3,4,5,7],weights=[8,14,20,20,16,12,10])[0],
                "형식": form_desc, "감정톤": tone, "길이": len_desc, "시간대": _timeflavor(kst_h)}
        specs.append(spec)
    prompt = RECIPE + f"""

작업: 아래 {n}개 사양으로 서로 다른 새 글을 창작하라. 각 사양: topic_seed(출발 소재), category, planned_comments(만들 댓글 수), 형식(글의 뼈대), 감정톤, 길이, 시간대(글이 올라온 시간 결).
**중요(루틴화 금지)**: 각 글은 사양의 **형식·감정톤·길이·시간대를 실제로 반영**해 서로 확연히 다른 글이 되게 하라. 특히:
- 모든 글이 "[상황 서술] + 나만그런가?/어떻게들함?" 질문으로 끝나는 판박이 금지. 형식이 하소연/후기/정보/잡담/드립이면 **질문 없이** 끝내도 된다.
- 시작을 매번 "요새/요즘/오늘/어제/다들"로 열지 마라. 본론부터, 감탄사, 한 단어, 대사 인용 등 도입을 다양하게.
- 길이 사양을 지켜 짧은 글은 진짜 짧게(한 줄), 긴 글은 여러 문장으로. 다 비슷한 길이로 수렴 금지.
- 이모티콘(ㅋㅋ/ㅠㅠ)도 감정톤 맞을 때만. 매 글 기계적으로 붙이지 마라.
- **시간대 감성 반영**: 사양의 시간대 결에 맞춰라(새벽 글을 낮 얘기처럼 쓰지 말 것).
- **구체 디테일로 리얼하게**(글의 절반 이상): 두루뭉술("좀 많이", "요즘")만 쓰지 말고 대충의 숫자·시간·금액·횟수·연차를 자연스럽게 박아라 — "3만원 깎임", "새벽 4시까지", "떼초 세 번 밀림", "5년째", "두 시간 대기", "10만원 꽁". 사람 말은 원래 구체적이다. 단 업소 특정될 정보(실명·위치·연락처)는 금지.
- **일 밖 세계도 가끔**: 24시간 일 얘기만 하는 사람은 없다. 드라마·유튜브·인스타·피부과·다이어트·연애·집·가족 같은 **커뮤 밖 일상**도 자연스럽게 섞어라(특히 낮/오프 글).
- **시기감 가끔**: 성수기/비수기, 명절, 연말, 날씨(더위·장마·추위) 같은 시점 얘기를 이따금.
출력은 **유효한 JSON 배열 하나만**(코드펜스/설명 금지):
[{{"category":"FREE","title":"...","body":"...","comments":[{{"parent_index":null,"author":1,"body":"..."}}]}}]
- comments 길이 = 해당 사양 planned_comments. parent_index 는 같은 글 안 **앞선(더 작은 0-based 인덱스)** 다른 댓글의 인덱스(답글) 또는 null(평면).
- **author (작성자 정체성, 필수)**: 정수 1,2,3…(서로 다른 익명. **같은 정수=같은 사람** 재등장) 또는 "op"(원글 작성자 자답).
  · 대부분의 댓글은 **서로 다른 사람** → 대체로 서로 다른 정수. 같은 사람이 다시 말할 때(자기 댓글 부연, OP와 주고받기)만 같은 정수 재사용.
  · "op"는 **원글 작성자가 자기 글의 댓글에 답하는 경우에만**, 반드시 **답글(parent_index 지정)** 로만. 평면 댓글(parent_index=null)엔 **절대 "op" 금지**.
  · 남이 원글에 다는 의견·위로·반박·질문은 전부 정수(다른 사람)다.
- id 는 넣지 마라(서버가 부여). 식별정보 절대 금지 규칙 엄수.

사양:
{json.dumps(specs, ensure_ascii=False)}"""
    # 엔진 폴백: codex → claude (또는 PRODUCE_ENGINES 순서). 한쪽 usage-limit/실패 시 자동 전환.
    fails = []  # (engine, why) — 모든 엔진 실패 시 텔레그램 경보용
    for name, fn in ENGINES:
        try:
            raw = fn(prompt)
        except Exception as e:
            fails.append((name, f"실행오류:{str(e)[:40]}"))
            sys.stderr.write(f"[{name}] 실행오류: {str(e)[:80]} -> 다음 엔진\n"); continue
        posts = _parse_and_gate(raw)
        if posts:
            sys.stderr.write(f"[{name}] produced {len(posts)}/{n}\n")
            # 어떤 엔진이 생성했는지 기록 (가시성/codex 복귀 감지용)
            try:
                (HERE / ".last_engine.json").write_text(
                    json.dumps({"engine": name, "ts": int(time.time()), "n": len(posts)}))
                # 직전 실패 마커 제거 (회복)
                (HERE / ".produce_fail.json").unlink(missing_ok=True)
            except Exception:
                pass
            return posts
        low = raw.lower()
        why = "usage limit" if "usage limit" in low else ("rate limit" if "rate limit" in low else "0개/파싱실패")
        fails.append((name, why))
        sys.stderr.write(f"[{name}] 실패({why}) -> 다음 엔진\n")
    # 모든 엔진 실패 — 실패 마커 기록(watchdog/buffer_alert 가 읽어 텔레그램 즉시 경보)
    sys.stderr.write("모든 엔진 실패 — buffer 리필 0\n")
    try:
        (HERE / ".produce_fail.json").write_text(json.dumps(
            {"ts": int(time.time()), "engines": [{"engine": e, "why": w} for e, w in fails]},
            ensure_ascii=False))
    except Exception:
        pass
    return []

def main():
    n = int(sys.argv[1]) if len(sys.argv)>1 else 12
    salt = sys.argv[2] if len(sys.argv)>2 else str(time.time_ns())
    batch = gen_batch(n, salt)
    with BUFFER.open("a") as f:
        for p in batch: f.write(json.dumps(p, ensure_ascii=False)+"\n")
    print(f"produced {len(batch)}/{n} -> {BUFFER}")

if __name__ == "__main__":
    main()
