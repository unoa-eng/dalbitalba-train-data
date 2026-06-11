#!/usr/bin/env python3
"""Finish Phase 2 Claude-lane artifacts deterministically.

This script completes the handoff queue:
- post_out/out04..11 from post_parts/part04..11
- bulk_out/out24..47 from bulk_parts/part24..47 with the v2 register
- bulk_out/out00..23 honorific register reduction
- garbled-thread comment regeneration after post rework
- comment normalization in all early/bulk outputs
"""

from __future__ import annotations

import json
import random
import re
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from dalbit_gen_normalize import COMPAT_TO_HANGUL, normalize

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
SEED = REPO / "runs/cycle10-phase1-claude-direct/threads_v3.jsonl"

POST_PARTS = BASE / "post_parts"
POST_OUT = BASE / "post_out"
BULK_PARTS = BASE / "bulk_parts"
BULK_OUT = BASE / "bulk_out"
EARLY = BASE / "dalbit_phase2_early_merged.jsonl"
GARBLED_IDS = BASE / "scripts/dalbit_garbled_ids.json"
REPORT = BASE / "phase2_completion_metrics.json"
SOURCE_FLAT = REPO / "runs/cycle10/data-representative-v1-clean/threads_flat/train.jsonl"

PROMPT_V2 = (
    "반말/단답 70%+, 존댓말 <=30%, ㅋ/ㅠ run 2~8 다양화, "
    "구두점 ?!..~ 자유, 걍 사용, 중앙값 약 30자, root_id+comment_index 보존"
)

HONORIFIC_RE = re.compile(
    r"(습니다|습니까|세요|셔요|어요|아요|해요|네요|겠네요|군요|죠|나요|"
    r"드릴게요|하시|해주세요|해보세요|같아요|좋아요|됩니다|입니다|입니까|"
    r"인데요|거예요|게요|되세요|보세요)"
)
PREFIX_RE = re.compile(r"^\[\d+(?:-\d+)?\]\s*")
SOURCE_PREFIX_RE = re.compile(r"^\s*(?:작성자|비회원)?\s*(?:\[\d+(?:-\d+)?\]\s*)+")
SOURCE_BAD_RE = re.compile(
    r"신고에\s*의해\s*블라인드|블라인드\s*(?:처리|되었습니다|된\s*게시)|"
    r"관리자에 의해|신고가 접수|이용이 제한|운영원칙|"
    r"\[전화번호\]|\[URL\]|https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+|"
    r"\b010[.\-\s]?\d|카톡\s*(?:한번|아이디|주세요|주세|문의)|"
    r"카카오\s*[A-Za-z0-9]|카카오톡|오픈\s*채팅|오픈채팅|오픈톡|텔레그램|"
    r"문의\s*(?:주세요|주세|환영|가능)|연락\s*(?:주세요|주세|바랍니다|가능|부탁)|"
    r"상담\s*(?:문의|환영|가능)|24\s*시\s*영업|개인\s*이벤트|담당\s*고르|"
    r"돈\s*벌러\s*오|오세요~|출근시\s*첫|첫\s*티씨|첫티씨|"
    r"풀티\s*지급|TC\s*\+|풀상주|칼수금|스타트톡|개수톡|출퇴근\s*차량|"
    r"여성용품|매점|비품|밀빵|잘챙겨드릴게요|모십니다|지원해드립니다|"
    r"도와드리겠습니다|와주시면\s*감사",
    re.I,
)
SOURCE_STYLE_RATE = 1.0

CAT_KEYWORDS = {
    "health": [
        "몸살", "감기", "한약", "영양제", "피부", "병원", "운동", "컨디션",
        "생리", "수면", "잠", "아파", "다이어트", "헤메", "붓기",
    ],
    "customer": [
        "손님", "진상", "단골", "예약", "테이블", "룸", "터치", "수위",
        "스토킹", "차단", "연락", "카톡", "신고", "경찰", "내용증명",
    ],
    "shop": [
        "가게", "마담", "매니저", "면접", "옮기", "강남", "쩜오", "일프로",
        "퍼블릭", "출근", "퇴근", "첫차", "택시", "대기", "콜", "초이스",
    ],
    "money": [
        "매출", "팁", "티씨", "월세", "돈", "카드", "대출", "생활비",
        "정산", "벌이", "입금", "청산",
    ],
    "rest": [
        "비번", "혼자", "산책", "한강", "카페", "영화", "낮잠", "집",
        "본가", "라면", "휴가", "쉬는날",
    ],
    "season": [
        "여름", "겨울", "봄", "가을", "월말", "중순", "연말", "송년",
        "신년", "크리스마스", "장마", "휴가철", "추석", "설",
    ],
}

CATEGORY_WORDS = {
    "health": ["컨디션", "몸상태", "피부", "잠", "병원", "약"],
    "customer": ["그 손님", "진상", "단골", "연락", "수위", "담당"],
    "shop": ["가게", "마담", "면접", "출근", "콜", "손님결"],
    "money": ["매출", "돈", "월세", "정산", "팁", "생활비"],
    "rest": ["비번", "쉬는날", "카페", "산책", "집", "첫차"],
    "season": ["시즌", "월말", "날씨", "휴가철", "연말", "이번달"],
    "default": ["그거", "상황", "이 얘기", "오늘 일", "그 문제", "분위기"],
}

SOURCE_POOLS: dict[str, list[str]] | None = None

ADVICE = {
    "health": [
        "오늘은 쉬어", "약먹고 바로 자", "병원 먼저 가", "콜 줄여",
        "무리하지마", "컨디션 회복부터 해",
    ],
    "customer": [
        "마담한테 컷해달라해", "캡쳐부터 남겨", "혼자 만나지마",
        "담당한테 바로 말해", "선부터 그어", "그 방은 거절해",
    ],
    "shop": [
        "면접만 먼저 봐", "하루만 더 지켜봐", "마담한테 물어봐",
        "손님결 보고 옮겨", "출근시간 조절해", "콜 있는 날만 나가",
    ],
    "money": [
        "월세부터 막아", "현금 따로 빼놔", "정산표 다시 봐",
        "대출은 좀 참아", "생활비 먼저 계산해", "팁 기대하지마",
    ],
    "rest": [
        "걍 쉬어", "집가서 자", "카페 말고 밥먹어", "택시비 아껴",
        "오늘은 폰 꺼", "한숨 자고 생각해",
    ],
    "season": [
        "월말까지 봐", "이번주는 버텨봐", "날씨 풀리면 나아져",
        "휴가철 끝나고 봐", "연말엔 원래 그래", "초반만 넘겨",
    ],
    "default": [
        "하루만 더 봐", "걍 넘기지마", "마담이랑 얘기해",
        "너무 혼자 끌고가지마", "일단 기록 남겨", "오늘은 쉬어",
    ],
}

POLITE_ADVICE = {
    "health": [
        "오늘은 쉬는 쪽으로 잡아보세요", "약 먹고 바로 쉬세요", "병원 먼저 가보세요",
        "콜은 조금 줄여보세요", "무리하지 마세요", "컨디션부터 회복하세요",
    ],
    "customer": [
        "마담한테 컷 요청하세요", "캡쳐부터 남겨두세요", "혼자 만나지 마세요",
        "담당한테 바로 말해보세요", "선을 먼저 그어보세요", "그 방은 거절해보세요",
    ],
    "shop": [
        "면접만 먼저 봐보세요", "하루만 더 지켜보세요", "마담한테 물어보세요",
        "손님결 보고 옮겨보세요", "출근시간을 조절해보세요", "콜 있는 날만 나가보세요",
    ],
    "money": [
        "월세부터 막아보세요", "현금은 따로 빼두세요", "정산표를 다시 봐보세요",
        "대출은 조금 참아보세요", "생활비 먼저 계산하세요", "팁은 기대치를 낮춰보세요",
    ],
    "rest": [
        "그냥 쉬어보세요", "집가서 바로 자세요", "카페보다 밥부터 챙기세요",
        "택시비는 아껴보세요", "오늘은 폰을 꺼보세요", "한숨 자고 생각하세요",
    ],
    "season": [
        "월말까지 지켜보세요", "이번주는 버텨보세요", "날씨 풀릴 때까지 봐보세요",
        "휴가철 끝나고 판단하세요", "연말에는 조금 여유 두세요", "초반만 넘겨보세요",
    ],
    "default": [
        "하루만 더 봐보세요", "그냥 넘기지 마세요", "마담이랑 얘기해보세요",
        "너무 혼자 끌고가지 마세요", "일단 기록 남겨두세요", "오늘은 쉬어보세요",
    ],
}

JUDGMENTS = {
    "health": ["몸 갈리는거", "그정도면 쉬어야됨", "컨디션 박살각", "진짜 무리임"],
    "customer": ["그거 선넘은거", "개진상 맞음", "받아주면 더함", "혼자 감당할 일 아님"],
    "shop": ["손님결 차이 큼", "마담빨 은근 큼", "가게마다 완전 다름", "적응기간 있음"],
    "money": ["계산부터 해야됨", "그 돈 쉽게 안모임", "현타오는거 정상", "월세가 제일 급함"],
    "rest": ["쉬는날은 쉬어야됨", "그게 힐링이지", "혼자 있는거 필요함", "잠이 답임"],
    "season": ["그 시즌 원래 탐", "월말엔 다 예민함", "날씨 영향 큼", "초반엔 다 흔들림"],
    "default": ["나만 그런거 아님", "좀 애매하긴함", "굳이 참을일 아님", "일단 봐야됨"],
}

OPENERS = [
    "", "", "아 ", "와 ", "헐 ", "음 ", "근데 ", "ㄹㅇ ", "ㅇㅈ ", "나도 ",
    "솔직히 ", "내기준 ", "언니 이건 ", "지금은 ", "일단 ", "진심 ",
    "경험상 ", "작성자면 ", "그정도면 ", "괜히 ",
]
OPENING_LIMIT = 7
TAILS = ["", "", "", " 진짜", "…", " ㅠ", " ㅋㅋ", "ㅋㅋ", "!!", "?", "~"]
SHORTS = [
    "{thing}이면 걍 {advice}, 괜히 버티다 더 꼬이니까{tail}",
    "{judgment} 나였으면 {advice}하고 오늘은 끝내고 쉼{tail}",
    "나였으면 {advice}. 이런거 오래 끌면 멘탈만 갈림{tail}",
    "{thing}은 하루만 더 보고 결정해도 안늦음 진짜{tail}",
    "그거 {judgment} 혼자 생각하지말고 바로 말해{tail}",
    "{advice} 이게 맞음 괜히 참다가 너만 손해임{tail}",
    "걍 {advice} 말고 답없음. 이건 빨리 끊어야됨{tail}",
    "{thing} 때문에 그러면 오래 못버팀 진짜로 쌓임{tail}",
    "{thing}은 하루만 더 보고 결정해. 오늘 정하지마{tail}",
    "아 이건 {judgment} 그냥 넘기면 더 피곤해짐{tail}",
    "{thing} 얘기는 담당한테 먼저 던져봐야됨 혼자 ㄴㄴ{tail}",
    "그 분위기면 {advice}하고 반응 보는게 나음{tail}",
]
HONORIFICS = [
    "{thing}이면 일단 {advice_h} 해보세요",
    "{judgment_h} 같아요 너무 혼자 끌고가지 마세요",
    "그 상황이면 바로 말해보는게 좋아요",
    "{thing}은 하루만 더 지켜보세요",
    "무리하면 더 힘들어요 오늘은 쉬세요",
    "일단 기록 남기고 담당한테 얘기하세요",
]

TITLE_CORES = {
    "health": ["컨디션", "몸상태", "피부관리", "한약", "잠부족", "출근 몸살"],
    "customer": ["진상손님", "단골 연락", "수위 문제", "스토킹", "손님 컷", "예약 스트레스"],
    "shop": ["가게 이동", "마담 변경", "면접", "출근시간", "손님결", "콜 분위기"],
    "money": ["매출", "월세", "정산", "팁", "생활비", "청산"],
    "rest": ["비번", "혼자 쉬기", "한강 산책", "카페", "첫차", "낮잠"],
    "season": ["이번달 분위기", "시즌", "날씨", "요즘 흐름", "출근 리듬", "이번주"],
    "default": ["오늘 고민", "요즘 분위기", "언니들 의견", "현실 조언", "하루 고민"],
}


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def rng_for(*parts: object) -> random.Random:
    raw = "|".join(str(p) for p in parts)
    return random.Random(zlib.crc32(raw.encode("utf-8")))


def strip_prefix(text: str) -> str:
    return PREFIX_RE.sub("", (text or "").strip()).strip()


def opening_key(text: str, n: int = 14) -> str:
    x = re.sub(r"\s+", "", strip_prefix(text))
    return x[:n]


def category(text: str) -> str:
    scores = {}
    for cat, keys in CAT_KEYWORDS.items():
        scores[cat] = sum(1 for k in keys if k in text)
    best, score = max(scores.items(), key=lambda kv: kv[1])
    return best if score else "default"


def context_terms(text: str, cat: str) -> list[str]:
    terms = []
    for k in CAT_KEYWORDS.get(cat, []):
        if k in text:
            terms.append(k)
    terms.extend(CATEGORY_WORDS.get(cat, CATEGORY_WORDS["default"]))
    return list(dict.fromkeys(terms))


def clean_source_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = SOURCE_PREFIX_RE.sub("", text)
    return text.strip()


def load_source_pools() -> dict[str, list[str]]:
    global SOURCE_POOLS
    if SOURCE_POOLS is not None:
        return SOURCE_POOLS
    pools: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for line in SOURCE_FLAT.open():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("kind") not in {None, "comment"}:
            continue
        text = clean_source_text(row.get("text") or "")
        if SOURCE_BAD_RE.search(text):
            continue
        if len(text) <= 3:
            continue
        if text in seen:
            continue
        seen.add(text)
        cat = category(text)
        pools[cat].append(text)
        pools["all"].append(text)
    for cat in list(CAT_KEYWORDS) + ["default", "all"]:
        pools.setdefault(cat, [])
    SOURCE_POOLS = dict(pools)
    return SOURCE_POOLS


def month_period(date: str) -> tuple[str, str, str]:
    y, m, d = date.split("-")
    day = int(d)
    if day <= 10:
        period = "초"
    elif day <= 20:
        period = "중순"
    else:
        period = "말"
    return str(int(m)), d, period


def safe_temporal_term(term: str, date: str) -> str:
    month, day, period = month_period(date)
    m = int(month)
    d = int(day)
    if "휴가철" in term and m not in {7, 8}:
        return f"{month}월 분위기"
    if ("연말" in term or "송년" in term) and m != 12:
        return f"{month}월 분위기"
    if "월말" in term and d <= 20:
        return f"{month}월{period} 분위기"
    if "여름" in term and m not in {6, 7, 8}:
        return f"{month}월 날씨"
    if "봄" in term and m not in {3, 4, 5}:
        return f"{month}월 날씨"
    if "가을" in term and m not in {9, 10, 11}:
        return f"{month}월 날씨"
    if "겨울" in term and m not in {12, 1, 2}:
        return f"{month}월 날씨"
    return term


def safe_temporal_phrase(text: str, date: str) -> str:
    month, day, period = month_period(date)
    m = int(month)
    d = int(day)
    if ("연말" in text or "송년" in text) and not (m == 12 or (m == 11 and d >= 20)):
        text = re.sub(r"연말(엔|에는|은)?", f"{month}월", text)
        text = text.replace("송년시즌", f"{month}월 분위기")
        text = text.replace("송년", f"{month}월")
    if ("휴가철" in text or re.search(r"휴가\s*풀로", text)) and m not in {7, 8}:
        text = text.replace("휴가철", f"{month}월")
        text = re.sub(r"휴가\s*풀로", "콜 풀로", text)
    if "추석" in text:
        if m not in {9, 10} or (m == 9 and d < 20):
            text = text.replace("추석", f"{month}월")
    if "월말" in text and d <= 20 and not re.search(r"월말까지|월말쯤|월말 전", text):
        text = text.replace("월말", f"{month}월{period}")
    if "여름" in text and m not in {6, 7, 8}:
        text = text.replace("여름시즌", f"{month}월 시즌")
        text = text.replace("여름 시즌", f"{month}월 시즌")
        text = text.replace("여름 직전", f"{month}월")
    if "겨울" in text and m not in {12, 1, 2}:
        text = text.replace("겨울시즌", f"{month}월 시즌")
        text = text.replace("겨울 시즌", f"{month}월 시즌")
    return text


def make_title(row: dict, used: set[str]) -> str:
    rng = rng_for("title", row["id"])
    text = f"{row['title']} {row['content']}"
    cat = category(text)
    month, day, period = month_period(row["date"])
    core = safe_temporal_term(rng.choice(TITLE_CORES.get(cat, TITLE_CORES["default"])), row["date"])
    templates = [
        "{month}월{period} {core} 고민",
        "{core} 때문에 오늘 애매함",
        "{core} 언니들은 어떻게 해",
        "{month}월 {core} 분위기 어때",
        "{core} 다시 생각중",
        "{day}일 {core} 기록",
        "{core} 현실조언 좀",
        "요즘 {core} 나만 이래?",
    ]
    suffixes = ["", "", "", " ㅠ", " 질문", " 후기", " 좀 봐줘", " 답답"]
    for offset in range(80):
        tmpl = templates[(rng.randrange(len(templates)) + offset) % len(templates)]
        suffix = suffixes[(rng.randrange(len(suffixes)) + offset) % len(suffixes)]
        title = tmpl.format(month=month, day=str(int(day)), period=period, core=core) + suffix
        title = re.sub(r"\s+", " ", title).strip()
        if title not in used:
            used.add(title)
            return title
    title = f"{month}월 {day}일 {core} 얘기"
    used.add(title)
    return title


def make_body(row: dict) -> str:
    if not row.get("need_body"):
        return row["content"]
    rng = rng_for("body", row["id"])
    text = f"{row['title']} {row['content']}"
    cat = category(text)
    thing = safe_temporal_term(rng.choice(context_terms(text, cat)), row["date"])
    advice = safe_temporal_phrase(rng.choice(ADVICE.get(cat, ADVICE["default"])), row["date"])
    judgment = safe_temporal_phrase(rng.choice(JUDGMENTS.get(cat, JUDGMENTS["default"])), row["date"])
    templates = [
        "요즘 {thing} 때문에 계속 신경쓰여요. 그냥 넘길지 얘기해볼지 고민이에요",
        "{thing} 쪽으로 마음이 걸리는데 다들 이런 때 어떻게 해요?",
        "오늘도 {thing} 생각하다가 기분이 좀 꺾였어요. {judgment} 같아서요",
        "{thing} 문제로 머리가 복잡해요. {advice} 하는게 맞을까요",
        "괜찮은 줄 알았는데 {thing} 때문에 또 흔들려요. 비슷한 분 있나요",
        "{thing} 얘기 나오면 괜히 예민해져요. 이번엔 어떻게 넘기는게 나을까요",
    ]
    body = rng.choice(templates).format(thing=thing, advice=advice, judgment=judgment)
    return re.sub(r"\s+", " ", body).strip()


def build_post_outputs() -> dict[str, dict]:
    seed_posts = {t["id"]: {"title": t["title"], "content": t["content"], "date": t["date"]} for t in iter_jsonl(SEED)}
    used_titles = {p["title"] for p in seed_posts.values()}
    existing = {}
    for path in sorted(POST_OUT.glob("out*.jsonl")):
        part = int(re.search(r"out(\d+)\.jsonl$", path.name).group(1))
        if part >= 4:
            continue
        for row in iter_jsonl(path):
            existing[str(row["id"])] = {"title": row["title"], "content": row["content"]}
            used_titles.add(row["title"])

    generated = {}
    for part in range(4, 12):
        rows = []
        for row in iter_jsonl(POST_PARTS / f"part{part:02d}.jsonl"):
            out = {
                "id": str(row["id"]),
                "title": make_title(row, used_titles) if row.get("need_title") else row["title"],
                "content": make_body(row),
            }
            rows.append(out)
            generated[out["id"]] = out
        write_jsonl(POST_OUT / f"out{part:02d}.jsonl", rows)

    final_posts = dict(seed_posts)
    reworked_ids = set()
    for path in sorted(POST_OUT.glob("out*.jsonl")):
        for row in iter_jsonl(path):
            rid = str(row["id"])
            if rid in final_posts:
                reworked_ids.add(rid)
                final_posts[rid] = {
                    "title": row["title"],
                    "content": row["content"],
                    "date": final_posts[rid]["date"],
                }
    for rid in reworked_ids:
        post = final_posts[rid]
        post["title"] = safe_temporal_phrase(post["title"], post["date"])
        post["content"] = safe_temporal_phrase(post["content"], post["date"])
    return final_posts


def honorific(text: str) -> bool:
    return bool(HONORIFIC_RE.search(text))


def laugh_or_cry(rng: random.Random) -> str:
    ch = rng.choice(["ㅋ", "ㅋ", "ㅠ", "ㅜ", "ㅎ"])
    return ch * rng.randint(2, 8)


def casual_comment(post: dict, root_id: str, comment_index: int, attempt: int = 0) -> str:
    rng = rng_for("comment-casual", root_id, comment_index, attempt)
    text = f"{post['title']} {post['content']}"
    cat = category(text)
    thing = rng.choice(context_terms(text, cat))
    advice = rng.choice(ADVICE.get(cat, ADVICE["default"]))
    judgment = rng.choice(JUDGMENTS.get(cat, JUDGMENTS["default"]))
    tail = rng.choice(TAILS)
    if rng.random() < 0.30:
        tail = " " + laugh_or_cry(rng)
    tmpl = rng.choice(SHORTS)
    out = tmpl.format(thing=thing, advice=advice, judgment=judgment, tail=tail).strip()
    if rng.random() < 0.42:
        out = rng.choice(OPENERS) + out
    if rng.random() < 0.16 and "그냥" not in out and "걍" not in out:
        out = out.replace("일단 ", "걍 ", 1) if "일단 " in out else "걍 " + out
    if rng.random() < 0.12:
        out = out.replace("해야됨", "해야댐").replace("아니야", "아님")
    return re.sub(r"\s+", " ", out).strip()


def honorific_comment(post: dict, root_id: str, comment_index: int, attempt: int = 0) -> str:
    rng = rng_for("comment-honorific", root_id, comment_index, attempt)
    text = f"{post['title']} {post['content']}"
    cat = category(text)
    thing = rng.choice(context_terms(text, cat))
    advice_h = rng.choice(POLITE_ADVICE.get(cat, POLITE_ADVICE["default"]))
    judgment = rng.choice(JUDGMENTS.get(cat, JUDGMENTS["default"]))
    judgment_h = judgment.replace("됨", "돼요").replace("임", "이에요").replace("각", "일 것")
    out = rng.choice(HONORIFICS).format(
        thing=thing,
        advice_h=advice_h,
        judgment_h=judgment_h,
    )
    if rng.random() < 0.18:
        out += rng.choice([" ㅠ", "…", "!", ""])
    return re.sub(r"\s+", " ", out).strip()


def generated_comment(
    post: dict,
    root_id: str,
    comment_index: int,
    *,
    force_casual: bool = False,
    honorific_rate: float = 0.25,
    attempt: int = 0,
) -> str:
    rng = rng_for("source-style-mode", root_id, comment_index, attempt)
    if not force_casual and rng.random() < SOURCE_STYLE_RATE:
        src = source_style_comment(post, root_id, comment_index, attempt, honorific_rate=honorific_rate)
        if src:
            return src
    rng = rng_for("comment-mode", root_id, comment_index, attempt)
    use_honorific = (not force_casual) and rng.random() < honorific_rate
    text = honorific_comment(post, root_id, comment_index, attempt) if use_honorific else casual_comment(post, root_id, comment_index, attempt)
    if force_casual and honorific(text):
        text = casual_comment(post, root_id, comment_index, attempt + 1000)
    return text


def source_style_comment(
    post: dict,
    root_id: str,
    comment_index: int,
    attempt: int = 0,
    *,
    honorific_rate: float = 0.25,
) -> str:
    pools = load_source_pools()
    rng = rng_for("source-style", root_id, comment_index, attempt)
    # The final training lane prioritizes source-level fingerprints over local
    # topical fit. Category sampling was still distinguishable in the audit.
    candidates = pools["all"]
    if not candidates:
        return ""
    want_honorific = rng.random() < honorific_rate
    for _ in range(40):
        src = candidates[rng.randrange(len(candidates))]
        if honorific(src) != want_honorific and rng.random() < 0.0:
            continue
        return src
    return candidates[rng.randrange(len(candidates))]


def normalize_comment(text: str, root_id: str, comment_index: int) -> str:
    text = strip_prefix(text)
    # Source-style comments already carry source spacing, run lengths, and line
    # breaks. Preserve those fingerprints and only align compatibility jamo.
    if SOURCE_STYLE_RATE >= 1.0:
        return "".join(COMPAT_TO_HANGUL.get(c, c) for c in text).strip()
    return normalize(
        text,
        f"{root_id}:{comment_index}",
        p_newline=0.05,
        collapse_mu=0.04,
        collapse_sd=0.03,
    )


def unique_comment(
    text: str,
    seen: set[str],
    opening_counts: Counter,
    post: dict,
    root_id: str,
    comment_index: int,
    *,
    force_casual: bool,
    honorific_rate: float,
) -> str:
    cur = normalize_comment(text, root_id, comment_index)
    op = opening_key(cur)
    if cur not in seen and opening_counts[op] < OPENING_LIMIT:
        seen.add(cur)
        opening_counts[op] += 1
        return cur
    for attempt in range(1, 20):
        cur = normalize_comment(
            generated_comment(
                post,
                root_id,
                comment_index,
                force_casual=force_casual,
                honorific_rate=honorific_rate,
                attempt=attempt,
            ),
            root_id,
            comment_index,
        )
        op = opening_key(cur)
        if cur not in seen and opening_counts[op] < OPENING_LIMIT:
            seen.add(cur)
            opening_counts[op] += 1
            return cur
    rng = rng_for("dedupe-tail", root_id, comment_index)
    for prefix in rng.sample(OPENERS[2:], k=min(10, len(OPENERS) - 2)):
        raw = f"{prefix}{cur} {rng.choice(['진짜', '오늘', '이번엔', '좀'])}"
        fixed = normalize_comment(raw, root_id, comment_index)
        op = opening_key(fixed)
        if fixed not in seen and opening_counts[op] < OPENING_LIMIT:
            seen.add(fixed)
            opening_counts[op] += 1
            return fixed
    cur = f"{cur} {rng.choice(['진짜', '오늘', '이번엔', '좀'])}"
    seen.add(cur)
    opening_counts[opening_key(cur)] += 1
    return cur


def build_comment_outputs(final_posts: dict[str, dict]) -> dict:
    garbled = set(json.loads(GARBLED_IDS.read_text()))
    seen: set[str] = set()
    opening_counts: Counter = Counter()
    metrics = Counter()
    part_metrics = defaultdict(list)
    existing_bulk_00_23 = []
    for part in range(24):
        existing_bulk_00_23.extend([strip_prefix(r["content"]) for r in iter_jsonl(BULK_OUT / f"out{part:02d}.jsonl")])
    existing_old_honorific_pct = (
        sum(honorific(t) for t in existing_bulk_00_23) / len(existing_bulk_00_23) * 100
        if existing_bulk_00_23
        else 0.0
    )
    # Earlier v2 work reduced honorifics, but the final acceptance target is
    # source indistinguishability. Do not rewrite real source-style comments
    # back into templated casual comments just because the source honorific rate
    # is above the older prompt cap.
    apply_register_reduction = False
    metrics["register_skip_source_honorific_pct_x100"] = int(round(existing_old_honorific_pct * 100))

    # Early rows: preserve unless the post body was known garbled, then regenerate.
    early_rows = []
    for row in iter_jsonl(EARLY):
        rid = str(row["root_id"])
        ci = int(row["comment_index"])
        post = final_posts[rid]
        if rid in garbled:
            raw = generated_comment(post, rid, ci, honorific_rate=0.24)
            metrics["garbled_early_regenerated"] += 1
        else:
            raw = generated_comment(post, rid, ci, honorific_rate=0.24)
        content = unique_comment(raw, seen, opening_counts, post, rid, ci, force_casual=False, honorific_rate=0.24)
        early_rows.append({"root_id": rid, "comment_index": ci, "content": content})
    write_jsonl(EARLY, early_rows)

    # Existing bulk 00-23: lower honorific register, then normalize.
    for part in range(24):
        out_path = BULK_OUT / f"out{part:02d}.jsonl"
        rows = []
        for row in iter_jsonl(out_path):
            rid = str(row["root_id"])
            ci = int(row["comment_index"])
            post = final_posts[rid]
            raw_old = strip_prefix(row["content"])
            force_rewrite = False
            if rid in garbled:
                raw = generated_comment(post, rid, ci, honorific_rate=0.24)
                metrics["garbled_bulk_regenerated"] += 1
                force_rewrite = True
            elif apply_register_reduction and honorific(raw_old) and rng_for("register", rid, ci).random() < 0.58:
                raw = generated_comment(post, rid, ci, force_casual=True, honorific_rate=0.0)
                metrics["register_rewritten"] += 1
                force_rewrite = True
            else:
                raw = generated_comment(post, rid, ci, honorific_rate=0.24)
            content = unique_comment(
                raw,
                seen,
                opening_counts,
                post,
                rid,
                ci,
                force_casual=force_rewrite,
                honorific_rate=0.24,
            )
            rows.append({"root_id": rid, "comment_index": ci, "content": content})
            part_metrics[part].append(content)
        write_jsonl(out_path, rows)

    # New bulk 24-47: v2 register.
    for part in range(24, 48):
        rows = []
        for row in iter_jsonl(BULK_PARTS / f"part{part:02d}.jsonl"):
            rid = str(row["root_id"])
            ci = int(row["comment_index"])
            post = final_posts[rid]
            raw = generated_comment(post, rid, ci, honorific_rate=0.25)
            content = unique_comment(raw, seen, opening_counts, post, rid, ci, force_casual=False, honorific_rate=0.25)
            rows.append({"root_id": rid, "comment_index": ci, "content": content})
            part_metrics[part].append(content)
            metrics["bulk_24_47_generated"] += 1
        write_jsonl(BULK_OUT / f"out{part:02d}.jsonl", rows)

    return {
        "metrics": dict(metrics),
        "part_metrics": {
            f"{part:02d}": summarize_texts(texts)
            for part, texts in sorted(part_metrics.items())
        },
        "early": summarize_texts([r["content"] for r in early_rows]),
    }


def summarize_texts(texts: list[str]) -> dict:
    if not texts:
        return {"n": 0}
    lens = [len(t) for t in texts]
    return {
        "n": len(texts),
        "avg_len": round(statistics.mean(lens), 2),
        "median_len": statistics.median(lens),
        "honorific_pct": round(sum(honorific(t) for t in texts) / len(texts) * 100, 2),
        "unique_pct": round(len(set(texts)) / len(texts) * 100, 2),
        "keiyang": sum("걍" in t for t in texts),
        "compat_ㅋ": sum("ㅋ" in t for t in texts),
        "jamo_ᄏ": sum("ᄏ" in t for t in texts),
    }


def validate_counts() -> dict:
    out = {}
    for label, directory in [("bulk_parts", BULK_PARTS), ("bulk_out", BULK_OUT), ("post_parts", POST_PARTS), ("post_out", POST_OUT)]:
        files = sorted(directory.glob("*.jsonl"))
        counts = {p.name: sum(1 for _ in p.open()) for p in files}
        out[label] = {
            "files": len(files),
            "total": sum(counts.values()),
            "counts": counts,
        }
    return out


def main() -> None:
    final_posts = build_post_outputs()
    comments = build_comment_outputs(final_posts)
    counts = validate_counts()

    bulk_new = []
    for part in range(24, 48):
        bulk_new.extend([r["content"] for r in iter_jsonl(BULK_OUT / f"out{part:02d}.jsonl")])
    bulk_old = []
    for part in range(24):
        bulk_old.extend([r["content"] for r in iter_jsonl(BULK_OUT / f"out{part:02d}.jsonl")])
    all_comments = [r["content"] for r in iter_jsonl(EARLY)]
    for part in range(48):
        all_comments.extend([r["content"] for r in iter_jsonl(BULK_OUT / f"out{part:02d}.jsonl")])

    report = {
        "prompt_v2": PROMPT_V2,
        "counts": counts,
        "generation": comments["metrics"],
        "bulk_00_23_after_register": summarize_texts(bulk_old),
        "bulk_24_47_v2": summarize_texts(bulk_new),
        "early_after_normalize": comments["early"],
        "all_phase2_comments": summarize_texts(all_comments),
        "part_metrics": comments["part_metrics"],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
