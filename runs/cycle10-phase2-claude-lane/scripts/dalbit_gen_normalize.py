#!/usr/bin/env python3
"""Claude 생성 텍스트 → 원천 사이트 분포 정규화 패스.

원천 실측 (threads_flat, 2026-06-11 진단):
- 호환 자모 0% / 옛한글 자모(U+1100) 40% 함유 → 전 호환자모 변환
- newline 함유율 37% → 절 경계 개행 주입
- space_ratio 0.162 (gen 0.231) → 댓글별 가변 띄어쓰기 붕괴
- '걍' 사용 → 일부 '그냥'→'걍' 치환
모든 난수는 (key) 시드 — 재현 가능.
"""
import random
import re
import zlib

COMPAT_TO_HANGUL = {
    'ㄱ': 'ᄀ', 'ㄲ': 'ᄁ', 'ㄴ': 'ᄂ', 'ㄷ': 'ᄃ', 'ㄸ': 'ᄄ', 'ㄹ': 'ᄅ', 'ㅁ': 'ᄆ',
    'ㅂ': 'ᄇ', 'ㅃ': 'ᄈ', 'ㅅ': 'ᄉ', 'ㅆ': 'ᄊ', 'ㅇ': 'ᄋ', 'ㅈ': 'ᄌ', 'ㅉ': 'ᄍ',
    'ㅊ': 'ᄎ', 'ㅋ': 'ᄏ', 'ㅌ': 'ᄐ', 'ㅍ': 'ᄑ', 'ㅎ': 'ᄒ',
    'ㅏ': 'ᅡ', 'ㅐ': 'ᅢ', 'ㅑ': 'ᅣ', 'ㅒ': 'ᅤ', 'ㅓ': 'ᅥ', 'ㅔ': 'ᅦ', 'ㅕ': 'ᅧ',
    'ㅖ': 'ᅨ', 'ㅗ': 'ᅩ', 'ㅘ': 'ᅪ', 'ㅙ': 'ᅫ', 'ㅚ': 'ᅬ', 'ㅛ': 'ᅭ', 'ㅜ': 'ᅮ',
    'ㅝ': 'ᅯ', 'ㅞ': 'ᅰ', 'ㅟ': 'ᅱ', 'ㅠ': 'ᅲ', 'ㅡ': 'ᅳ', 'ㅢ': 'ᅴ', 'ㅣ': 'ᅵ',
}

CLAUSE_RE = re.compile(r"((?:요|다|죠|함|임|음|셈|네요|어요|아요|는데|지만|니까)[.!?~ᅲᅮᄏᄒ]*) ")

# 원천 ᄏrun 분포: {1:0.14, 2:0.29, 3:0.20, 4:0.13, 5:0.07, 6+:0.17}
KRUN_POP = [1, 2, 3, 4, 5, 6, 7, 8]
KRUN_W = [14, 29, 20, 13, 7, 9, 5, 3]

def _resample_runs(text: str, ch: str, rng) -> str:
    return re.sub(ch + "{2,}", lambda m: ch * rng.choices(KRUN_POP[1:], KRUN_W[1:])[0], text)

def normalize(
    text: str,
    key: str,
    p_newline: float = 0.5,
    collapse_mu: float = 0.30,
    collapse_sd: float = 0.18,
) -> str:
    rng = random.Random(zlib.crc32(key.encode()) ^ 0x7A30)
    # 1. 자모 변환 (원천 인코딩과 동일하게)
    text = "".join(COMPAT_TO_HANGUL.get(c, c) for c in text)
    # 1b. 웃음/울음 run 길이를 원천 분포로 리샘플
    for ch in ("ᄏ", "ᄒ", "ᅲ", "ᅮ"):
        text = _resample_runs(text, ch, rng)
    # 2. '그냥' → '걍' 일부 치환 (40%)
    if "그냥" in text and rng.random() < 0.4:
        text = text.replace("그냥", "걍", 1)
    # 3. 개행 주입: 긴 댓글의 절 경계 1곳 (원천 newline 함유율 ~37%)
    if len(text) > 34 and "\n" not in text and rng.random() < p_newline:
        matches = list(CLAUSE_RE.finditer(text))
        if matches:
            m = rng.choice(matches)
            text = text[: m.end(1)] + "\n" + text[m.end(1) + 1:]
    # 4. 띄어쓰기 붕괴: 댓글별 가변 비율 (전체 space_ratio ~0.16 목표)
    collapse = max(0.0, rng.gauss(collapse_mu, collapse_sd))
    out = []
    for ch in text:
        if ch == " " and rng.random() < collapse:
            continue
        out.append(ch)
    return "".join(out).strip()

if __name__ == "__main__":
    import json, sys, glob, statistics
    srcs = sys.argv[1:-1]
    out_path = sys.argv[-1]
    n = 0
    with open(out_path, "w") as fo:
        for pat in srcs:
            for f in sorted(glob.glob(pat)):
                for line in open(f):
                    r = json.loads(line)
                    key = f"{r['root_id']}:{r['comment_index']}"
                    r["content"] = normalize(r["content"], key)
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
    print(f"normalized {n} -> {out_path}")
