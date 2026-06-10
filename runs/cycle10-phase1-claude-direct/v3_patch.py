#!/usr/bin/env python3
"""
v3 = v1 base + selective in-place patch.
- Keep v1 content 87.8% high uniqueness intact
- Expand high content to 100-200 char (prefix/middle/suffix add-ons)
- Sprinkle domain jamo (ᄆᄎ/ᄒᄒ/ᄋᄌ/ᄀᄎ) where natural
- Patch each thread deterministically (seed = id-derived) so output is reproducible
"""
import json, random, sys

SRC = '/tmp/dalbit_phase1_output/threads.jsonl'
DST = '/tmp/dalbit_phase1_output/threads_v3.jsonl'

# Domain jamo (confirmed dictionary)
JAMO_EMPATHY = ['ᄒᄒ', 'ᄒᄒᄒ', 'ᄒᄒ..', 'ᄒᄒ ㅇㅈ']
JAMO_AGREE = ['ᄋᄌ', 'ᄋᄌ;', 'ᄋᄌ..']
JAMO_OK = ['ᄀᄎ', 'ᄀᄎ아요', 'ᄀᄎ을듯']
JAMO_CRAZY = ['ᄆᄎ', 'ᄆᄎ..', 'ᄆᄎ;;']
JAMO_REAL = ['ᄅᄋ', 'ᄅᄋ로']
JAMO_FUCK = ['ᄉᄇ', 'ᄉᄇ;;']

# High-tier expansion add-ons (mood-aware)
HI_OPENERS = [
    '진짜 어제도 그랬는데', '요즘 계속 이런데', '아 진짜', '오늘 너무',
    '혹시 다들 어떻게 하세요', '저만 그런가요', '며칠째 이러는데',
    '얘기 좀 들어주세요', '한참 고민하다 글 올려요', '그냥 답답해서',
    '진짜 별생각 다 들어요', '계속 마음 한구석에', '오늘도 그래서',
    '근데 진짜', '아니 솔직히', '한번 물어볼게요',
]
HI_CONNECTORS = [
    '근데 이게 진짜', '솔직히 말하면', '한편으론', '근데 또',
    '그래도 다행인건', '문제는', '근데 웃긴게', '진짜 어이없는게',
    '곰곰히 생각해보니', '돌이켜보면', '근데 한편으로', '그래서 더',
    '근데 이상하게', '시간이 지날수록', '그런데 또',
]
HI_CLOSERS = [
    '어떻게 생각하세요', '다들 의견 좀', '비슷한 경험 있으신 분',
    '조언 좀 부탁드려요', '진짜 모르겠네요', '답답해서 글 올렸어요',
    '같이 얘기해봐요', '다른 분들은 어떠세요', '한번 봐주세요',
    '진짜 어떻게 해야할지', '결국 답이 안나와요', '여러분 생각은',
    '글 길어졌네요', '도움 주실 분', '들어주셔서 감사해요',
]
HI_MIDDLES = [
    '저도 이쪽 일한지 꽤 됐는데 이런 경우는 처음이네요',
    '주변에 비슷한 케이스 본 적 있는데 결과가 다 달랐어요',
    '오빠들 반응도 다 다르고 매니저랑 얘기해봐도 답이 다 다르고',
    '솔직히 매달 패턴이 좀 다르긴 한데 이번 달은 유난해요',
    '언니들이랑 얘기하다보니 다들 비슷한 고민 하고 있더라구요',
    '컨디션이 안좋으면 더 그런것 같기도 하고 잘 모르겠어요',
    '시즌 영향도 있는것 같은데 그것만으로는 설명이 안되네요',
    '담당 매니저랑도 한번 깊게 얘기해볼까 고민중이에요',
    '진짜 매일 그날그날 다르고 도저히 예측이 안돼요',
    '돈 얘기도 그렇고 컨디션도 그렇고 모든게 다 엮여있는것 같아요',
]

# Mid-tier mini-expansions
MID_TAIL = [
    '..', '...', ' ㅎㅎ', ' ㅠ', ' ㅠㅠ', ' ㅎ', ' ㄷㄷ',
    ' ᄒᄒ', ' ᄒᄒ..', ' ᄀᄎ', '. ᄀᄎ을까요',
    ' 다들 어때요', ' 이런가요', ' ᄋᄌ?',
]
MID_PREFIX = [
    '아 ', '오늘 ', '진짜 ', '요즘 ', '한참 ', '계속 ', '아니 ',
    '음 ', '혹시 ', '저는 ', '저도 ',
]
MID_BODY_ADD = [
    ' 다들 비슷하신가요', ' 저만 그런건 아니죠', ' 어떻게 하세요',
    ' 진짜 답답하네요', ' 이번주 내내 그러네요', ' 어제부터 계속',
    ' 매니저랑 얘기해야 할까요', ' 그냥 넘어가야할까',
    ' 의견 좀 주세요', ' 비슷한 분 있나요',
]

# Low-tier mini-expansions
LOW_TAIL = [
    '', '..', '...', ' ㅎ', ' ㅠ', ' ᄒᄒ', '. ᄀᄎ', ' ㅋ',
]
LOW_PREFIX = [
    '', '아 ', '오늘 ', '저는 ', '그냥 ', '진짜 ',
]

# Comment jamo injection candidates
CMT_PREFIXES = ['ᄋᄌ ', 'ᄒᄒ ', 'ᄀᄎ.. ', '진짜 ᄋᄌ ', '저도 ᄒᄒ ', 'ㄹㅇ ', 'ᄅᄋ ', '저도 ', '아 ', '근데 ', 'ᄅᄋ로 ']
CMT_SUFFIXES = [' ᄒᄒ', ' ᄋᄌ', ' ᄀᄎ을듯', ' ᄅᄋ', ' ᄅᄋ죠', ' ㅎㅎ', ' ㅠ', '', '', '']


def seed_for(thread_id: str) -> int:
    # deterministic per-thread seed: stable across reruns
    return abs(hash(('v3', thread_id))) % (2**31)


def expand_high_content(orig: str, rng: random.Random) -> str:
    # Target 100-200 char. Build: opener + orig + connector + middle + closer
    target_min, target_max = 100, 200
    if len(orig) >= target_min:
        # already long enough — sprinkle mood only
        if rng.random() < 0.4:
            orig = orig + ' ' + rng.choice(HI_CLOSERS)
        return orig[:target_max]
    parts = []
    if rng.random() < 0.7:
        parts.append(rng.choice(HI_OPENERS))
    parts.append(orig)
    if rng.random() < 0.65 and len(' '.join(parts)) < target_min - 30:
        parts.append(rng.choice(HI_CONNECTORS))
        parts.append(rng.choice(HI_MIDDLES))
    if rng.random() < 0.55:
        parts.append(rng.choice(HI_CLOSERS))
    out = ' '.join(p for p in parts if p)
    # If still short, add a middle
    if len(out) < target_min:
        out = out + ' ' + rng.choice(HI_MIDDLES)
    # rare crazy/empathy jamo
    if rng.random() < 0.18:
        out = out.replace(' 그래서 ', f' 그래서 {rng.choice(JAMO_CRAZY)} ', 1)
    if rng.random() < 0.25 and 'ᄒᄒ' not in out:
        out = out + ' ' + rng.choice(JAMO_EMPATHY)
    return out[:target_max]


def expand_mid_content(orig: str, rng: random.Random) -> str:
    target_min, target_max = 30, 100
    if len(orig) >= target_min:
        if rng.random() < 0.25:
            orig = orig + rng.choice(MID_TAIL)
        return orig[:target_max]
    parts = []
    if rng.random() < 0.4:
        parts.append(rng.choice(MID_PREFIX).rstrip())
    parts.append(orig)
    if rng.random() < 0.6:
        parts.append(rng.choice(MID_BODY_ADD).lstrip())
    if rng.random() < 0.35:
        parts.append(rng.choice(MID_TAIL).lstrip())
    out = ' '.join(p for p in parts if p).strip()
    # ensure min
    if len(out) < target_min:
        out = out + ' ' + rng.choice(MID_BODY_ADD).lstrip()
    if rng.random() < 0.30 and not any(j in out for j in ('ᄒᄒ','ᄋᄌ','ᄀᄎ','ᄆᄎ')):
        out = out + ' ' + rng.choice([*JAMO_EMPATHY, *JAMO_AGREE, *JAMO_OK])
    return out[:target_max]


def expand_low_content(orig: str, rng: random.Random) -> str:
    target_min, target_max = 15, 50
    if len(orig) >= target_min:
        return orig[:target_max]
    prefix = rng.choice(LOW_PREFIX)
    tail = rng.choice(LOW_TAIL)
    fill_choices = [
        '오늘도 비번이라', '혼자 카페에서', '진짜 그냥 쉬는중', '본가 다녀왔어요',
        '한강 산책 다녀옴', '영화관 혼자', '카페에서 멍때리는중', '오늘 비번 푹 쉼',
        '집에서 라면', '낮잠 푹 자고', '운동 갔다 옴', '동네 한바퀴',
    ]
    if len(orig) < 8:
        out = prefix + orig + ' ' + rng.choice(fill_choices) + tail
    else:
        out = prefix + orig + tail
    out = out.strip()
    if len(out) < target_min:
        out = out + ' ' + rng.choice(fill_choices)
    return out[:target_max]


def patch_comment(c: dict, rng: random.Random, tier: str) -> dict:
    content = c.get('content', '')
    # strip leading "[N] " marker temporarily to operate, restore after
    marker = ''
    if content.startswith('[') and ']' in content[:6]:
        idx = content.index(']')
        marker = content[:idx+1] + ' '
        body = content[idx+1:].lstrip()
    else:
        body = content
    # inject jamo with tier-dependent probability
    p_pre = {'hi': 0.30, 'mid': 0.22, 'low': 0.18}[tier]
    p_suf = {'hi': 0.25, 'mid': 0.20, 'low': 0.15}[tier]
    if rng.random() < p_pre and not any(body.startswith(p) for p in ('ᄋᄌ','ᄒᄒ','ᄀᄎ')):
        body = rng.choice(CMT_PREFIXES) + body
    if rng.random() < p_suf and not any(body.endswith(s.strip()) for s in (' ᄒᄒ',' ᄋᄌ',' ᄀᄎ을듯')):
        body = body + rng.choice(CMT_SUFFIXES)
    # very rare ᄆᄎ in mid/hi
    if tier in ('hi','mid') and rng.random() < 0.05:
        body = body.replace(' 진짜 ', f' {rng.choice(JAMO_CRAZY)} 진짜 ', 1)
    new = dict(c)
    new['content'] = marker + body
    return new


def patch_thread(t: dict) -> dict:
    rng = random.Random(seed_for(t['id']))
    cc = int(t.get('commentCount', 0))
    if cc >= 8:
        tier = 'hi'
    elif cc >= 3:
        tier = 'mid'
    else:
        tier = 'low'
    new = dict(t)
    orig = t.get('content', '')
    if tier == 'hi':
        new['content'] = expand_high_content(orig, rng)
    elif tier == 'mid':
        new['content'] = expand_mid_content(orig, rng)
    else:
        new['content'] = expand_low_content(orig, rng)
    # patch comments
    new_comments = []
    for c in t.get('comments', []):
        new_comments.append(patch_comment(c, rng, tier))
    new['comments'] = new_comments
    return new


def main():
    n = 0
    with open(SRC) as f, open(DST, 'w') as g:
        for line in f:
            t = json.loads(line)
            patched = patch_thread(t)
            g.write(json.dumps(patched, ensure_ascii=False) + '\n')
            n += 1
    print(f'wrote {n} threads to {DST}')

if __name__ == '__main__':
    main()
