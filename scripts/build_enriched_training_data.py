#!/usr/bin/env python3
"""
build_enriched_training_data.py — 학습 데이터 확장 빌더

원천 크롤 데이터에서 다음 관계를 학습할 수 있는 데이터셋 생성:
  1. 제목 ↔ 본문 적합성
  2. 본문/제목 → 예상 조회수/댓글수 (인기도 감각)
  3. 글에 적합한 댓글 (맥락 적합성)
  4. 댓글 reply 흐름 (대화 구조)
  5. 은어/초성 자연스러운 사용

출력:
  - cpt_enriched.jsonl: 메타데이터 포함 CPT 코퍼스
  - sft_enriched.jsonl: 제목-본문-댓글-인기도 관계 SFT 쌍

Usage:
    # SSH로 데스크탑의 원본 데이터 접근
    python scripts/build_enriched_training_data.py --crawl-host desktop-c6hsmq5

    # 로컬에 이미 복사된 경우
    python scripts/build_enriched_training_data.py --crawl-dir /path/to/crawled-data-v2
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Obsidian DATA-QUALITY-SPEC 기반 프로모 필터
PROMO_RE = re.compile(
    r'카카오\s*(?:톡\s*)?(?:아이디|ID)?\s*[:：\s]\s*[A-Za-z0-9_\-]{3,}|'
    r'카톡\s*(?:아이디|ID)?\s*[:：\s]\s*[A-Za-z0-9_\-]{3,}|'
    r'텔레?\s*[:：]\s*@?\S{3,20}|'
    r'라인\s*[:：]\s*\S{3,20}|'
    r'오픈\s*(?:채팅|카톡|톡)|'
    r'밀빵\s*(?:확실|가능|보장)|'
    r'풀\s*(?:상주|케어)|'
    r'출근\s*문의',
    re.IGNORECASE
)

PII_PHONE = re.compile(r'010[- .]?\d{3,4}[- .]?\d{4}')
PII_KAKAO_ID = re.compile(r'카카오톡\s*-\s*[A-Za-z0-9_]{3,}')


def is_promo(text: str) -> bool:
    return bool(PROMO_RE.search(text))


def mask_pii(text: str) -> str:
    text = PII_PHONE.sub('[전화번호]', text)
    text = PII_KAKAO_ID.sub('카톡 [KAKAO]', text)
    return text


def load_crawl_data(crawl_dir: str = None, crawl_host: str = None) -> list[dict]:
    """Load raw crawl data from local dir or SSH remote"""
    if crawl_dir:
        import glob
        files = sorted(Path(crawl_dir).glob('*.json'))
        all_posts = []
        for f in files:
            data = json.loads(f.read_text(encoding='utf-8'))
            all_posts.extend(data)
        return all_posts

    if crawl_host:
        cmd = f'''ssh {crawl_host} "python3 -c \\"
import json, glob
files = sorted(glob.glob('/mnt/c/Users/mapdr/Desktop/queenalba-crawler/crawled-data-v2/*.json'))
all_posts = []
for f in files:
    data = json.load(open(f))
    all_posts.extend(data)
print(json.dumps(all_posts))
\\""'''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"SSH error: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        return json.loads(result.stdout)

    raise ValueError("Either --crawl-dir or --crawl-host required")


def rescale_views_for_new_site(original_views: int, post_date: str = None) -> int:
    """
    원본 조회수를 신생사이트 현실에 맞게 리스케일.
    - 초기(1~2개월): 2~30회
    - 성장기(3~6개월): 20~150회
    - 안정기(6개월+): 100~500회
    지수적 성장 곡선 적용.
    """
    import math
    import random
    # Normalize original views to a 0-1 percentile
    # Original data: median ~1800, p95 ~2400
    percentile = min(1.0, original_views / 3000)

    # Simulate site age progression (random for training variety)
    # site_month: 1~12 (uniformly sampled for diverse training data)
    site_month = random.randint(1, 12)

    # Exponential growth base: month 1 → base 15, month 12 → base 400
    growth_base = 15 * math.exp(0.27 * (site_month - 1))

    # Apply percentile spread (popular posts get more views)
    spread = 0.3 + percentile * 1.5  # 0.3x ~ 1.8x of base
    views = int(growth_base * spread * (0.7 + random.random() * 0.6))

    return max(2, views)


def view_bucket(views: int) -> str:
    """신생사이트 기준 조회수 버킷"""
    if views < 10: return "신규"
    if views < 50: return "소수"
    if views < 150: return "보통"
    if views < 500: return "인기"
    return "핫"


def comment_bucket(count: int) -> str:
    """댓글수 버킷"""
    if count == 0: return "무반응"
    if count <= 2: return "소수"
    if count <= 5: return "보통"
    if count <= 15: return "활발"
    return "폭발"


def clean_comment_text(raw: str) -> str:
    """댓글에서 reply 태그를 정리하되 구조는 보존"""
    # [1-1] 비회원 [1]  -> 답글 구조 보존
    # 하지만 순수 텍스트로 정리
    text = re.sub(r'^\[\d+(-\d+)*\]\s*', '', raw.strip())
    text = re.sub(r'^비회원\s*\[\d+(-\d+)*\]\s*', '', text)
    return text.strip()


def build_cpt_enriched(posts: list[dict]) -> list[dict]:
    """메타데이터 포함 CPT 코퍼스"""
    rows = []
    for post in posts:
        content = post.get('content', '').strip()
        title = post.get('title', '').strip()
        if not content or not title:
            continue
        if is_promo(content) or is_promo(title):
            continue

        original_views = int(post.get('views', 0))
        comment_count = int(post.get('commentCount', 0))
        views = rescale_views_for_new_site(original_views, post.get('date'))

        # Post with metadata context
        meta_prefix = f"[조회수:{view_bucket(views)}|댓글:{comment_bucket(comment_count)}] "
        text = mask_pii(f"{title}\n{content}")

        rows.append({
            "text": meta_prefix + text,
            "kind": "post",
            "source_id": post.get('id', ''),
            "views": views,
            "comment_count": comment_count,
            "view_bucket": view_bucket(views),
            "comment_bucket": comment_bucket(comment_count),
        })

        # Comments with context
        comments = post.get('comments', [])
        for i, comment in enumerate(comments):
            c_text = comment.get('content', '').strip()
            if not c_text or is_promo(c_text):
                continue

            cleaned = clean_comment_text(c_text)
            if len(cleaned) < 3:
                continue

            cleaned = mask_pii(cleaned)

            # Build context: title + content excerpt + parent comment if reply
            is_reply = re.match(r'\[\d+-\d+\]', c_text)
            parent_ref = None
            if is_reply and i > 0:
                # Find parent comment
                parent_tag = re.match(r'\[(\d+)\]', c_text.split('-')[0] + ']')
                if parent_tag:
                    parent_idx = int(parent_tag.group(1)) - 1
                    if 0 <= parent_idx < len(comments):
                        parent_ref = clean_comment_text(comments[parent_idx].get('content', ''))

            context_parts = [f"제목: {title}"]
            context_parts.append(f"원글: {content[:100]}")
            if parent_ref:
                context_parts.append(f"부모댓글: {parent_ref[:80]}")
            context_parts.append(f"댓글: {cleaned}")

            rows.append({
                "text": mask_pii("\n".join(context_parts)),
                "kind": "context_comment" if parent_ref else "comment",
                "source_id": post.get('id', ''),
                "views": views,
                "is_reply": bool(is_reply),
            })

    return rows


def build_sft_enriched(posts: list[dict]) -> list[dict]:
    """제목-본문-댓글-인기도 관계 SFT 쌍"""
    rows = []
    for post in posts:
        content = post.get('content', '').strip()
        title = post.get('title', '').strip()
        comments = post.get('comments', [])
        if not content or not title:
            continue
        if is_promo(content) or is_promo(title):
            continue
        if not comments:
            continue

        original_views = int(post.get('views', 0))
        comment_count = int(post.get('commentCount', 0))
        views = rescale_views_for_new_site(original_views, post.get('date'))

        # Clean comments
        clean_comments = []
        for c in comments:
            c_text = c.get('content', '').strip()
            if not c_text or is_promo(c_text):
                continue
            cleaned = clean_comment_text(c_text)
            if len(cleaned) >= 3:
                clean_comments.append(mask_pii(cleaned))

        if not clean_comments:
            continue

        # SFT pair: post → best comment (first non-promo)
        rows.append({
            "instruction": f"[{view_bucket(views)}] {mask_pii(title)}",
            "input": mask_pii(content[:300]),
            "output": clean_comments[0],
            "kind": "title_content_comment",
            "source_id": post.get('id', ''),
            "views": views,
            "view_bucket": view_bucket(views),
            "comment_count": comment_count,
        })

        # Reply chain pairs (if multiple comments)
        for i in range(1, min(len(clean_comments), 5)):
            prev = clean_comments[i-1]
            curr = clean_comments[i]
            rows.append({
                "instruction": f"제목: {mask_pii(title[:50])}",
                "input": prev[:200],
                "output": curr,
                "kind": "reply_chain",
                "source_id": post.get('id', ''),
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl-dir', type=str, help='Local crawl data directory')
    parser.add_argument('--crawl-host', type=str, default='desktop-c6hsmq5',
                        help='SSH host with crawl data')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print("Loading crawl data...")
    posts = load_crawl_data(args.crawl_dir, args.crawl_host)
    print(f"Loaded {len(posts)} posts")

    # Filter posts with content
    posts = [p for p in posts if isinstance(p, dict) and p.get('content') and p.get('title')]
    print(f"Posts with content+title: {len(posts)}")

    print("\nBuilding enriched CPT...")
    cpt_rows = build_cpt_enriched(posts)
    print(f"CPT enriched rows: {len(cpt_rows)}")

    # Stats
    kinds = Counter(r['kind'] for r in cpt_rows)
    print(f"  kinds: {dict(kinds)}")
    view_buckets = Counter(r.get('view_bucket', '') for r in cpt_rows if r.get('view_bucket'))
    print(f"  view buckets: {dict(view_buckets)}")

    print("\nBuilding enriched SFT...")
    sft_rows = build_sft_enriched(posts)
    print(f"SFT enriched rows: {len(sft_rows)}")
    sft_kinds = Counter(r['kind'] for r in sft_rows)
    print(f"  kinds: {dict(sft_kinds)}")

    if not args.dry_run:
        cpt_path = PROJECT_ROOT / "cpt_enriched.jsonl"
        sft_path = PROJECT_ROOT / "sft_enriched.jsonl"

        with open(cpt_path, 'w', encoding='utf-8') as f:
            for row in cpt_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        with open(sft_path, 'w', encoding='utf-8') as f:
            for row in sft_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        print(f"\nWritten: {cpt_path} ({len(cpt_rows)} rows)")
        print(f"Written: {sft_path} ({len(sft_rows)} rows)")
    else:
        print("\n[DRY RUN] No files written")
        # Show samples
        print("\n=== CPT Sample ===")
        print(json.dumps(cpt_rows[0], ensure_ascii=False, indent=2)[:300])
        print("\n=== SFT Sample ===")
        print(json.dumps(sft_rows[0], ensure_ascii=False, indent=2)[:300])


if __name__ == '__main__':
    main()
