#!/usr/bin/env python3
"""
clean_context_stream_promo.py — Obsidian DATA-QUALITY-SPEC 기반 프로모 정리

대상: cpt_context_stream.jsonl (활성 budget30_v2 학습 입력)
기준: Obsidian ai-detection-research/_system/DATA-QUALITY-SPEC.md

--dry-run: 통계만 출력, 파일 미수정
--classify-tc: TC 관련 행을 프로모/일반으로 분류해서 별도 출력
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "cpt_context_stream.jsonl"
BACKUP_DIR = PROJECT_ROOT / "archive" / "dataset-backups"

# ── Obsidian DATA-QUALITY-SPEC 기반 패턴 ──
# PII 마스킹 (제거가 아닌 치환)
PII_PATTERNS = [
    (re.compile(r'010[- .]?\d{3,4}[- .]?\d{4}'), '[전화번호]', 'phone'),
    (re.compile(r'\S+@\S+\.\S+'), '[이메일]', 'email'),
    (re.compile(r'\d{2,4}-\d{2,4}-\d{4,7}'), '[계좌]', 'account'),
]

# 프로모/광고 제거 패턴 (행 자체를 제거)
PROMO_REMOVE_PATTERNS = [
    # 카카오/텔레그램/라인 연락처
    (r'카카오\s*(?:톡\s*)?(?:아이디|ID)?\s*[:：\s]\s*[A-Za-z0-9_\-]{3,}', 'kakao_id'),
    (r'카톡\s*(?:아이디|ID)?\s*[:：\s]\s*[A-Za-z0-9_\-]{3,}', 'kakao_id'),
    (r'카카오\s+[a-zA-Z][a-zA-Z0-9]{2,}', 'kakao_bare'),
    (r'카톡\s+[a-zA-Z][a-zA-Z0-9]{2,}', 'kakao_bare'),
    (r'[Kk]akao\s*(?:ID)?\s*[:：]\s*[A-Za-z0-9_\-]{3,}', 'kakao_eng'),
    (r'텔레?\s*[:：]\s*@?\S{3,20}', 'telegram_id'),
    (r'라인\s*[:：]\s*\S{3,20}', 'line_id'),
    (r'오픈\s*(?:채팅|카톡|톡)', 'openchat'),
    # 채용 템플릿
    (r'밀빵\s*(?:확실|가능|보장)', 'recruiter_milbbang'),
    (r'풀\s*(?:상주|케어)', 'recruiter_fullcare'),
    (r'출근\s*문의', 'recruiter_inquiry'),
    (r'스타트톡\s*개수톡', 'recruiter_starttalk'),
    (r'하루\s*평균\s*\d+\s*방', 'recruiter_rooms'),
    (r'문의\s*(?:주|하)\s*(?:세요|십시오|시면)', 'solicitation'),
    (r'[1-3]부\s*\d+인\s*\d+조', 'recruiter_shift'),
    (r'\d+방\s*팀사장', 'recruiter_team'),
    (r'G팀\s*\d+방', 'recruiter_gteam'),
    # 연락처 혼합
    (r'\[전화번호\].*카[카톡]', 'phone_kakao_combo'),
    (r'카[카톡].*\[전화번호\]', 'kakao_phone_combo'),
    # 급여/조건 유인
    (r'(?:지명비|보장|일급|시급)\s*\d+\s*(?:만원|만)', 'salary_lure'),
    (r'(?:출근부터|퇴근까지)\s*.{0,15}(?:케어|관리|안전)', 'care_template'),
]

PROMO_RE = re.compile('|'.join(p[0] for p in PROMO_REMOVE_PATTERNS), re.IGNORECASE)

# TC 관련 — 제거하지 않고 분류만
TC_RE = re.compile(r'T\.?C|티씨|팁제', re.IGNORECASE)

# TC + 프로모 혼합 패턴 (이건 제거 대상)
TC_PROMO_RE = re.compile(
    r'(?:T\.?C|티씨|팁제).{0,30}(?:문의|연락|카[카톡]|실장|출근)|'
    r'(?:문의|연락|카[카톡]|실장|출근).{0,30}(?:T\.?C|티씨|팁제)',
    re.IGNORECASE
)


def classify_row(text: str) -> tuple[str, str | None]:
    """
    Returns (action, reason)
    action: 'keep' | 'remove' | 'mask'
    """
    # 1. 프로모 패턴 매칭
    m = PROMO_RE.search(text)
    if m:
        # Find which pattern matched
        for pat, name in PROMO_REMOVE_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return 'remove', name
        return 'remove', 'promo_generic'

    # 2. TC + 프로모 혼합
    if TC_PROMO_RE.search(text):
        return 'remove', 'tc_promo_combo'

    # 3. 순수 TC 언급은 유지 (업계 일반 용어)
    # 4. PII 마스킹 (제거 아닌 치환)
    return 'keep', None


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Apply PII masking, return (masked_text, list_of_masked_types)"""
    masked_types = []
    for pattern, replacement, pii_type in PII_PATTERNS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            masked_types.append(pii_type)
    return text, masked_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--classify-tc', action='store_true',
                        help='Output TC rows classification to tc_classification.jsonl')
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(INPUT_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    print(f"Input: {INPUT_FILE.name} ({total:,} rows)")

    # Backup
    if not args.dry_run:
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup = BACKUP_DIR / f'context-stream-promo-{ts}'
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(INPUT_FILE, backup / INPUT_FILE.name)
        print(f"Backup: {backup}")

    # Process
    kept = []
    removed = []
    reason_counts = Counter()
    pii_counts = Counter()
    tc_rows = []

    for row in rows:
        text = row.get('text', '')

        # PII masking first
        masked_text, pii_types = mask_pii(text)
        if pii_types:
            row['text'] = masked_text
            for pt in pii_types:
                pii_counts[pt] += 1

        # Classify
        action, reason = classify_row(row['text'])

        if action == 'remove':
            removed.append(row)
            reason_counts[reason] += 1
        else:
            kept.append(row)

        # TC classification
        if args.classify_tc and TC_RE.search(text):
            tc_rows.append({
                'text': text[:200],
                'action': action,
                'reason': reason,
                'source_id': row.get('source_id', ''),
                'kind': row.get('kind', ''),
            })

    # Write
    if not args.dry_run:
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    if args.classify_tc and tc_rows:
        tc_file = PROJECT_ROOT / 'runs' / 'tc_classification.jsonl'
        with open(tc_file, 'w', encoding='utf-8') as f:
            for row in tc_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"\nTC classification: {tc_file} ({len(tc_rows)} rows)")
        tc_removed = sum(1 for r in tc_rows if r['action'] == 'remove')
        tc_kept = sum(1 for r in tc_rows if r['action'] == 'keep')
        print(f"  TC removed (프로모 혼합): {tc_removed}")
        print(f"  TC kept (일반 대화): {tc_kept}")

    # Report
    print(f"\n{'='*60}")
    print(f"PROMO CLEANING REPORT — cpt_context_stream.jsonl")
    print(f"{'='*60}")
    print(f"  Before: {total:>8,}")
    print(f"  Removed: {len(removed):>8,}  ({len(removed)/total*100:.2f}%)")
    print(f"  Kept:    {len(kept):>8,}")
    print()
    print("  Removal reasons:")
    for reason, count in reason_counts.most_common():
        print(f"    {reason:25s} {count:>6,}")
    print()
    print("  PII masking applied:")
    for pii_type, count in pii_counts.most_common():
        print(f"    {pii_type:25s} {count:>6,}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
