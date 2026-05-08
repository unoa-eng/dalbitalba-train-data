---
title: 데이터 품질 스펙 — PII·중복·유해 필터
type: spec
status: active
domain: system
owner: unoa-eng
priority: P0
confidence: high
created: 2026-04-21
updated: 2026-04-21
tags:
  - type/spec
  - status/active
  - domain/system
  - priority/P0
---

# 🛡️ 데이터 품질 스펙

> 크롤 원본과 AI 생성 결과 양 방향에서 PII 노출·중복 도배·유해 콘텐츠를 차단하는 규정.
> Gap 분석 A-1 해소.

---

## 1. 적용 범위

| 단계 | 적용 대상 |
|---|---|
| Stage-In | 크롤 데이터 로드 시 (`corpus-etl.ts`, `analysis/*.py`) |
| Stage-Out | AI 생성 직후 (`guardrails.ts`, `similarity-guard.ts`) |
| Archive | Obsidian 샘플 저장 시 (`expand-samples.py`) |

---

## 2. PII 필터 (전수)

### 마스킹 대상
| 항목 | 정규식 | 치환 |
|---|---|---|
| 휴대폰 번호 | `010[- ]?\d{3,4}[- ]?\d{4}` | `[PHONE]` |
| 카카오톡 ID | `카톡\s*[:：]?\s*\S{3,20}` | `카톡 [KAKAO]` |
| 계좌번호 | `\d{2,4}-\d{2,4}-\d{4,7}` | `[ACCOUNT]` |
| 이메일 | `\S+@\S+\.\S+` | `[EMAIL]` |
| 주소 | (서울/부산/… + 구/동/번지) | `[ADDRESS]` |
| 실명 추정 | 한글 2-4자 + 님/씨 (화이트리스트 제외) | 유지 (닉네임 관례) |
| 업장 실명 | `\S{2,10}(주점\|클럽\|라운지\|룸)` | 유지 (업계 일반명) |

### 구현
```ts
// apps/web/lib/ai/pii-sanitize.ts (신규 생성 예정)
export function sanitizePII(text: string): string {
  return text
    .replace(/010[- ]?\d{3,4}[- ]?\d{4}/g, '[PHONE]')
    .replace(/카톡\s*[:：]?\s*\S{3,20}/g, '카톡 [KAKAO]')
    .replace(/\d{2,4}-\d{2,4}-\d{4,7}/g, '[ACCOUNT]')
    .replace(/\S+@\S+\.\S+/g, '[EMAIL]');
}
```

### 검증
- 샘플 100건 수동 검수 후 적용
- Stage-Out 이후 0 PII 노출 목표 (0.1% 허용 오류)

---

## 3. 중복·도배 감지

### 중복 정의
| 수준 | 판정 기준 | 액션 |
|---|---|---|
| Exact | md5(content) 동일 | 폐기 |
| Near (L1) | cosine(emb) ≥ 0.95 | 기존 코드 `similarity-guard` 재사용 |
| Near (L2) | 3-gram Jaccard ≥ 0.7 | 로그 남기고 1개만 유지 |
| 도배 | 동일 author 1시간 내 10+ | 전부 archive |

### 크롤 원본 적용
```python
# 크롤 로드 시 dedupe (expand-samples.py 확장)
import hashlib
seen = set()
for p in posts:
    h = hashlib.md5(p['content'].encode()).hexdigest()
    if h in seen:
        continue
    seen.add(h)
```

### 측정 대상
- 현재 크롤 11,288건에 Exact 중복 n건? (미측정 — 다음 사이클)

---

## 4. 유해 컨텐츠 사전 차단

### 차단 카테고리 (Perspective API·KoBBQ 기준)
| 카테고리 | 허용 | 예외 사유 |
|---|---|---|
| 혐오 발언 (성별·인종·장애) | ❌ | 커뮤니티 규정 위배 |
| 자해·자살 직접 유도 | ❌ | 안전 |
| 미성년자 성적 묘사 | ❌ | 법적 |
| 불법 약물·무기 거래 | ❌ | 법적 |
| 개인 신상 특정 | ❌ | PII 위임 |
| 일반 욕설 (ㅅㅂ, 개새 등) | ✅ (커뮤니티 일상어) | 도메인 현실성 |
| 업계 은어 (초이스, 쩜오) | ✅ | 도메인 전문어 |

### 구현 layer
1. Writer 프롬프트 system 섹션에 "금지 카테고리" 명시 ✅ (이미 존재)
2. 생성 직후 `guardrails.ts` 키워드 매칭 ✅
3. (신규) HuggingFace `korean-hate-speech` 분류기 옵션 호출
4. (신규) 월 1회 KoBBQ 편향 감사

---

## 5. Obsidian 샘플 저장 시

### 현재 상태 (확장 후)
- 1,546 stratified 샘플 저장 — PII 마스킹 **미적용** ⚠️
- 재배포 금지 경고는 `START-HERE.md` 에만

### 개선
- `expand-samples.py` 에 `sanitize_for_vault(text)` 호출 추가
- 샘플 frontmatter 에 `pii_sanitized: true` 필드 추가

---

## 6. 운영 측정 지표

| 지표 | 목표 | 측정 주기 |
|---|---|---|
| PII 노출 0건 (Stage-Out) | 100% | 주 1회 수동 |
| Near-dup 비율 | < 2% | 생성 즉시 |
| 유해 차단률 | > 99% | 월 1회 회고 |
| 샘플 PII 마스킹 준수 | 100% | 변경 시 |

---

## 7. 체크리스트 (적용 전)

- [ ] `pii-sanitize.ts` 생성 및 `writer.ts`, `commenter.ts` 호출
- [ ] `expand-samples.py` 에 PII 마스킹 프리패스 추가
- [ ] 크롤 원본 Exact 중복 측정 후 로그
- [ ] `guardrails.ts` 유해 키워드 리스트 확장
- [ ] 월간 회고 템플릿에 "유해·PII 감사" 섹션

---

## 🔗 관련

- [[_system/GAP-ANALYSIS-2026-04-21]]
- [[_system/AI-DISCLOSURE-POLICY]]
- [[_system/EVAL-PROTOCOL]]
