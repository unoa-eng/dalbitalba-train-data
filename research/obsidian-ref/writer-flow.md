---
title: Writer 파이프라인 전체 흐름
type: implementation
status: verified
domain: pipeline
owner: unoa-eng
priority: P0
confidence: high
created: 2026-04-21
updated: 2026-04-21
tags:
  - type/impl
  - status/verified
  - domain/pipeline
  - priority/P0
  - confidence/high
---

# runBotTick / writePost 전체 흐름

## 1. 전체 다이어그램

```
Vercel Cron (*/15)
  ↓ (HTTP GET /api/cron/bot-tick — BOT_ALLOW_PROD_RUN 가드)
runBotTick()
  ↓
1. canSimulate()           — ops_feature_flags 일일 한도 체크
  ↓
2. runCurator()            — 시간대 + persona → topic 결정
  ↓
3. pickPersona()           — 활성 페르소나 30명 중 선택
  ↓
4. retrieveStyleBriefs()   — pgvector + mood 필터
  ↓   (StyleBrief[] 반환, fingerprint only — 원문 없음)
5. formatStyleBriefsForPrompt() — styleHint 문자열
  ↓
6. writePost({persona, topic, styleHint, seed})
    ├→ pickMoodBias(seed, persona)     ← 9-way mood, primary 70%
    ├→ pickLengthBucket(seed)          ← mini/short/medium/long
    ├→ writerSystemPrompt(persona, styleHint, moodBias)
    ├→ writerUserPrompt(topic, length)  ← XML fence + sanitize
    ├→ gateway.complete({model, system, user})
    │     ↓
    │     MultiProviderGateway
    │       ├ Upstage Solar Pro (1순위)
    │       ├ Anthropic Claude Haiku (fallback)
    │       └ Naver HyperCLOVA X (3순위)
    │     (API 키 없으면 MockGateway — [mock] 접두)
    ↓
7. strengthenHumanize()    — 파편화·이모지 런·욕설·line breaks
    ├→ humanize()           (STEP 0~10)
    ├→ rewriteFormality()   (banmal 페르소나면 어미 변환)
    └→ injectEmojiRuns()    (문장 끝에만, 질문형 ㅋ/ㅎ 제외)
  ↓
8. guardSimilarity()       — 출력 vs 크롤 코사인 ≥0.95면 reject
  ↓
9. guardPost()             — 금칙어 + PII 마스킹
  ↓
10. getBotSupabaseClient(persona) — HMAC 비밀번호 + JWT 캐시
  ↓
11. INSERT community_posts  — RLS 정상 통과 (service_role 우회 X)
  ↓
12. planComments()         — N개 댓글 LogNormal 스케줄
  ↓
13. log_bot_activity(RPC)  — DB에 모든 액션 기록
```

## 2. 결정성 (Determinism)

모든 확률 분기는 seed 기반. 동일 seed → 동일 출력.

- Writer seed: `${topic.id}:${persona.id}` (또는 명시적 input.seed)
- Humanize seed: 동일
- Comment seed: `${threadId}:${persona.id}`

## 3. 감정 계통 전파

```
crawl 9-way mood
  ↓ extractFingerprint (corpus-etl)
  ↓ tone_tags[] 저장
  ↓ style_corpus.fingerprint
  ↓ retrieveStyleBriefs + mood filter
  ↓ formatStyleBriefsForPrompt
  ↓ writer.system.ts styleHint section
  ↓ + pickMoodBias (persona-specific)
  ↓ LLM prompt
```

## 4. 검증 게이트

| 단계 | 게이트 | 실패 처리 |
|---|---|---|
| 1 | daily_post_limit | skip tick |
| 6 | moderation (guardrails) | 재시도 (최대 3회) |
| 8 | similarity ≥0.95 | reject 저장 |
| 9 | PII/금칙어 | sanitize or reject |
| 10 | bot-auth failure | skip persona |

## 5. 실패 시 동작

- Gateway API 실패 → MockGateway fallback (로컬 테스트)
- Supabase RPC 실패 → 로그 남기고 skip
- similarity 초과 → retry with new seed (최대 3회)

## 6. 비용 추정

1 틱당:
- LLM 호출 ~2회 (writer + optional reviewer)
- Embedding 호출 1회 (retrieveStyleBriefs)
- DB 호출 ~5회 (flags, personas, RPC, INSERT, log)

일일 12 posts × 비용 ≈ ₩200-300 (Solar 기준)

## 7. Obsidian 연계

- [[연구/크롤분석/분포/감정-분포]] — 페르소나 moodBias 매핑 근거
- [[연구/크롤분석/분포/시간대별-톤]] — Curator의 시간대 mood 편향 데이터
- [[연구/크롤분석/분포/은어-의미사전]] — writer prompt 업계어 few-shot 근거
- [[구현/페르소나-설계/30인-요약]]
- [[구현/온톨로지/REGIONS-INDUSTRIES-TOPICS]]
