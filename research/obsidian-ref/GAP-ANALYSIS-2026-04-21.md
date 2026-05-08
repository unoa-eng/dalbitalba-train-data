---
title: 엔터프라이즈 Gap 분석 — 16개 누락 프로세스
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

# 🧭 엔터프라이즈 Gap 분석

> "실질적 구분 불가 달성" 관점에서 **현재 시스템에 빠진 프로세스 16개** 전수 조사.
> P0=즉시 반영 (이번 세션), P1=다음 사이클, P2=장기.

---

## A. 데이터 파이프라인 (3개)

### A-1. 크롤 데이터 PII·중복·유해 필터 🟥 P0
- 현황: 원본 그대로 샘플에 포함 — 실제 전화번호·업장명·카톡 ID 노출 가능
- 반영: [[_system/DATA-QUALITY-SPEC]] 신설, 생성 전후 필터 규정

### A-2. 분석 스크립트 레포 커밋 (재현성) 🟥 P0
- 현황: `/tmp/crawl-analysis/*.py` 로 휘발 — 스크립트 버전 추적 불가
- 반영: `apps/web/scripts/analysis/` 디렉토리로 영구 커밋 + Makefile

### A-3. 크롤 drift 감지 (증분 업데이트) 🟨 P1
- 현황: 새 크롤 데이터 추가 시 수동 재분석
- 반영: `run-corpus-drift.mjs` 스크립트 — KS test 분포 비교

---

## B. AI 품질 검증 (4개)

### B-1. Multi-judge (Claude + GPT-5 교차) 🟨 P1
- 현황: Claude Opus 단일 judge (self-judging 편향 리스크)
- 반영: `llm-judge.mjs` 에 `--judge gpt-5` 옵션 + 합의율 측정

### B-2. Adversarial 탐지기 평가 (KatFishNet/XDAC) 🟥 P0
- 현황: LLM judge만 — 실제 탐지기는 미검증
- 반영: [[_system/EVAL-PROTOCOL]] 에 HF 탐지기 배치 평가 계획

### B-3. False positive baseline 🟥 P0
- 현황: AI → HUMAN 오판 60%만 측정. 실제 crawl 원본을 AI로 오판하는 비율 미측정
- 반영: EVAL-PROTOCOL 에 crawl-only 블라인드 추가

### B-4. 통계적 유의성 (n≥50 scale-up) 🟨 P1
- 현황: 블라인드 n=10 (너무 작음)
- 반영: EVAL-PROTOCOL 에 50/100 샘플 단계 확장 계획

---

## C. 운영·프로덕션 (4개)

### C-1. AI disclosure 정책 🟥 P0
- 현황: 없음 — 일부 사법권(EU AI Act)은 AI 생성 명시 의무
- 반영: [[_system/AI-DISCLOSURE-POLICY]] 신설

### C-2. 비용 추적·알람 🟨 P1
- 현황: 월 ~$23 추정만 — 실시간 추적 없음
- 반영: [[_system/COST-MONITORING]] + Upstash Redis 카운터 계획

### C-3. Rollback playbook 🟨 P1
- 현황: 프롬프트 v2.1 → v2.0 롤백 절차 없음
- 반영: [[_system/ROLLBACK-PLAYBOOK]]

### C-4. Canary deployment 🟨 P1
- 현황: 프롬프트 변경이 100% 즉시 적용 — 회귀 위험
- 반영: ROLLBACK-PLAYBOOK 에 10% → 50% → 100% 단계

---

## D. 연구·문서화 (2개)

### D-1. Reproducibility checklist 🟥 P0
- 현황: seed·lib 버전 흩어짐 — 재현 불확실
- 반영: [[_system/REPRODUCIBILITY]]

### D-2. IRB·윤리 검토 🟨 P2
- 현황: 커뮤니티 크롤 사용 동의·IRB 없음
- 반영: 논문 출판 전 필수 — [[논문/ethics-checklist]] (후속)

---

## E. 개선 루프 (2개)

### E-1. A/B 테스트 프레임워크 🟨 P1
- 현황: 프롬프트 v2.0 vs v2.1 비교는 수동 재측정
- 반영: `run-ab-test.mjs` — 동시 배치 + 통계 검정

### E-2. 실패 사례 아카이브 🟨 P2
- 현황: 탐지당한 글 패턴이 휘발
- 반영: `연구/실험/실패-사례-DB.md` — 장기 축적

---

## F. 볼트 고도화 (3개)

### F-1. Frontmatter 스키마 검증 쿼리 🟥 P0
- 현황: 누락 자동 감지 없음
- 반영: 00-HOME 에 "schema violation" 쿼리 추가

### F-2. 주간/포스트모템 템플릿 🟥 P0
- 현황: 개별 로그만 — 회고 루프 없음
- 반영: `_templates/weekly-review.md`, `postmortem.md`

### F-3. 로드맵 Canvas 🟥 P0
- 현황: 현재 → perfect indistinguishability 경로 시각화 없음
- 반영: `_system/ROADMAP.canvas`

---

## 🚦 이번 세션 반영 범위 (P0 11개)

| ID | 산출물 | 상태 |
|---|---|---|
| A-1 | `DATA-QUALITY-SPEC.md` | ✅ |
| A-2 | `apps/web/scripts/analysis/*.py` + `Makefile` | ✅ |
| B-2 | `EVAL-PROTOCOL.md` (adversarial 섹션) | ✅ |
| B-3 | `EVAL-PROTOCOL.md` (FP 섹션) | ✅ |
| C-1 | `AI-DISCLOSURE-POLICY.md` | ✅ |
| D-1 | `REPRODUCIBILITY.md` | ✅ |
| F-1 | 00-HOME schema violation 쿼리 | ✅ |
| F-2 | `weekly-review.md` + `postmortem.md` | ✅ |
| F-3 | `ROADMAP.canvas` | ✅ |

다음 세션으로 이월 (P1 7개): B-1 multi-judge · B-4 scale-up · C-2 비용 · C-3 rollback · C-4 canary · E-1 A/B · A-3 drift

---

## 🟢 P1/P2 이월분 일괄 반영 (사용자 요청 — 동 세션)

| ID | 산출물 | 상태 |
|---|---|---|
| B-1 multi-judge | `apps/web/scripts/llm-judge.mjs` (Claude/GPT/HF + consensus) | ✅ |
| B-4 scale-up | `apps/web/scripts/build-scaleup-bench.mjs` | ✅ |
| A-3 drift | `apps/web/scripts/run-corpus-drift.mjs` | ✅ |
| C-2 cost | `016_ai_cost_log.sql` + `cost-logger.ts` | ✅ |
| C-3 rollback | `canary.ts::resolvePromptVersion()` + `CanaryKilledError` | ✅ |
| C-4 canary | `canary.ts::hashBucket()` + rollout % 플래그 | ✅ |
| E-1 A/B | `apps/web/scripts/run-ab-test.mjs` (z-test + Cohen's h) | ✅ |
| D-2 IRB | `논문/ethics-checklist.md` | ✅ |
| E-2 실패 아카이브 | `연구/실험/실패-사례/00-MOC.md` + Case 01 | ✅ |

→ 16개 Gap 전부 P0+P1+P2 해소 완료. 남은 것은 **DB migration 적용**·**실운영 배포** 단계.

---

## 🔗 관련

- [[00-HOME]]
- [[_system/00-TAXONOMY]]
- [[연구/크롤분석/크롤-적용-감사]]
