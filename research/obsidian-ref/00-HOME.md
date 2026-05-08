---
title: 🏠 볼트 홈 (AI Detection Research)
type: moc
status: verified
domain: system
owner: unoa-eng
priority: P0
confidence: high
scope: vault
created: 2026-04-21
updated: 2026-04-21
tags: [type/moc, status/verified, domain/system, priority/P0]
---

# 🏠 AI Detection Research — 엔터프라이즈 볼트

> **목적**: 한국어 커뮤니티 AI 생성글의 **실질적 구분 불가** 수준 달성·실증
> **현재 성과**: Claude Opus 4.7 블라인드 60% (random 50%), 통계 점수 0.739 / 60.4% 탐지율
> **체계 버전**: Enterprise v1.0 (2026-04-21)

[![status](https://img.shields.io/badge/status-verified-34D399)](#)
[![priority](https://img.shields.io/badge/priority-P0-FF5757)](#)
[![coverage](https://img.shields.io/badge/crawl-11,288-60A5FA)](#)

---

## 🗺️ 허브 (Maps of Content)

| 허브 | 범위 | 링크 |
|---|---|---|
| 🛠️ 구현 | 페르소나·파이프라인·온톨로지·하네스 | [[구현/00-MOC]] |
| 🔬 연구 | 크롤·선행연구·실험 | [[연구/00-MOC]] |
| 🧪 실험 | humanize·프롬프트 튜닝 로그 | [[연구/실험/00-MOC]] |
| 📊 크롤 분석 | 11,288 포스트·55,849 댓글 | [[연구/크롤분석/00-MOC]] |
| 📝 논문 | 섹션별 초안 | [[논문/00-MOC]] |
| 🗣️ 회의·로그 | 일일·주간·회의록 | [[회의-및-로그/00-MOC]] |
| 📐 체계 | 택사노미·스키마·가이드 | [[_system/00-TAXONOMY]] |

---

## 🚦 프로젝트 상태 대시보드

### 검증된 P0 결과
```dataview
TABLE status, confidence, file.mtime AS "수정"
FROM "연구" OR "구현"
WHERE priority = "P0" AND status = "verified"
SORT file.mtime DESC
LIMIT 10
```

### 진행 중 작업
```dataview
TABLE type, domain, priority
FROM ""
WHERE status = "active" OR status = "review"
SORT priority ASC, file.mtime DESC
LIMIT 10
```

### 최근 실험
```dataview
TABLE verdict, result, baseline, metric
FROM "연구/실험"
WHERE type = "experiment"
SORT file.mtime DESC
LIMIT 5
```

### 초안/TODO
```dataview
LIST
FROM ""
WHERE status = "draft"
  AND !contains(file.folder, "주제별-샘플")
  AND !contains(file.folder, "_templates")
SORT file.mtime DESC
LIMIT 15
```

---

## 📐 체계 (Governance)

### 핵심 스펙
- [[_system/00-TAXONOMY|볼트 택사노미]] — 타입·상태·도메인 정의
- [[_system/01-FRONTMATTER-SCHEMA|Frontmatter 스키마]] — 필수/권장 필드
- [[_system/02-TAG-HIERARCHY|태그 계층]] — 5차원 nested
- [[_system/03-STYLE-GUIDE|스타일 가이드]] — 색상·callout·표

### 엔터프라이즈 프로세스 (P0)
- [[_system/OMNI-AUDIT-FRAMEWORK|OMNI-AUDIT 12관점 누락 발굴 프레임워크]] ⭐
- [[_system/AUDIT-LEDGER|감사 사이클 레저]]
- [[_system/API-KEY-CONFIG|API 키 구성 — 탐지 회피 실효 기준 (시나리오 B 최소)]] ⭐
- [[_system/HUMAN-EVAL-PROTOCOL|인간 구분 불가 검증 프로토콜]] ⭐
- [[_system/SECURITY-POSTURE|보안 posture — 자산 3-Tier · GitHub 공개 범위]] 🔐

### 🔐 보안 게이트 (실수 커밋 방지 + 예산 락)
| 게이트 | 효과 |
|---|---|
| `.gitignore` (Tier 1) | 민감 파일 스테이징 자체 차단 |
| `pre-commit hook` | 이미 스테이징된 민감 패턴·API 키 형식 감지 후 commit 차단 |
| `pre-push hook` | OMNI-AUDIT + 블라인드 dry-run |
| `gateway.ts::wrapWithCostGuard` | 모든 AI 호출 전 월 예산 체크 + 비용 로깅 (**자동**) |

설치 한 번:
```bash
cd apps/web/scripts/analysis && make install-hooks
```

> **자동 실행 조건** (전부 세션 독립):
> - 🤖 **GitHub Actions**: 매주 월요일 09:00 KST + main push 시
> - 🪝 **pre-push hook**: `git push` 마다 (설치: `make install-hooks`)
> - 🖐️ **수동**: `cd apps/web/scripts/analysis && make cycle` (OMNI + 블라인드 n=50 페어드)

- [[_system/VAULT-HEALTH|Vault 건강 — 규모 vs 성능 vs 충분성]] ⭐
- [[_system/E2E-AUDIT-L3|End-to-End 탐지 회피 감사 (L3)]]
- [[_system/GAP-ANALYSIS-2026-04-21|Gap 분석 (16개 누락 프로세스)]]
- [[_system/DATA-QUALITY-SPEC|데이터 품질 (PII·중복·유해)]]
- [[_system/EVAL-PROTOCOL|평가 프로토콜 (multi-judge·adversarial·FP)]]
- [[_system/AI-DISCLOSURE-POLICY|AI Disclosure 정책]]
- [[_system/REPRODUCIBILITY|Reproducibility 체크리스트]]
- [[_system/COST-MONITORING|비용 모니터링]]
- [[_system/ROLLBACK-PLAYBOOK|Rollback·Canary Playbook]]

### ⚠️ Schema Violation 감지 (자동)
```dataview
LIST
FROM ""
WHERE (!status OR !type OR !domain)
  AND file.name != "00-HOME"
  AND !contains(file.folder, "주제별-샘플")
  AND !contains(file.folder, "_templates")
  AND !contains(file.folder, "attachments")
SORT file.mtime DESC
LIMIT 20
```
→ 결과가 있으면 해당 노트에 `type/status/domain` 필드를 [[_system/01-FRONTMATTER-SCHEMA|스키마]] 기준으로 추가하세요.

## 📋 템플릿 (Templater)

- [[_templates/experiment|실험 노트]]
- [[_templates/research-note|연구 노트]]
- [[_templates/literature-review|선행연구]]
- [[_templates/implementation-note|구현 노트]]
- [[_templates/daily-log|일일 로그]]
- [[_templates/meeting|회의록]]
- [[_templates/weekly-review|주간 회고]] ⭐
- [[_templates/postmortem|Postmortem]] ⭐

---

## 🔑 핵심 산출물

| 파일 (repo) | 역할 |
|---|---|
| `apps/web/lib/ai/humanize.ts` | 통계적 humanize 변환기 |
| `apps/web/lib/ai/personas.ts` | 30인 페르소나 + moodBias |
| `apps/web/lib/ai/writer.ts` | Claude Sonnet 호출 + RAG |
| `apps/web/lib/ai/rag.ts` | pgvector + mood filter |
| `apps/web/lib/ai/hybrid-retrieval.ts` | BM25 + RRF |
| `apps/web/lib/ai/reranker.ts` | Cross-encoder rerank |
| `apps/web/scripts/llm-judge.mjs` | LLM-as-a-Judge 스켈레톤 |
| `apps/web/scripts/run-indist-regression.mjs` | 회귀 게이트 |

> 저장소: `unoa-labs/dalbitalba` · 브랜치: `feat/ai-indistinguishability-refine`

---

## ⚙️ 플러그인

- **Dataview** ✅ — 본 홈의 자동 쿼리
- **Templater** ✅ — `_templates/` 기반 신규 노트
- **Smart Connections** (선택) — [[START-HERE#Smart Connections 수동 활성화]]

---

## 📅 최근 갱신 (전체)

```dataview
TABLE status, type, domain, dateformat(file.mtime, "MM-dd HH:mm") AS "mtime"
FROM ""
WHERE file.name != "00-HOME"
  AND !contains(file.folder, "주제별-샘플")
  AND !contains(file.folder, "_templates")
SORT file.mtime DESC
LIMIT 12
```

---

## 📥 인박스 워크플로우

1. 착상·메모는 `0-INBOX/YYYY-MM-DD.md` 로 즉시 기록 (Templater: `_templates/daily-log`)
2. **매주 금요일** 정리:
   - 실험/발견 → `연구/` 해당 폴더로 이동 (frontmatter 스키마 적용)
   - 회의/로그 → `회의-및-로그/YYYY-MM/` 로 이동
   - 폐기 → `status: archived` 표시 후 유지
