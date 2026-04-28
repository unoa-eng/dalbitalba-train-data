---
title: 평가 프로토콜 — Multi-judge·Adversarial·False Positive·Scale-up
type: spec
status: active
domain: harness
owner: unoa-eng
priority: P0
confidence: high
created: 2026-04-21
updated: 2026-04-21
tags:
  - type/spec
  - status/active
  - domain/harness
  - priority/P0
---

# 🔬 평가 프로토콜

> Indistinguishability 를 **편향 없이** 검증하기 위한 공식 절차.
> Gap 분석 B-2 · B-3 · B-4 해소.

---

## 1. 평가 프레임 (4-축)

```
         AI 생성            HUMAN 원본
        ┌─────────┐        ┌─────────┐
Judge ◀─┤         │        │         ├─▶ Judge
        │   AI    │        │  Crawl  │
        │(targets)│        │ (orig)  │
        └─────────┘        └─────────┘
             ▲                  ▲
             │                  │
        [잘못된 AI 판정]     [잘못된 HUMAN 판정]
         = Recall            = False Positive
```

**핵심 지표 4종**:
1. AI→AI 정답률 (TP)
2. AI→HUMAN 오판 (FN) — 현재 측정 대상
3. HUMAN→HUMAN 정답률 (TN)
4. **HUMAN→AI 오판 (FP)** — 현재 미측정 ⚠️

**완전 구분 불가** = Judge 정확도 = 50%(random chance) = FP율 ≈ FN율.

---

## 2. Judge 구성 (Multi-judge)

| Judge | 모델 | 역할 | 상태 |
|---|---|---|---|
| J1 | Claude Opus 4.7 | 1차 판정 | ✅ 구현 |
| J2 | GPT-5 | 교차 검증 | ⬜ 다음 사이클 |
| J3 | Hello-SimpleAI/chatgpt-detector-roberta | 실탐지기 baseline | ⬜ |
| J4 | KatFishNet (ACL 2025) | 한국어 특화 탐지기 | ⬜ 재현 후 |

### 합의 규칙
- J1·J2 동의 시 판정 확정
- 불일치 시 J3 결정적
- J4 는 독립 평가 (비교용)

### 구현 계획 (`llm-judge.mjs` 확장)
```bash
node llm-judge.mjs --judge claude-opus-4-7 --samples bench/v1.json
node llm-judge.mjs --judge gpt-5 --samples bench/v1.json
node llm-judge.mjs --judge hf-detector --samples bench/v1.json
# 합의 계산
node llm-judge.mjs --consensus out/judge-*.json
```

---

## 3. Adversarial 탐지기 평가

### 2025 SOTA 탐지기
| 모델 | 배포 | 한국어 지원 | 접근 |
|---|---|---|---|
| Hello-SimpleAI/chatgpt-detector-roberta | HF | 제한적 | Inference API |
| KatFishNet | 논문(ACL 2025) | 최적화 | 재현 필요 |
| XDAC (arXiv 2024.11) | GitHub | 한국어 | 배포 |
| HRET benchmark | 기관 | 한국어 | 평가만 |

### 프로토콜
1. **Batch**: 검증셋 50 샘플 × {AI 생성, HUMAN 원본}
2. **Run**: 각 탐지기를 blind하게 호출
3. **Metric**: Accuracy, F1, AUC
4. **Gate**: 모든 탐지기 Accuracy ≤ 60% (random baseline 대비 10%p 이내)

### 일정
- 2026-04-28: HF chatgpt-detector-roberta 배치 측정
- 2026-05-05: XDAC 배포본 적용 시도
- 2026-05-12: KatFishNet 재현 (코드 공개 시)

---

## 4. False Positive Baseline ⚠️ 미측정

### 가설
실제 crawl 원본을 AI로 오판하는 비율이 ~40% (= 우리 생성물 AI 판정률 40%)이면 **실질적 구분 불가**.

### 프로토콜
```
Set A: 크롤 원본 25건 (stratified)
Set B: AI 생성 25건 (stratified)
Judge: J1, J2, J3 독립
Blind: 출처 숨김
Metric: HUMAN→AI 오판 (FP), AI→HUMAN 오판 (FN)
Target: |FP - FN| < 10%p
```

### 현재 데이터 (2026-04-21 블라인드)
- n=10 (5+5)
- AI→HUMAN 오판: **80%** (4/5) — 구분 불가 근접
- HUMAN→HUMAN 정답: **100%** (5/5) — ??? (FP=0%)
- → FP=0% vs FN=80% 비대칭 → **Judge 가 AI 판정을 기피하는 편향** 확인됨

### 조치
- J1 (Claude) 재판정 시 "판정 비율 강제 균등화" 프롬프트 실험
- J2 (GPT-5) 교차 확인으로 편향 완화

---

## 5. Scale-up 블라인드 (n=50, n=100)

### 현재
- n=10: 정확도 60% (random 50%)
- 95% 신뢰구간: [31%, 83%] — **너무 넓음**

### 확장 계획
| 단계 | n | 95% CI 폭 | 일정 |
|---|---|---|---|
| v1 | 10 | ±26%p | ✅ 2026-04-21 |
| v2 | 50 | ±14%p | 2026-04-28 |
| v3 | 100 | ±10%p | 2026-05-05 |
| v4 | 500 | ±4%p | 2026-05-19 (최종) |

### 샘플 구성 (stratified)
- 주제 8개 × 길이 3종 × (AI/HUMAN) = **48 cells**
- cell 당 최소 1건 → n=48 가 실질 최소
- 권장: cell 당 2건 = n=96

### 권한 구조
- 실험 조수(blind rater): 대조셋 미접근
- 원본-AI 매핑 테이블: 별도 암호 파일
- 판정 종료 후 매핑 공개

---

## 6. Benchmark 데이터셋 공식화

### 현재
- `.omc/autopilot/bench/` — 비구조화

### 목표
```
bench/
  v1-20260421/
    samples.jsonl           # {id, text, source: "ai"|"human", meta}
    ground-truth.csv        # id, source, topic, length
    README.md               # 생성 조건, seed
  v2-20260428/              # 50 샘플 확장
```

---

## 7. Continuous evaluation (회귀 방어)

### 기존 게이트 (`run-indist-regression.mjs`)
- threshold 0.70
- max drop 0.05

### 확장
| 게이트 | 기준 | 실패 시 |
|---|---|---|
| 통계 스코어 | overall ≥ 0.70 | ❌ block |
| 이전 대비 drift | |Δ| < 0.05 | ⚠️ warn |
| J1 블라인드 (n≥10) | 정확도 ≤ 60% | ⚠️ warn |
| J3 탐지기 | Accuracy ≤ 60% | ❌ block (배포 시) |
| FP 편향 | |FP-FN| < 15%p | ⚠️ warn |

---

## 8. 재현성 (Reproducibility 교차)

모든 평가는 다음 조건 기록:
- seed, Python/Node 버전
- 사용 모델 버전 (`claude-opus-4-7-20260315` 등)
- 프롬프트 semver
- 결과 JSON commit hash

→ [[_system/REPRODUCIBILITY]]

---

## 🔗 관련

- [[_system/GAP-ANALYSIS-2026-04-21]]
- [[_system/REPRODUCIBILITY]]
- [[연구/실험/humanize-튜닝/2026-04-21-blind-test]]
- [[구현/하네스-엔지니어링]]
