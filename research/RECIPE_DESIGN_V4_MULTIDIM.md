# Recipe Design V4 — Multi-Dimensional Domain Mastery

**Date**: 2026-05-04
**Goal**: 단어/주제/말투/대화구조/세계지식/페르소나 **6개 차원 전부**가
veteran-level (호빠 종사자 baseline)에 도달하는 완전한 도메인 적응
모델 학습.

**Predecessor**: TRAINING_DESIGN_V3.md (어휘 위주 + 단일 SFT 가정),
chain_train.sh (Round 1), chain_train_round2.sh (config-only Round 2).

**Why V4**: D1.5/D2/Step C 결과로 6차원 중 1-2개만 부분 학습됨.
veteran 도달은 차원별 직교 기법 + 단계적 누적 필요.

---

## 0. 6 Dimensions of Domain Mastery

### L1 — Lexicon (어휘)
- 정의: 도메인 토큰 (TC/밀빵/쩜오/티씨/마담/초이스/갯수/와리/텐카/
  도파민/호빠/케어/빠꾸/풀묶 + 디시 줄임말 ㅋㅋ/ㅈㄴ/ㅇㅇ)
- D2 격차: domain_keyword_alignment = 0.20 (target ≥0.5)
- 핵심 결함: tokenizer가 BPE 분해 → "밀빵" = "밀"+"빵"으로 처리 중

### L2 — Topics (주제)
- 정의: 출근/조회/팁/초이스/단골/진상/마감/이직/돈/가족/미래
- D2 격차: bigram_jsd = 0.45 (target ≤0.08, 5.6×)
- 핵심 결함: corpus가 67K rows로 작음, 주제 다양성 부족

### L3 — Style/Tone (말투)
- 정의: 짧은 문장, ㅋㅋㅋ/ㅠㅠ 빈출, 줄임말, 비격식 어말, 솔직한 톤
- D2 격차: tone_distribution_match = 0.21 (target ≤0.15)
- 격차 closure: Step C에서 0.10으로 통과 (rep_penalty 효과)

### L4 — Pragmatics (대화구조)
- 정의: thread → post → comment → reply 계층, 공감/조언/장난 패턴
- B0 격차: structure_fidelity_pct = 0.0 (base는 구조 0% 학습)
- D2 측정 미수행 → 추정 partial 학습

### L5 — World/Domain Knowledge (세계지식)
- 정의: 강남/역삼/논현 위치 관계, 호빠/룸살롱/노래방 차이, 직급
  (실장/언니/이모/마담), 시스템 (TC/밀빵/와리/케어)
- D2 격차: 부분 학습 (역삼 over-generated 10×, 호빠 ratio 1.12 정상)
- 핵심 결함: 사실 지식 노출 부족 (소규모 코퍼스 1-2회 노출)

### L6 — Persona (화자 일관성)
- 정의: 한 글쓴이 안에서 언어 스타일/관점 일관성, 다른 글쓴이끼리는 변별
- D2 측정 미수행
- Round 2 recipe에 페르소나 prefix만 부분 적용 (full persona LoRA 아님)

---

## 1. Stage-by-Stage Recipe (8 Stages)

### Stage 0 — Audit & Annotation (foundation)

**목표**: 기존 corpus를 6차원으로 라벨링.

**작업**:
- 모든 글에 topic 태그 (multi-label classifier 또는 manual)
- 모든 글에 tone 라벨 (formal/informal/comic/serious/sad/...)
- 모든 글에 persona 추출 (저자별 누적 글 수, 직급 추정)
- per-dimension 기준점 측정 (raw corpus 자체에서 distributional baseline)

**비용/시간**: $0 + 1주

**산출물**:
- `runs/audit/topic_labels.jsonl`
- `runs/audit/tone_labels.jsonl`
- `runs/audit/persona_index.jsonl`
- `runs/audit/dimension_baseline.json`

---

### Stage 1 — Tokenizer Adaptation (L1 fundamental)

**근거**: RedWhale 2024, lit review 권장. 현재 1단계 축소됨 — 본 stage가
이를 정상화.

**작업**:
1. 도메인 어휘 ~200-500개 식별 (corpus frequency + manual curation)
2. `extend_tokenizer_v3.py` 확장 — 구조 마커 + 도메인 어휘 모두 추가
3. Embedding warmup pass: 새 토큰 임베딩만 학습 (CPT_EMBED_WARMUP_STEPS)
4. tokenizer 효율성 검증 (한국어 BPE byte/token 비율 측정)

**비용/시간**: $5-10 + 1-2주

**메트릭**:
- token efficiency (avg tokens / sentence) 감소
- domain_keyword_alignment 측정 시 명시적 토큰 매치 가능
- B0+L1 ablation: structure_fidelity 0%→? 

---

### Stage 2 — Knowledge Injection (L2 + L5)

**근거**: Synthetic CPT / EntiGraph (ICLR 2025 Oral, Gao et al.),
D-CPT Law (Zhang 2024), ICL-APT (2025).

**3-phase CPT**:

**Phase A: Broad CPT** (general Korean + light domain)
- 기존 cpt_corpus.v3.jsonl + 일반 한국어 replay 30%
- 1 epoch
- 목적: base의 한국어 분포 유지하면서 도메인 토큰 노출

**Phase B: Synthetic CPT** (EntiGraph-augmented)
- 도메인 entity 추출 → Claude API로 entity 관계 재서술
- 67K corpus → 670K (10×) synthetic + 조정
- 2-3 epoch
- 목적: 사실 지식 다양한 표현으로 노출 → log-linear 향상

**Phase C: Clean Domain CPT** (high-quality only)
- 베테랑 글쓴이 ID로 필터링 (글 수 50+)
- 1 epoch
- 목적: 최고 품질 분포로 마무리

**비용/시간**: $50-100 (Claude API 합성) + $20-40 (3-phase CPT) + 2-3주

**메트릭**:
- bigram_jsd 5.6× → 2-3× 격차
- length_kl 11× → 4× 격차
- L5 entity recall 측정 (강남/역삼/직급 정확성)

---

### Stage 3 — Style/Tone Alignment (L3)

**근거**: Style-classifier-weighted loss (custom),
NEFTune (Jain et al. NeurIPS 2024), ORPO (Hong et al. 2024).

**작업**:
1. Tone classifier 훈련 (binary: formal vs informal, multi: tone categories)
2. SFT data를 tone-graded weighting으로 학습:
   - informal/comic/sad는 weight 1.5
   - formal/clinical은 weight 0.5
3. NEFTune 적용 (TRL 1줄 옵션)
4. Tone-mismatched sample은 ORPO rejected pair로 사용

**비용/시간**: $10-20 + 1-2주

**메트릭**:
- tone_distribution_match 0.10 (Step C 수준 유지) + classifier confidence ≥0.85
- length_kl 1× 이내

---

### Stage 4 — Topic Coverage (L2 정밀화)

**근거**: Curriculum learning, topic-balanced sampling (다수 연구).

**작업**:
1. Stage 0 topic 라벨 기반으로 SFT batch sampling (rare topic 가중)
2. Topic curriculum: 빈도 높은 topic 먼저 → 희소 topic 나중
3. 누락 topic은 EntiGraph로 합성 데이터 추가 생성

**비용/시간**: $10 + 1주

**메트릭**:
- per-topic generation rate (희소 topic 0% → ≥0.05)
- topic coverage 20개 topic 모두 ≥0.05

---

### Stage 5 — Pragmatic Alignment (L4)

**근거**: Multi-turn dialogue training, thread-conditioned SFT
(이미 sft_thread_conditioned.jsonl 부분 적용).

**작업**:
1. Thread chain training: post → comment_1 → reply → comment_2 ...
2. Q&A pair 추출 + SFT
3. Reply pattern training (위로/공감/장난/조언/비판 5분류)
4. Multi-turn coherence loss 추가

**비용/시간**: $10-15 + 1주

**메트릭**:
- structure_fidelity_pct ≥80%
- multi-turn coherence (3턴 dialogue test)

---

### Stage 6 — Preference Alignment (cross-dimension)

**근거**: ORPO (Hong 2024), DPO (Rafailov 2023), KTO (Ethayarajh 2024).

**작업**:
1. Preference pairs 생성:
   - chosen: veteran-style (Stage 0 persona index 상위)
   - rejected: base reactivation (학원광고 형식 등)
2. Multi-dimension preference: 각 차원 위반 sample을 rejected로
3. ORPO 1-2 epoch (Round 2 recipe + 확장)

**비용/시간**: $15-25 + 1주

**메트릭**:
- All 7 deterministic metrics PASS
- 인간 평가 panel score (다음 stage)

---

### Stage 7 — Iterative Refinement (mutator + self-reward)

**근거**: chain_train_round2_mutator (existing), Self-Rewarding
Language Models (Yuan et al. 2024).

**작업**:
1. Round-N mutator: 메트릭 fail → 자동 recipe 변형 (existing)
2. Self-reward: 모델이 자기 출력을 평가 → 새 preference data
3. 3-5 cycle iteration

**비용/시간**: cycle당 $5-15 + ongoing

**메트릭**: 안정 수렴 (cycle별 메트릭 변화 <5%)

---

### Stage 8 — Multi-Dimensional Evaluation (gate)

**작업**:
1. **결정론적 게이트** (현 phase6_eval, 7-metric)
2. **차원별 게이트**:
   - L1: domain_keyword_alignment ≥0.5
   - L2: bigram_jsd ≤0.20 (현실적 임계값으로 완화)
   - L3: tone classifier confidence ≥0.85
   - L4: structure_fidelity ≥80%
   - L5: entity recall test (강남/직급 30개 question)
   - L6: persona consistency cross-thread test
3. **인간 평가 panel** (호빠 종사자 또는 원본 글쓴이 5+명):
   - blind A/B (Round 1 vs Round V4 vs raw)
   - "이게 진짜 호빠 글 같나?" 5점 리커트
   - veteran 라벨 (initial/intermediate/veteran) 분류 정확도

**비용/시간**: $5 + human eval (지원금 또는 인터뷰)

**최종 게이트**:
- 7 deterministic metrics 모두 PASS
- 차원별 ≥80% PASS
- 인간 panel "veteran" 라벨 ≥70%

---

## 2. 누적 비용 / 시간 추정

| Stage | 비용 | 시간 |
|---|---|---|
| 0 Audit | $0 | 1주 |
| 1 Tokenizer | $5-10 | 1-2주 |
| 2 CPT | $70-140 | 2-3주 |
| 3 Style | $10-20 | 1-2주 |
| 4 Topic | $10 | 1주 |
| 5 Pragmatic | $10-15 | 1주 |
| 6 Preference | $15-25 | 1주 |
| 7 Iterate (3 cycles) | $15-45 | 2-3주 |
| 8 Eval (deterministic + human) | $5 + human cost | 1주 |
| **합계** | **$140-270 + human eval** | **11-15주** |

---

## 3. Round 2 (현 launcher)와의 매핑

Round 2 = config 변경 (LoRA r=128 + DoRA + ORPO + persona prefix).
이는 **Stage 6의 부분 시행**이지 Stage 1-5는 미반영.

→ Round 2는 V4의 Stage 6 부분 빠른 검증으로 가치 있음.
   Round 2 결과로 ORPO 기법의 효과 측정 → V4 Stage 6 본격 진행 시
   더 정확한 epoch/lr 선택 가능.

---

## 4. Risk / Caveats

- **합성 데이터 품질**: Claude/GPT-4가 호빠 톤을 정확히 모사 못 할 수도.
  → Stage 2 phase B 실험적, 결과 이상 시 폐기
- **Tokenizer 확장의 임베딩 학습 품질**: 새 토큰의 임베딩이 충분히
  학습 안 되면 오히려 악화. → Stage 1 ablation 필수
- **인간 평가 비용/접근성**: 호빠 종사자 panel 모집이 현실적으로
  어려울 수 있음. → 차선: 도메인 전문 reviewer 또는 원본 글쓴이 reach
- **베테랑 정의의 모호함**: 객관적 ground truth 부재. → corpus의 글
  많이 쓴 사람들을 proxy로 사용
- **임계값 재조정**: bigram_jsd ≤0.08은 현실적으로 도달 불가능 가능.
  ≤0.20이 8B fine-tune 도달선. lit review와 sanity check 필요.

---

## 5. V4 vs 기존 설계 비교

| 차원 | V3 (TRAINING_DESIGN_V3) | Round 2 | **V4** |
|---|---|---|---|
| Stage 수 | 2 (CPT + SFT) | 5 (CPT broad/clean, SFT, ORPO, eval) | **8 (multi-dim)** |
| 차원 명시성 | ❌ | 부분 | **6차원 명시** |
| Tokenizer expansion | 부분 (구조 마커만) | 부분 | **Stage 1 정식** |
| Synthetic CPT | ❌ | ❌ | **Stage 2 Phase B** |
| Topic balance | ❌ | ❌ | **Stage 4** |
| Style classifier | ❌ | ❌ | **Stage 3** |
| Multi-turn pragmatics | partial | partial | **Stage 5 정식** |
| Persona LoRA | ❌ | prefix only | **Stage 5/6 발전** |
| Human eval | mention only | ❌ | **Stage 8 의무** |
| Iterative mutator | ❌ | exists | **Stage 7 통합** |

---

## 6. Action Items (다음 세션부터)

### 즉시 가능 (무료)
- Stage 0 audit 실행 (corpus를 6차원으로 라벨링)
- Stage 1 vocab 후보 200개 식별
- 인간 평가 panel 모집 가능성 탐색

### 다음 RunPod 사이클
- Round 2 실행 ($5-7) — V4 Stage 6 부분 검증으로 활용

### 본격 V4 진행 (별도 예산 책정)
- Stage 1+2가 가장 ROI 높음
- 합계 ~$80-150에 11-15주 일정 필요
