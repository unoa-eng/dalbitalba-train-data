# Round-N Evidence-Based Audit (2026-05-04)

## 목적
LITERATURE_REVIEW_2025.md가 권장하는 evidence-based 기법 중
현 레시피에 미적용된 것을 식별, 다음 라운드 후보로 정리.

## 검토 대상 (lit review 인용 + 미적용 가능성 의심)

### 1. Vocab Expansion (도메인 어휘 토큰화)

**Lit review 근거**: RedWhale (2024) — 한국어 특화 토크나이저 + 3단계 CPT
권장. 현 프로젝트는 예산상 "1단계 LoRA CPT로 축소" 명시.

**현 상태**: `scripts/extend_tokenizer_v3.py`에서 **구조 마커만** 추가.
- `<|post|>`, `<|/post|>`, `<|/comment|>`
- `<|comment depth=0|>` ~ `<|comment depth=5|>`

**누락**: 호빠 도메인 어휘 (밀빵, 쩜오, 티씨, TC, 마담, 초이스, 갯수,
와리, 텐카, 도파민, 호빠, 케어, 빠꾸, 풀묶 등)는 base tokenizer가
BPE로 분해. 새 토큰으로 추가 + 임베딩 학습 단계 부재.

**예상 효과 (D2/Step C 결과 기반 추정)**:
- domain_keyword_alignment 0.20 → 0.45+ (개별 어휘를 토큰 단위로 학습)
- bigram_jsd 일부 closure (도메인 토큰 분포 매칭)
- coherence 향상 (특히 "밀빵 0%" 같은 0% 키워드)

**구현 부담**: 중. extend_tokenizer_v3.py에 도메인 어휘 추가 + 임베딩
warmup 단계 추가. CPT_LORA_EMBED=1 옵션 확장.

---

### 2. Synthetic CPT (EntiGraph)

**Lit review 근거**: Gao et al. ICLR 2025 Oral. 소규모 코퍼스(~50K)는
사실 지식을 한두 번만 노출 → 모델 내재화 실패. EntiGraph로 entity
관계를 다양한 형태로 재합성하면 합성 토큰 수에 대해 log-linear
성능 향상.

**현 상태**: 완전 미적용. `scripts/*synth*` 검색 0건. 합성 CPT 코드 없음.

**기법 개요**:
1. 호빠 도메인 entity 추출 (가게명/직책/용어 등 — 이미 부분적으로 obsidian
   persona에 있음)
2. Entity 쌍 간 관계를 LLM(Claude/GPT-4)으로 재서술 (수만 가지 표현)
3. 합성된 텍스트를 CPT 데이터에 추가
4. 결과: 67K → 670K 토큰 (10×) 또는 그 이상

**예상 효과**:
- bigram_jsd: 5.6× 격차의 절반 이상 closure 가능 (분포 다양성 증가)
- domain_keyword_alignment: 0.20 → 0.40+ (entity 노출 빈도 증가)
- length_kl: 호빠 톤의 길이 분포가 더 robust해짐

**구현 부담**: 중-대. Claude API 호출 ~$50-100, 파이프라인 작성 1-2주.

---

### 3. D-CPT Law 기반 Scaling 실험

**Lit review 근거**: Zhang et al. 2024. 도메인 CPT의 epoch / 모델 크기 /
데이터 혼합 비율 최적 조합을 소규모 실험으로 예측.

**현 상태**: 미적용. 현재 CPT_NUM_EPOCHS=1은 직관/예산 기반 결정.
D-CPT Law 적용 시 더 적은 비용으로 최적 epoch 도달 가능성.

**예상 효과**: 추측 단계.

---

### 4. ICL-APT (kNN 기반 데이터 증강)

**Lit review 근거**: arXiv 2504.19856. kNN으로 유사 도메인 텍스트 자동
검색 → 데이터 증강. 기존 DAPT 대비 IR +3.5%, GPU 시간 4배 절감.

**현 상태**: 미적용.

---

### 5. NEFTune (noise embedding fine-tuning)

**Lit review 근거**: lit review에 미인용. 외부 reference로 0.5-2% 추가
효과 알려짐. TRL 내장.

**현 상태**: 미적용.

**평가**: 작은 효과, lit review 미인용 → 우선순위 낮음.

---

## 권장 우선순위 (Round-N 적용 후보)

| 우선순위 | 기법 | 비용 | ROI | 비고 |
|---|---|---|---|---|
| **1** | Vocab Expansion (도메인 어휘 토큰 추가 + 임베딩 학습) | 중 (코드 ~1-2일) | **높음** | 0% 도메인 어휘 직접 해결 |
| **2** | Synthetic CPT (EntiGraph) | 대 ($50-100 + 1-2주) | **높음** | 데이터량 부족 본질 해결 |
| **3** | D-CPT Law scaling | 중 (ablation cycle 2-3회) | 중 | epoch/lr 최적화 |
| **4** | ICL-APT | 중 | 중 | 보조적 |
| **5** | NEFTune | 소 (1줄 추가) | 소 | 작은 무료 효과 |

## Round 2 (현 launcher 패치)와의 직교성

Round 2 = LoRA r=128 + DoRA + ORPO + persona-weighted SFT.
이는 위 1~5와 직교적 — 동시 적용 가능.

진정한 베테랑급 도달 경로는:
- Round 2 (configuration 변경) +
- Round 3+ (Vocab Expansion + Synthetic CPT — 이 audit의 1+2)

## 본 audit의 한계

- "예상 효과" 수치는 lit review 추정값 + D2/Step C 격차 분석 기반.
  실제 효과는 작은 ablation cycle 1-2회로 확인 권장.
- Round 2 결과를 보지 않은 상태에서 작성 → Round 2 이후 다시 audit 가치.
