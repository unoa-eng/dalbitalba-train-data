# 한국어 커뮤니티 도메인 적응 LLM 파인튜닝 — 문헌 조사

> **조사 일자**: 2026-04-28
> **목적**: Qwen3-8B-Base를 한국 밤문화 커뮤니티 스타일로 적응시키기 위한 학습 설계의 학술적 기반 구축
> **프로젝트 사양**: LoRA CPT → SFT 2-stage, RunPod L40S, $30 예산, 코퍼스 ~67K rows

---

## 1. Domain-Adaptive Pretraining (DAPT) / Continued Pretraining

### 1.1 핵심 논문

| 논문 | 저자/출처 | 핵심 발견 |
|------|-----------|-----------|
| **Don't Stop Pretraining** (ACL 2020) | Gururangan et al. | DAPT가 도메인 과제에서 일관적 성능 향상; TAPT(Task-Adaptive)와 결합 시 최대 효과 |
| **LoRA Learns Less and Forgets Less** (TMLR 2024) | Biderman et al. | LoRA는 Full FT 대비 학습량은 적지만 망각도 적음. CPT에서는 Full FT와의 격차가 높은 rank에서도 좁혀지지 않음 |
| **LoRA vs Full Fine-tuning: An Illusion of Equivalence** (2024) | Kotha et al. | LoRA와 Full FT는 표면적으로 유사한 성능이지만, weight space에서 근본적으로 다른 해를 찾음. LoRA는 저차원 부분공간에서의 적응으로 일반 지식 보존에 유리 |
| **Synthetic Continued Pretraining** (ICLR 2025 Oral) | Gao et al. | 소규모 코퍼스를 EntiGraph로 합성 확장 후 CPT하면 QA 정확도가 log-linear로 향상. 단순 패러프레이즈보다 entity 간 관계 재구성이 핵심 |
| **D-CPT Law** (2024) | Zhang et al. | 도메인 특화 CPT의 스케일링 법칙 제안: 학습 스텝, 모델 크기, 데이터 혼합 비율의 최적 조합을 소규모 실험으로 예측 가능 |
| **Efficient Domain-adaptive CPT for German** (2025) | arXiv 2504.19856 | ICL-Augmented Pretraining(ICL-APT): kNN으로 도메인 관련 텍스트를 증강하여 기존 DAPT 대비 IR 메트릭 3.5% 향상, GPU 시간 4배 절감 |
| **Data Mixing Agent** (2025) | arXiv 2507.15640 | 강화학습으로 도메인별 데이터 혼합 비율을 동적으로 학습하는 최초의 모델 기반 프레임워크 |

### 1.2 소규모 코퍼스 CPT의 효과

- **한계**: 소규모 코퍼스(~50-100K rows)는 사실 지식(factual knowledge)을 한두 번만 노출하므로 모델이 사실을 내재화하기 어려움
- **해결책 1 — Synthetic CPT**: EntiGraph 방식으로 entity 추출 → entity 간 관계를 다양한 형태로 재합성 → 합성 토큰 수에 대해 log-linear 성능 향상 (ICLR 2025)
- **해결책 2 — Data Replay**: 일반 코퍼스의 10-30%를 도메인 데이터와 섞어 학습하면 기존 능력 유지 + 도메인 적응 동시 달성
- **해결책 3 — ICL-APT**: kNN으로 유사 도메인 텍스트를 자동 검색하여 타겟 데이터 증강

### 1.3 LoRA vs Full FT vs QLoRA (소규모 데이터)

| 방법 | CPT 적합성 | SFT 적합성 | 메모리 | 특징 |
|------|-----------|-----------|--------|------|
| **Full FT** | 최적 | 우수 | ~120GB (8B) | CPT에서 가장 높은 도메인 적응력, 하지만 메모리 제약 |
| **LoRA (r=64+)** | 차선 (격차 존재) | 최적 | ~20GB | CPT에서 Full FT 대비 열세이나, SFT에서는 동등. 망각 최소화 |
| **QLoRA (4-bit)** | 가능 | 우수 | ~8-10GB | 메모리 75-80% 절감. 8B 모델을 L40S에서 여유있게 학습 |

**프로젝트 시사점**:
- 우리 코퍼스(~67K rows)는 **스타일 적응**이 목표이지 사실 지식 주입이 아님 → LoRA CPT로 충분
- QLoRA 4-bit로 CPT하면 L40S 48GB에서 batch size를 크게 잡을 수 있어 학습 속도 향상
- **권장**: QLoRA CPT (r=64, alpha=64) → LoRA SFT (r=32, alpha=32)로 2-stage 진행

### 1.4 Catastrophic Forgetting 방지 전략

| 전략 | 방법 | 효과 | 출처 |
|------|------|------|------|
| **Data Replay** | 일반 코퍼스 10-30%를 배치에 혼합 | 가장 안정적. 소스 도메인 성능 유지 | 다수 연구 합의 |
| **L2-LoRA** | 레이어별 차등 L2 정규화 (하위 레이어 강하게) | 사전학습 지식 보존 + 상위 레이어 적응 허용 | arXiv 2501.13669 |
| **LoRA 자체의 정규화 효과** | 저차원 부분공간 제약 | LoRA가 내재적 정규화기 역할 — 과적합/망각 동시 방지 | Biderman et al. 2024 |
| **Orthogonal LoRA (LB-CL)** | Gradient projection으로 직교성 유지 | 이전 지식과 새 지식의 간섭 최소화 | NeurIPS 2024 |
| **KL-Divergence 증류** | 배치별/스테이지별 KL 손실 추가 | 원본 모델의 출력 분포를 기준으로 드리프트 제한 | 다수 연구 |

**프로젝트 시사점**:
- LoRA 자체가 정규화 역할을 하므로, 소규모 CPT에서는 **LoRA + 소량 일반 데이터 replay (10-15%)**만으로 충분
- 영어/중국어 능력은 우리 목표에 불필요하므로, 망각에 대한 우려 수준은 낮음
- 한국어 일반 능력(문법, 논리)이 손상되지 않도록 KoBEST 등으로 모니터링 권장

### 1.5 최적 하이퍼파라미터 설정

| 파라미터 | 권장 범위 | 근거 |
|----------|----------|------|
| **Learning Rate** | 1e-4 ~ 5e-4 (LoRA), 1e-5 (embed/lm_head) | Unsloth 권장: embed/lm_head는 별도로 낮은 LR 사용 |
| **LoRA Rank (CPT)** | 64-128 | CPT에서는 높은 rank가 유리; r=64 이상에서 도메인 적응력 확보 |
| **LoRA Alpha** | rank와 동일 (alpha=rank) | alpha/rank=1 또는 2; 높은 rank에서는 rsLoRA (alpha/sqrt(rank)) 권장 |
| **Target Modules** | q,k,v,o,gate,up,down_proj | MLP+Attention 모두 타겟팅하면 정확도 향상 (Unsloth 권장) |
| **Warmup** | 5-10% of total steps | 안정적 수렴을 위한 표준 설정 |
| **Epochs (CPT)** | 1-3 | 소규모 데이터에서 2-3 epoch까지 유효; 과적합 모니터링 필수 |
| **Epochs (SFT)** | 2-5 | 데이터 품질이 높으면 3 epoch 내로 수렴 |

---

## 2. Style Transfer / Sociolect Adaptation

### 2.1 핵심 논문

| 논문 | 저자/출처 | 핵심 발견 |
|------|-----------|-----------|
| **Toward Informal Language Processing** (NAACL 2024) | Sun et al. | LLM의 슬랭 이해/생성 능력을 체계적으로 벤치마킹. 모델 크기 증가와 슬랭 능력은 비례하지 않으며, 파인튜닝이 필수 |
| **Understanding Slang with LLMs** (EMNLP 2024) | ACL Anthology | 문화 간 슬랭 이해의 차이를 모델링. 지역/커뮤니티별 변이 포착에는 도메인 특화 데이터가 핵심 |
| **How do Language Models Generate Slang** (2025) | arXiv 2509.15518 | LLM의 슬랭 생성 메커니즘을 체계적으로 비교. 생성 품질은 학습 데이터의 슬랭 밀도에 크게 의존 |
| **The Sociolinguistic Foundations of Language Modeling** (Frontiers AI 2024) | 사회언어학적 관점 | 언어 모델은 본질적으로 언어 변이(variety)를 모델링함; 이 통찰이 개발/배포 전략에 반영되어야 함 |

### 2.2 커뮤니티 특화 말투 학습 방법론

**핵심 원칙**: 스타일/말투 적응은 사실 지식 학습과 달리, **분포적 특성**(토큰 빈도, n-gram 패턴, 종결어미 분포)의 학습에 가까움

1. **CPT로 분포 학습**: 커뮤니티 원문을 그대로 CPT에 투입하면 모델이 해당 커뮤니티의 토큰 분포를 흡수
2. **SFT로 조건부 생성 학습**: "이 맥락에서 이런 스타일로 답변하라"를 instruction으로 학습
3. **스타일 토큰 보존**: 전처리 시 ㅋㅋ, ㅠㅠ, ㅈㄴ, ㅅㅂ 등 비표준 토큰을 제거하지 않고 보존해야 함

### 2.3 비표준어/은어 처리 전략

우리 코퍼스의 특징적 표현들:

| 유형 | 예시 | 처리 방법 |
|------|------|-----------|
| **초성 축약** | ㅈㄴ(존나), ㄹㅇ(리얼), ㅅㅂ(시발) | Qwen tokenizer가 자모를 개별 토큰으로 처리하는지 확인 필요 |
| **이모티콘 종결** | ㅋㅋ, ㅠㅠ, ㅜㅜ | 문장 종결 패턴으로 학습 — 제거 금지 |
| **구어체 축약** | ~임, ~함, ~음 | 명사형/서술형 반말 종결 — 빈도 높으므로 자연 학습 |
| **해요체** | ~요 (18.9%) | 지배적 register — 모델이 기본값으로 학습해야 함 |
| **업계 은어** | TC, 밀빵, 케어 등 | CPT에서 빈번 노출로 토큰 임베딩 조정 |

**프로젝트 시사점**:
- Qwen3 tokenizer가 한국어 자모(ㅋ, ㅠ, ㅈ 등)를 어떻게 처리하는지 사전 검증 필수
- 초성체(ㅈㄴ, ㅅㅂ 등)가 의미 있는 토큰 시퀀스로 분리되는지 확인
- CPT 코퍼스에서 이런 표현의 빈도가 충분해야 모델이 학습 가능 (현재 ㅈㄴ 7.6%, ㅅㅂ 3.6%로 충분)
- **데이터 클리닝 시 비표준어를 제거하지 말 것** — 이것이 우리의 핵심 학습 대상

---

## 3. Korean NLP 특화

### 3.1 핵심 논문

| 논문 | 저자/출처 | 핵심 발견 |
|------|-----------|-----------|
| **RedWhale: An Adapted Korean LLM** (2024) | Vo & Jung | 한국어 특화 토크나이저 + 3단계 CPT 전략으로 KoBEST 벤치마크 최고 성능. 9.7B 토큰에도 수렴 미도달 → 추가 학습 여지 존재 |
| **Making Qwen3 Think in Korean with RL** (2025) | Dnotitia Inc. | Qwen3-14B를 SFT + GRPO RL로 한국어 사고 체계 적응. AIME에서 +6.6% 향상. 한국어 chain-of-thought 완전 구현 |
| **KIT-19** (2024) | arXiv 2403.16444 | 19개 한국어 NLP 태스크를 위한 instruction 데이터셋. 각 5,000 예제 — 한국어 SFT의 기준선 |
| **Code-Switching Curriculum Learning** (ACL 2025) | Findings | 다국어 모델의 교차언어 적응에 코드 스위칭 커리큘럼이 효과적 |
| **Optimizing Korean-Centric LLMs via Token Pruning** (2026) | arXiv 2604.16235 | 한국어 중심 LLM에서 불필요 토큰 프루닝으로 효율 향상 |

### 3.2 Qwen 모델의 한국어 성능

- **Qwen2.5**: 29개 이상 언어 지원, 한국어 포함. 다국어 기반 모델 중 한국어 성능 상위권
- **Qwen3**: 사고 모드(thinking mode) 지원. 한국어 내부 추론(inner monologue)이 가능하지만, 기본적으로 영어로 사고하는 경향
- **Qwen3-8B-Base**: Instruct 버전이 아닌 Base 모델이므로, CPT + SFT로 완전한 도메인 적응 가능
- **주의점**: Qwen의 한국어 토크나이저는 중국어/영어 최적화 → 한국어 토큰 효율성이 한국어 특화 모델보다 낮을 수 있음

### 3.3 한국어 초성/은어 처리

- **초성체(초성체)**: 한국어 음절의 초성만으로 의미를 전달 (ㅎㄱ = 한글, ㅈㄴ = 존나)
- **검색/자동완성에서 널리 사용**: Choseong extraction은 한국어 NLP 파이프라인의 표준 기능
- **LLM에서의 처리**: 대부분의 다국어 LLM은 초성체를 개별 자모 토큰으로 처리하나, 의미론적 이해는 학습 데이터에 의존
- **권장**: CPT에서 초성체가 포함된 원문을 충분히 노출시키면, 모델이 문맥적 의미를 학습할 수 있음

**프로젝트 시사점**:
- Qwen3-8B-Base는 한국어 토크나이저 효율이 RedWhale보다 낮지만, CPT 목적으로는 충분
- RedWhale의 3단계 전략(토크나이저 적응 → 임베딩 학습 → 전체 CPT)을 참고하되, 예산 제약상 LoRA CPT 1단계로 축소
- Making Qwen3 Think in Korean 논문의 SFT + RL 2단계 전략은 우리 SFT 설계에 직접 참고 가능

---

## 4. SFT Design for Dialogue/Comment Generation

### 4.1 핵심 논문

| 논문 | 저자/출처 | 핵심 발견 |
|------|-----------|-----------|
| **Fine-Tuning LLMs for Multi-Turn Dialogues** (ICML 2024) | ACM DL | Cross-entropy + KL divergence 최적화로 멀티턴 대화 맥락 이해 향상. 전체 대화를 입력으로 모든 라운드의 응답에 loss 계산 |
| **Review-Instruct** (2025) | arXiv 2505.11010 | Ask-Respond-Review 3역할 에이전트로 멀티턴 대화 합성. LLaMA2-13B에서 MT-Bench 유의미 향상 |
| **The Best Instruction-Tuning Data are Those That Fit** (ICLR 2025) | OpenReview | 타겟 모델의 분포에 맞는 데이터만 선별하면 소량으로도 높은 효과. 분포 외 데이터는 성능 저하 유발 |
| **Massive SFT Experiments** (EMNLP 2025) | ACL Anthology | 대규모 SFT 실험에서 데이터 품질/다양성/모델 정합성의 상호작용을 체계적으로 분석 |
| **Beyond Single-Turn Survey** (2025) | arXiv 2504.04717 | 멀티턴 LLM 상호작용의 포괄적 서베이. 코드 생성, 수학 추론, 의료 대화 등 다양한 응용 |

### 4.2 Thread-Aware Comment Generation 설계

우리 커뮤니티는 **게시글 → 댓글 → 답글** 구조이므로, 다음과 같은 계층적 컨텍스트 설계가 필요:

```
[System] 밤문화 커뮤니티의 회원으로서 자연스러운 댓글을 작성하세요.

[Context: Post]
제목: {title}
내용: {content}
작성자: {author_type}  # 선택적

[Context: Parent Comment]  # 답글인 경우에만
{parent_comment}

[Instruction]
위 게시글에 대해 커뮤니티 스타일로 댓글을 작성하세요.

[Response]
{target_comment}
```

### 4.3 Instruction Design이 생성 품질에 미치는 영향

**핵심 발견** (ICLR 2025):
- SFT 데이터의 response가 타겟 모델의 기존 분포와 맞을수록 효과가 큼
- **시사점**: CPT 이후 모델이 이미 커뮤니티 스타일을 습득한 상태에서 SFT를 수행하면, instruction-response 정합성이 자연스럽게 높아짐 → **CPT-first 전략의 정당성**

**Instruction 다양화 전략**:
- 2-4K개의 고품질 Q&A만으로도 소형 모델이 대형 모델과 동등한 instruction following 달성 가능
- 우리 데이터(44K SFT 쌍)는 양적으로 충분 → **품질 필터링에 집중**

### 4.4 Context Window 활용 전략

| 전략 | 설명 | 적용 |
|------|------|------|
| **Flat Context** | 게시글 + 댓글을 단순 연결 | 간단하지만 구조 정보 손실 |
| **Structured Context** | 게시글, 부모 댓글, 형제 댓글을 구조적으로 분리 | 우리 설계에 적합 |
| **Dynamic Truncation** | 게시글 길이에 따라 댓글 수 조절 | 메모리 효율화 |
| **Thread-Ordered** | 시간순으로 이전 댓글들을 포함 | 대화 흐름 학습에 유리 |

**프로젝트 시사점**:
- 게시글 중간값 40자, 80%가 100자 이하 → context window 여유가 충분
- 댓글 트리 전체를 포함할 수 있으므로, **Structured Context**로 부모 댓글 + 게시글을 모두 포함
- 답글(13,995건)은 부모 댓글과 함께 구성하여 대화 맥락 학습

---

## 5. Evaluation Framework

### 5.1 핵심 메트릭 및 논문

| 메트릭 | 유형 | 설명 | 논문/출처 |
|--------|------|------|-----------|
| **Perplexity** | 자동 | 모델이 도메인 텍스트를 얼마나 잘 예측하는지. 낮을수록 좋음 | 표준 메트릭 |
| **MAUVE Score** | 자동 | 생성 텍스트와 인간 텍스트의 분포적 유사도. KL divergence 기반 | Pillutla et al. (JMLR 2024) |
| **n-gram 일치율** | 자동 | 커뮤니티 특유 n-gram(ㅋㅋ, ~요, ㅈㄴ 등)의 빈도 분포 비교 | 커스텀 메트릭 |
| **X-Turn Pass-Rate** | 인간 | 대화 턴 수별 인간 판별 통과율 | X-TURING (arXiv 2408.09853) |
| **Turing Test** | 인간 | 인간과 AI 구분 실험. GPT-4.5가 73% 통과 | Jones & Bergen (2025) |
| **Style Consistency** | 자동+인간 | 종결어미 분포, 이모티콘 사용률, 문장 길이 분포의 원본 대비 유사도 | 커스텀 메트릭 |

### 5.2 도메인 특화 평가 체계 설계

#### 5.2.1 자동 평가 (Automatic)

1. **Domain Perplexity**: val_set.v2 (2,451건)에 대한 perplexity 측정
   - CPT 전 vs CPT 후 비교
   - SFT 전 vs SFT 후 비교
   - 목표: CPT 후 20-40% perplexity 감소

2. **MAUVE Score**: 생성 댓글 500개 vs 실제 댓글 500개의 분포 비교
   - 목표: MAUVE > 0.85 (인간-인간 수준 근접)

3. **Style Fingerprint 일치율**:
   - 종결어미 분포 (해요체 비율, 반말 비율)
   - 비언어적 표현 빈도 (ㅋㅋ, ㅠㅠ, ㅈㄴ)
   - 문장 길이 분포
   - 원본 분포와의 KL divergence or JS divergence

4. **어휘 커버리지**: 커뮤니티 은어/초성체 사용률

#### 5.2.2 인간 평가 (Human Evaluation)

1. **Blind Turing Test**: 실제 댓글과 생성 댓글을 섞어 판별 실험
   - 5-point 척도: 확실히 AI / 아마 AI / 모르겠음 / 아마 인간 / 확실히 인간
   - 목표: "확실히 AI" 비율 < 20%

2. **Style Appropriateness**: 커뮤니티 회원이 "이 댓글이 우리 커뮤니티에 어울리는가" 평가
   - 1-5 Likert 척도
   - 목표: 평균 4.0 이상

3. **Context Relevance**: 게시글에 대한 댓글의 맥락 적절성

### 5.3 평가 타이밍

| 단계 | 평가 항목 | 기준 |
|------|----------|------|
| CPT 후 | Domain Perplexity, 일반 능력(KoBEST subset) | PPL 감소 + 일반 능력 유지 |
| SFT 후 | MAUVE, Style Fingerprint, 생성 샘플 정성 검토 | MAUVE > 0.85 |
| 최종 | Blind Turing Test, Style Appropriateness | 인간 판별 정확도 < 60% |

---

## 6. Small Budget Training Optimization

### 6.1 L40S (48GB) 최적 활용 전략

| 설정 | 값 | 근거 |
|------|------|------|
| **Quantization** | QLoRA 4-bit (NF4) | 메모리 75-80% 절감, 8B 모델을 ~10GB로 로드 |
| **Batch Size** | micro_batch=4, grad_accum=8 → effective=32 | L40S 48GB에서 QLoRA 시 충분한 여유 |
| **Sequence Length (CPT)** | 2048 | 게시글/댓글이 짧으므로 2048이면 충분 |
| **Sequence Length (SFT)** | 1024-2048 | Context(게시글+부모댓글) + Response 고려 |
| **Mixed Precision** | bf16 | L40S가 bf16 지원; fp16보다 안정적 |
| **Gradient Checkpointing** | 활성화 | 메모리 절약의 핵심. 약 30% 메모리 절감 |
| **FlashAttention-2** | 활성화 | Attention 연산 2-4x 가속 |
| **Optimizer** | AdamW 8-bit (bitsandbytes) | Optimizer state 메모리 50% 절감 |

### 6.2 Unsloth 활용

- **속도**: Hugging Face transformers 대비 2-2.5x 빠른 학습
- **메모리**: 60% 적은 VRAM 사용
- **CPT 지원**: `UnslothTrainer` + `UnslothTrainingArguments`로 embedding_learning_rate 별도 설정
- **Qwen3 지원**: Unsloth가 Qwen 계열을 공식 지원

### 6.3 $30 예산 분배 계획

RunPod L40S 시간당 약 $0.69 기준 (community cloud):

| 단계 | 예상 시간 | 비용 | 설명 |
|------|----------|------|------|
| **환경 설정** | 0.5h | $0.35 | 패키지 설치, 데이터 업로드 |
| **CPT** | 8-12h | $5.5-8.3 | ~47K rows × 2-3 epochs |
| **SFT** | 6-10h | $4.1-6.9 | ~44K pairs × 3-5 epochs |
| **평가/생성** | 2-3h | $1.4-2.1 | 샘플 생성, MAUVE 계산 |
| **버퍼** | - | $5-10 | 재학습/디버깅 여유 |
| **합계** | 17-26h | ~$12-18 | 예산 내 2회 시도 가능 |

### 6.4 효율 극대화 전략

1. **Mac Mini에서 0.6B로 사전 검증**: 하이퍼파라미터 탐색을 로컬에서 수행 → RunPod은 최종 학습만
2. **Packing**: 짧은 텍스트를 하나의 시퀀스로 패킹하여 padding 낭비 제거 (Unsloth 지원)
3. **Early Stopping**: val_set 기반 perplexity 모니터링으로 불필요한 epoch 방지
4. **체크포인트 저장 전략**: 매 epoch + best model 저장 → 최적 지점에서 SFT 시작

---

## 7. 종합 시사점 및 권장 학습 파이프라인

### 7.1 학술적 근거에 기반한 설계 원칙

1. **CPT-first는 학술적으로 정당**: DAPT 연구(Gururangan 2020) + Synthetic CPT(ICLR 2025)가 도메인 적응에 CPT의 효과를 입증
2. **LoRA CPT의 한계는 인지하되 실용적 선택**: LoRA가 Full FT보다 CPT에서 열세이나(Biderman 2024), 우리 목표는 사실 지식이 아닌 스타일 적응 → LoRA로 충분
3. **SFT 데이터 품질 > 양**: ICLR 2025 연구에서 2-4K 고품질 데이터로도 충분한 효과 입증
4. **MAUVE가 스타일 평가의 최적 자동 메트릭**: 분포적 유사도 측정으로 스타일 일치도 정량화 가능
5. **Synthetic CPT는 미래 개선 옵션**: 현재 코퍼스가 부족할 경우 EntiGraph 방식의 합성 확장 고려

### 7.2 권장 파이프라인

```
[Phase 0] 로컬 검증 (Mac Mini, Qwen3-0.6B)
  - 토크나이저 자모 처리 확인
  - 하이퍼파라미터 소규모 탐색
  - SFT instruction 템플릿 검증

[Phase 1] CPT (RunPod L40S)
  - 데이터: cpt_corpus.v2 (46,856 rows) + 일반 한국어 10-15% replay
  - 방법: QLoRA 4-bit, r=64, alpha=64
  - Epochs: 2-3, LR=2e-4, embed_LR=1e-5
  - 평가: val_set perplexity 모니터링

[Phase 2] SFT (RunPod L40S)
  - 데이터: sft_pairs.v2 (44,474 pairs)
  - 방법: LoRA (CPT adapter 위에), r=32, alpha=32
  - Epochs: 3, LR=1e-4
  - Instruction: Structured Context (게시글 + 부모댓글 + 지시문)

[Phase 3] 평가
  - 자동: Domain PPL, MAUVE, Style Fingerprint
  - 인간: Blind Turing Test (소규모)
```

---

## Sources

### Domain-Adaptive Pretraining
- [Don't Stop Pretraining (ACL 2020)](https://www.researchgate.net/publication/343303103_Don't_Stop_Pretraining_Adapt_Language_Models_to_Domains_and_Tasks)
- [Efficient Domain-adaptive CPT for German (2025)](https://arxiv.org/abs/2504.19856)
- [Synthetic Continued Pretraining (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)
- [D-CPT Law (2024)](https://arxiv.org/html/2406.01375v1)
- [Data Mixing Agent (2025)](https://arxiv.org/abs/2507.15640)
- [Domain Adaptation for e-Commerce (ACL 2025)](https://aclanthology.org/2025.acl-industry.74.pdf)

### LoRA / Catastrophic Forgetting
- [LoRA Learns Less and Forgets Less (TMLR 2024)](https://openreview.net/pdf/ab275bf1fed6b1e3642e0eb167b4b7cbd7810a94.pdf)
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence (2024)](https://arxiv.org/html/2410.21228v3)
- [How Much is Too Much? LoRA Rank Trade-offs (2024)](https://arxiv.org/html/2512.15634v1)
- [Hierarchical Regularization for Catastrophic Forgetting (2025)](https://arxiv.org/html/2501.13669v2)
- [Learning Rate Scaling across LoRA Ranks (2026)](https://arxiv.org/abs/2602.06204)
- [Learning Rate Matters: Vanilla LoRA May Suffice (2026)](https://arxiv.org/pdf/2602.04998)

### Style Transfer / Slang
- [Toward Informal Language Processing (NAACL 2024)](https://aclanthology.org/2024.naacl-long.94/)
- [Understanding Slang with LLMs (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.869.pdf)
- [How do Language Models Generate Slang (2025)](https://www.arxiv.org/pdf/2509.15518)
- [Sociolinguistic Foundations of Language Modeling (Frontiers AI 2024)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1472411/full)

### Korean NLP
- [RedWhale: Korean LLM via Efficient CPT (2024)](https://arxiv.org/abs/2408.11294)
- [Making Qwen3 Think in Korean with RL (2025)](https://arxiv.org/abs/2508.10355)
- [KIT-19: Korean Instruction Toolkit (2024)](https://arxiv.org/html/2403.16444v1)
- [Optimizing Korean-Centric LLMs via Token Pruning (2026)](https://arxiv.org/html/2604.16235)

### SFT / Multi-turn Dialogue
- [Fine-Tuning LLMs for Multi-Turn Dialogues (ICML 2024)](https://dl.acm.org/doi/10.1145/3651671.3651702)
- [Review-Instruct: Multi-Turn Conversations (2025)](https://arxiv.org/html/2505.11010v1)
- [The Best Instruction-Tuning Data are Those That Fit (ICLR 2025)](https://openreview.net/forum?id=4jFSekBaDT)
- [Beyond Single-Turn Survey (2025)](https://arxiv.org/html/2504.04717v1)
- [Fine-tuning LLMs for Domain Adaptation (Nature 2025)](https://www.nature.com/articles/s41524-025-01564-y)

### Evaluation
- [MAUVE Score (JMLR 2024)](https://www.jmlr.org/papers/volume24/23-0023/23-0023.pdf)
- [LLMs Pass the Turing Test (2025)](https://arxiv.org/abs/2503.23674)
- [X-TURING: Enhanced Turing Test (2024)](https://arxiv.org/abs/2408.09853)
- [LLM Evaluation in 2025](https://www.techrxiv.org/users/927947/articles/1304989)

### Training Optimization
- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Unsloth Continued Pretraining Guide](https://unsloth.ai/docs/basics/continued-pretraining)
- [RunPod Fine-Tuning Guide](https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget)
- [Fine-Tuning Infrastructure: LoRA, QLoRA at Scale (2025)](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)
- [How to Fine-tune LLMs in 2025 (Philschmid)](https://www.philschmid.de/fine-tune-llms-in-2025)
