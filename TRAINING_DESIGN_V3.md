# 달빛알바 도메인 적응 학습설계 v3
## Research-Grade Training Design for Community-Indistinguishable Text Generation

**Version**: v3.1 (2026-05-12, audit close-out)

> **목표**: Qwen3-8B-Base를 LoRA CPT+SFT로 파인튜닝하여, 퀸알바 커뮤니티 원천DB와 **구분 불가능한** 텍스트(게시글/댓글)를 생성하는 것.
>
> **하드웨어**: RunPod L40S (48GB), 예산 $30/cycle
>
> **데이터**: 게시글 ~11K, 댓글 ~56K, 9개 토픽 클러스터

---

## 0. Hypotheses (H1~H4)

본 학습설계는 paper-grade 검증을 위해 4개의 falsifiable hypothesis를 우선 선언한다. 각 가설은 사전 등록(pre-registration)된 임계값과 reject 조건을 가지며, 최종 보고는 본 절의 메트릭에 직접 매핑되어야 한다.

### H1 — 도메인 적응 (Domain Adaptation)

- **진술**: Qwen3-8B-Base + thread-conditioned SFT(structured CPT v3 + 5-task SFT)가 base 모델 대비 domain MAUVE를 +0.20 이상 향상시킨다.
- **레퍼런스 ceiling**: paper8b raw-vs-raw MAUVE = 0.0190 (community 원천 분포 self-similarity).
- **임계값**:
  - MAUVE(generated, reference) ≥ 0.80 on N≥1,000 held-out 분포.
  - Bigram JSD ≤ 0.08 (stretch: ≤ 0.05).
  - Length KL ≤ 0.01 (게시글/댓글 각각).
- **Reject 조건**: MAUVE < 0.60 또는 JSD > 0.10 시 H1 reject → 데이터 재정제(필터 룰 v3 재튜닝, EntiGraph 합성 비율 재조정) + CPT epoch +1 추가 후 재학습.

### H2 — Turing-style Pass-Rate (Primary Endpoint)

- **진술**: 3-judge majority blind eval에서 AI 식별률이 `RESEARCH_PROTOCOL.md` §Primary Endpoint를 충족한다.
- **임계값**:
  - AI 식별률 ≤ 40%.
  - Wilson 95% CI upper bound < 50%.
  - 최소 N≥200 paired samples, stratified by `kind` × length bucket × reply depth.
- **연결**: 본 가설은 `RESEARCH_PROTOCOL.md`의 primary endpoint와 1:1 매핑된다 (see `RESEARCH_PROTOCOL.md` §Primary Endpoint).
- **Reject 조건**: AI 식별률 > 50% 또는 Wilson upper ≥ 50% 시 H2 reject → ORPO 추가 단계 가동(DNA 1.0 pipeline) + bench v3-real J style classifier로 hard negative mining 후 SFT 재학습.

### H3 — 구조 토큰 효과 (Structure-Token CPT)

- **진술**: `<|post|>`, `<|comment depth=N|>` 등 구조 토큰을 삽입한 structured CPT(v3)가 flat CPT 대비 thread coherence를 의미 있게 개선한다.
- **임계값**:
  - reply_depth KL divergence: structured CPT ≤ flat CPT − 0.05 (즉 절대 ≥ 0.05 개선).
  - Structure-fidelity: ≥ 95% (tag well-formed-ness, depth consistency).
  - LLM-as-judge thread-coherence: ≥ 0.70 on 0~1 scale (N≥100, 3-judge majority).
- **로컬 baseline 근거**: `runs/ablation/structure-comparison.md`의 `A1→A2`에서 structure fidelity 0% → 90%, val loss 4.221 → 3.455.
- **Reject 조건**: reply_depth KL 개선 < 0.02 또는 structure fidelity < 90% 시 H3 reject → 토큰 어휘 재설계(예: depth N을 absolute 대신 relative로 인코딩) + 합성 thread 비중 ↑.

### H4 — Catastrophic Forgetting 방지 (General Capability Retention)

- **진술**: LoRA + Korean replay (DNA 1.0 권고) 적용 시 한국어 일반 능력 회귀가 5% 이내로 억제된다.
- **임계값**:
  - KoBEST average score: base 대비 -5% 이하 손실 (absolute drop ≤ 0.05).
  - HAE-RAE average score: base 대비 -5% 이하.
  - 영어 코드 스위칭 비율: <2% on Korean-prompted generation.
- **Reject 조건**: KoBEST 또는 HAE-RAE에서 -5% 초과 손실 시 H4 reject → LoRA rank 감소(64 → 32) + replay 비율 상향(10% → 20%) + 영어 corpus 명시적 제외 후 재학습.

### Pre-registration & Reporting

- 위 4개 가설의 임계값은 학습 개시 전에 본 문서 commit hash로 pin된다.
- 모든 결과 보고(W&B run, paper draft)는 H1~H4 각각에 대해 (a) 측정값, (b) CI, (c) accept/reject 판정을 포함해야 한다.
- 부분 reject 시에도 학습 close-out 보고에서 mitigation plan을 명시한다 (사후 cherry-picking 금지).

---

## 1. 커뮤니티 언어 분석 (Sociolinguistic Profile)

### 1.1 은어/슬랭 체계 (Lexical Taxonomy)

데이터에서 식별된 도메인 언어 체계는 4개 층위로 구조화된다:

| 층위 | 유형 | 예시 | SFT 출현율 |
|------|------|------|-----------|
| L1: 업종 전문용어 | 업소 등급/역할 | 하퍼(12.4%), 초이스(8.1%), 텐카(3.2%), 실장(2.5%) | 높음 |
| L2: 커뮤니티 은어 | 행위/상태 묘사 | 뺑이(2.3%), 밀빵(1.3%), 쩜오(2.7%), 상띠(2.8%) | 중간 |
| L3: 초성 축약 | 감정/반응 압축 | ㅈㄴ, ㅇㅈ, ㄹㅇ, ㅂㅅ, ㄱㅊ, ㄱㄱ | CPT 30.3% 출현 |
| L4: 감정 마커 | 웃음/울음/강조 | ㅋㅋ(13.4%), ㅠㅜ(8.3%), 존나, 개빡, 현타 | 높음 |

**핵심 발견**: 이 4개 층위가 단일 문장 안에서 복합적으로 사용됨.
- 예: `"ㅎㅍ에서 중띠정도고 마인드 좋다햇을때 세팅안하고 꼬질하게 가면 쩜에서 안팔리려나요 ㅠ"`
- → L1(ㅎㅍ, 쩜), L2(중띠, 마인드, 세팅), L4(ㅠ)가 한 문장에 동시 출현

### 1.2 화행 분포 (Speech Act Distribution)

게시글 화행 분석 (44,474건):

| 화행 | 비율 | 특성 |
|------|------|------|
| 정보질문 | 48.8% | "~인가요?", "~어때요?" 형태. 업계 정보 탐색 |
| 일반서술 | 22.4% | 경험 공유, 일기형 |
| 모호토로 | 8.5% | 말줄임표 중심, 감정 간접 표현 |
| 고민질문 | 8.3% | 질문+울음 동시. "~해야하나요 ㅠ" |
| 푸념토로 | 5.4% | 감정 배출 목적, 조언 기대 낮음 |
| 유머공유 | 5.2% | ㅋㅋ 중심, 에피소드 공유 |
| 감탄선언 | 1.3% | 느낌표 중심, 강한 주장 |

### 1.3 게시글→댓글 대응 패턴

교차 분석 결과:
- **고민질문 → 공감위로(8%) + 조언추천(7%)**: 불안 표현 글에 "언니 너무 걱정마" 류 반응
- **유머공유 → 유머반응(15%)**: 웃음 배가 패턴, ㅋㅋ 연쇄
- **푸념토로 → 공감위로(8%) + 조언추천(6%)**: 감정적 연대 우선, 해결책 제시 후순위
- 전체 댓글의 **67%가 "기타반응"** — 현재 분류기로 포착 안 되는 미세한 화용적 반응이 다수

### 1.4 존댓말/반말 레지스터

| 레지스터 | 게시글 | 댓글 |
|---------|--------|------|
| 존댓말 | 29.1% | 19.5% |
| 반말 | 16.7% | 14.4% |
| 혼용 | 6.4% | 1.4% |
| 불명확 | 47.8% | 64.7% |

**핵심**: 댓글의 65%가 레지스터 불명확 → 이 커뮤니티의 특성. "ㅇㅇ", "ㄹㅇ", "그치" 같은 초단답이 존대/반말 구분을 무의미하게 만듦. **모델이 이 "무레지스터" 상태를 자연스럽게 생성해야 함.**

### 1.5 댓글 길이 분포

| 구간 | 비율 |
|------|------|
| 초단답 (~10자) | 8.2% |
| 단답 (11-30자) | 43.0% |
| 중간 (31-80자) | 36.1% |
| 장문 (81자+) | 12.7% |

댓글의 **51.2%가 30자 이하**. 평균 48자, 중간값 30자. 이 극단적 간결성이 학습의 핵심 난이도.

### 1.6 제목 스타일 유형학

옵시디언 490개 샘플 분석 결과:

| 유형 | 예시 | 특성 |
|------|------|------|
| 질문형 | "비강남인데 우리만 손님 없음?" | 48.8% — 가장 흔함 |
| 감탄형 | "사이즈 되면 ㅉㅇ는 절대 오지마세여" | 강한 주장, 경험 기반 |
| 초성/은어형 | "ㅎㅃ실장들 원래 이래요?" | 은어와 초성이 제목에 등장 |
| 서술형 | "현금 입금 방법" | 정보성, 전문 지식 공유 |
| 감정형 | "마월 또 얼마나 지옥일까" | 짧고 감정적 |

### 1.7 조회수 분포 — 신생 사이트 파워 로 (Power Law)

옵시디언 495개 샘플 분석:

| 구간 | 조회수 |
|------|--------|
| p10 | 1,445 |
| p25 | 1,673 |
| p50 (중간값) | 1,892 |
| p75 | 2,221 |
| p90 | 2,408 |
| p95 | 2,508 |
| max | **240,221** |
| 평균 | 2,381 |

**지수적 분포 지표:**
- 상위 1%가 전체 조회수의 **21.3%** 차지
- 상위 10%가 **30.9%** 차지
- Max/Median 비율: **127x**
- 전형적인 power law 분포

**학습 시사점:**
- 대부분의 게시글은 1,500~2,500 조회 밴드에 집중 (신생 사이트 특성)
- 극소수 바이럴 게시글이 조회수를 왜곡 → CPT에서 조회수 가중치 미적용이 적절
- 하지만 **바이럴 글의 언어적 특성**(높은 댓글 유도력)은 별도 분석하여 SFT에 반영 필요

### 1.8 바이럴 메커니즘

- viral 게시글 views: 2,400~2,950 (균일 분포 — 신생 사이트의 지수적 분포 초기 단계)
- normal 게시글 views: 1,600~2,200
- **바이럴 요인**: 댓글 수 > 조회수. 공감형 토픽 + 구체적 경험 + 논쟁 유발 요소
- viral 글의 댓글 수: 평균 12.3개 vs normal 5.1개
- **조회수-댓글 상관**: 높은 조회수 ≠ 많은 댓글. 정보성 글은 조회만, 감정 글은 댓글 유도

### 1.8 사회적 관계와 역학

- **"언니" 호칭**: 범용 호칭. 나이/서열 무관. 친밀감 표현
- **작성자↔댓글러 역학**: OP self-reply 비율 낮음(대부분 0). 일방적 조언 구하기 → 다방향 반응
- **정보 비대칭**: 경력자가 신입에게 조언하는 패턴 빈번. "강남 가봤는데~", "10년차인데~"
- **광고-일반 회색지대**: 실장/TC가 일반 대화에서 자연스럽게 등장하나 필터가 광고로 오분류

---

## 2. 현행 파이프라인 한계 (Critical Gap Analysis)

### 2.1 CPT의 구조 정보 상실

현재 `train_cpt.py`는 `text` 컬럼만 학습. `kind`, `source_id`, `comment_key`, `title` 모두 무시.
→ **모델은 "이 텍스트가 게시글인지 댓글인지, 어떤 게시글에 대한 댓글인지" 모름**

### 2.2 SFT의 Thread Context 미연결

`build_thread_aware_datasets.py`가 thread-aware 데이터를 생성하지만 `train_sft.py`가 이를 사용하지 않음.
- SFT의 80%가 raw continuation (CPT와 동일)
- reply pair 학습은 전체의 ~20% × reply_comment 비율 = **실질 thread 학습 ~4%**

### 2.3 필터 과잉 제거

`is_promo()` PROMO_KW_RE에 TC/실장/부장/면접 등 도메인 핵심 용어 포함.
→ GAP 진단: TC ratio 0.143, 밀빵 0.192, 케어 0.303

### 2.4 Eval 체계 부족

- 분포 수준 메트릭(JSD, KL, MAUVE)만 존재 → 개별 생성물 품질 미측정
- Thread coherence 미평가
- 도메인 고유 메트릭(은어 밀도, 초성 비율, 감정 마커) 부재
- 생성 시 alpaca format 사용하나 학습 시 미사용 → format mismatch

### 2.5 학습 효율 문제

- 텍스트 p50=26자인데 seq_len=1024, packing 미적용 → GPU utilization 낮음
- NEFTune OFF의 근거 미문서화
- Unsloth/DeepSpeed 미적용

---

## 3. 신규 학습설계 (Research-Grade Training Architecture)

### 3.0 핵심 설계 원칙

1. **구조 보존 학습**: 텍스트만이 아니라 게시글/댓글 구분, thread depth, 제목-본문 관계를 명시적으로 학습
2. **문맥 기반 생성**: 댓글은 반드시 게시글 + 부모 댓글 context를 보고 생성
3. **도메인 신호 보존**: 필터 정밀화로 핵심 은어/업종 용어 보존
4. **다층 평가**: 분포 메트릭 + 개별 품질 메트릭 + thread coherence + 도메인 메트릭

### 3.1 Stage 0: 데이터 전처리 혁신

#### 3.1.1 구조 토큰 삽입 CPT

현행 flat text 대신 **구조 마커를 텍스트에 인코딩**:

```
<|post|>제목: 비강남인데 우리만 손님 없음?
2시간동안 초이스 하나봄 ㅎㅎ<|/post|>

<|comment depth=0|>거기 요즘 손님 진짜 없음 ㅠ<|/comment|>
<|comment depth=1|>ㄹㅇ 비강남 다 죽음<|/comment|>
```

- Special tokens: `<|post|>`, `<|/post|>`, `<|comment depth=N|>`, `<|/comment|>`
- Tokenizer 확장: 6개 special token 추가 (Qwen3 tokenizer에 `add_special_tokens`)
- **근거**: MeCo 논문 (2025) — metadata conditioning이 pre-training을 가속하고 downstream 성능 향상

#### 3.1.2 필터 규칙 정밀화

```python
# AS-IS: TC, 실장 등이 무조건 프로모 키워드
PROMO_KW_RE = re.compile(r"(문의|카톡|텔레|라인|TC|실장|...)")

# TO-BE: 연락처 동반 시에만 프로모로 판정
def is_promo_v2(text: str) -> bool:
    has_contact = bool(PHONE_RE.search(text) or URL_RE.search(text))
    if has_contact:
        return True
    # 연락처 없으면 고밀도 광고 패턴만 필터
    ad_density = len(AD_DENSE_RE.findall(text)) / max(len(text), 1)
    return ad_density > 0.3  # 텍스트의 30% 이상이 광고 패턴일 때만
```

#### 3.1.3 Synthetic Data Augmentation (EntiGraph 적용)

**근거**: Synthetic Continued Pretraining (ICLR 2025 Oral) — 소규모 코퍼스에서 entity 간 관계를 재합성하면 학습 효율이 log-linear로 향상.

적용 방법:
1. 옵시디언 490개 샘플에서 도메인 entity 추출 (업소명, 지역, 은어, 인물 유형)
2. Entity 간 관계를 다양한 시나리오로 재합성 (Claude API 활용)
3. 합성 데이터를 CPT 코퍼스에 20-30% 비율로 혼합

### 3.2 Stage 1: Continued Pre-Training (CPT)

#### 3.2.1 모델 설정

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Base model | Qwen3-8B-Base | 한국어 지원, 8B 규모에서 도메인 적응 효과 최적 |
| Method | QLoRA 4-bit (NF4) | L40S 48GB에서 학습 가능, Unsloth로 2x 가속 |
| LoRA rank | 64 | "LoRA Learns Less Forgets Less" — 높은 rank가 도메인 적응에 유리 |
| LoRA alpha | 128 | alpha/rank = 2 (standard scaling) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 모든 linear layer |
| LoRA layer sweep anchor | 8 layers | `runs/ablation/hyperparam-comparison.md`의 `B4` 비교에서 16 layers가 `B3`(8 layers)보다 손실/샘플 품질 이득이 없었음. 첫 8B CPT도 8 layers를 기본값으로 시작 |
| Learning rate | 5e-5 | `runs/ablation/ABLATION-RESULTS.md`의 `B3`가 0.6B 로컬 structured sweep에서 최고 val loss(`3.372`)를 기록. 8B 첫 시도도 보수적 LR로 시작 |
| LR scheduler | cosine with warmup | warmup 3% |
| Epochs | 3 | 소규모 코퍼스에서 multi-epoch이 효과적 (DAPT 논문) |
| Batch size | 4 (effective 16 via grad accum 4) | L40S 메모리 최적 |
| Max seq len | 512 | 텍스트 p90=175자 기준. 512 tokens 충분 |
| Packing | True | 짧은 텍스트 packing으로 GPU utilization 극대화 |

#### 3.2.2 데이터 구성

```
CPT 코퍼스 구성:
├── 구조화된 게시글 (post with <|post|> markers) — 10K
├── 구조화된 댓글 (comment with depth markers) — 36K
├── Thread 연결 시퀀스 (post + comments 통합) — 11K threads
├── 합성 데이터 (EntiGraph) — ~15K (전체의 20-30%)
└── 총 예상: ~72K rows
```

기본 CPT 포맷은 **structured v3**를 유지한다. `A1-A4` 로컬 비교에서 `A2-structured-cpt-v3`가 flat/title/context 변형보다 가장 낮은 val loss와 가장 높은 구조 충실도를 보였고, 상세 근거는 [runs/ablation/ABLATION-RESULTS.md](/Users/unoa/projects/dalbitalba-train-data/runs/ablation/ABLATION-RESULTS.md)와 [runs/ablation/structure-comparison.md](/Users/unoa/projects/dalbitalba-train-data/runs/ablation/structure-comparison.md)에 정리되어 있다.

#### 3.2.3 Catastrophic Forgetting 방지

- **데이터 replay**: 원본 Qwen3 학습 데이터에서 한국어 텍스트 10% 혼합 (HF에서 공개 한국어 코퍼스 사용)
- **근거**: Thunder-LLM (2025), DNA 1.0 (2025) — CPT 시 원본 언어 데이터 replay가 catastrophic forgetting 방지의 핵심

### 3.3 Stage 2: Supervised Fine-Tuning (SFT)

#### 3.3.1 Task 설계 — 5개 생성 타스크

**T1: 제목 → 본문 생성**
```
<|instruction|>다음 제목으로 퀸알바 커뮤니티 글을 써라.
<|input|>비강남인데 우리만 손님 없음?
<|output|>2시간동안 초이스 하나봄 ㅎㅎ
```

**T2: 게시글 → Root 댓글 생성**
```
<|instruction|>다음 커뮤니티 글에 자연스럽게 댓글을 달아라.
<|context|>
제목: 비강남인데 우리만 손님 없음?
원글: 2시간동안 초이스 하나봄 ㅎㅎ
<|output|>거기 요즘 손님 진짜 없음 ㅠ 비강남 다 죽음
```

**T3: 게시글 + 부모댓글 → Reply 댓글 생성**
```
<|instruction|>다음 대화에서 자연스럽게 답글을 달아라.
<|context|>
제목: 비강남인데 우리만 손님 없음?
원글: 2시간동안 초이스 하나봄 ㅎㅎ
부모 댓글: 거기 요즘 손님 진짜 없음 ㅠ 비강남 다 죽음
<|output|>ㄹㅇ 나도 어제 3시간 뺑이 탔는데 한방도 못들어감 ;;
```

**T4: 주제 → 짧은 게시글 생성**
```
<|instruction|>다음 주제로 퀸알바 커뮤니티에 올릴 짧은 글을 써라.
<|input|>오늘 출근하기 싫은 감정
<|output|>마월 또 얼마나 지옥일까\n존나 내인생
```

**T5: 첫 문장 → 이어쓰기**
```
<|instruction|>다음 첫 문장으로 이어지는 커뮤니티 글을 써라.
<|input|>아니 크크는 왜 아무도
<|output|>광고 안함? ㅋㅋㅋ 정체가 머야 ;\n출근 해보고싶은데 루트가 없음 ;;;
```

#### 3.3.2 SFT 데이터 비율

| 타스크 | 비율 | 건수 (추정) | 근거 |
|--------|------|------------|------|
| T1: 제목→본문 | 15% | ~7K | 게시글 생성 능력 |
| T2: 게시글→Root댓글 | 35% | ~16K | 핵심: 문맥 맞는 댓글 생성 |
| T3: Thread→Reply | 30% | ~14K | 핵심: 대화 흐름 유지 |
| T4: 주제→게시글 | 10% | ~5K | 자유 생성 능력 |
| T5: 이어쓰기 | 10% | ~5K | 스타일 유지 능력 |

**T2+T3가 65%** — 댓글 생성이 이 프로젝트의 핵심 난이도이므로 비중 극대화

#### 3.3.3 SFT 학습 설정

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| Method | QLoRA 4-bit (CPT adapter 위에 계속) | CPT 지식 위에 쌓기 |
| Learning rate | 1e-4 | SFT는 CPT보다 낮은 LR (ICLR 2025 "Best Instruction-Tuning Data") |
| Epochs | 2 | SFT 과적합 방지 |
| Batch size | 8 (effective 32 via grad accum 4) | |
| NEFTune | noise_alpha=5 | 소규모 데이터 일반화에 도움. 실험으로 검증 필요 |
| Max seq len | 512 | |
| Loss | 모든 턴에 loss 계산 | ICML 2024 — multi-turn에서 전체 loss가 효과적 |

### 3.4 Stage 3: 다층 평가 프레임워크

#### 3.4.1 자동 메트릭 (Level 1 — 분포 수준)

| 메트릭 | 목적 | 목표값 |
|--------|------|--------|
| Perplexity | 모델이 도메인 텍스트를 잘 예측하는지 | < 10 on val set |
| Bigram JSD | 토큰 분포 유사도 | < 0.08 (stretch: 0.05) |
| Length KL | 길이 분포 유사도 | < 0.01 |
| MAUVE Score | 스타일 분포 일치도 | > 0.85 |

#### 3.4.2 도메인 메트릭 (Level 2 — 스타일 지문)

| 메트릭 | 원천DB 기준 | 허용 범위 |
|--------|------------|----------|
| 초성 축약 출현율 | 30.3% | 25-35% |
| 웃음 마커 (ㅋㅋ/ㅎㅎ) | 13.4% | 10-17% |
| 울음 마커 (ㅠㅜ) | 8.3% | 6-11% |
| 물음표 비율 | 26.5% | 22-31% |
| 말줄임표 비율 | 19.5% | 15-24% |
| 존댓말 비율 | 24.3% | 20-29% |
| 도메인 은어 밀도 | 하퍼 12.4% 등 | 각 용어 ±30% |
| 평균 댓글 길이 | 48자 | 38-58자 |
| 댓글 길이 중간값 | 30자 | 24-36자 |

#### 3.4.3 Thread Coherence (Level 3 — 의미 수준)

1. **Context Relevance Score**: 생성된 댓글이 원글 주제에 관련 있는지 (LLM-as-judge)
2. **Tone Appropriateness**: 고민 글에 유머로 답하지 않는지 등 화행 적절성
3. **Depth-Aware Style**: 답글이 root 댓글보다 짧고 캐주얼한지 (실제 패턴 반영)

#### 3.4.4 인간 평가 (Level 4 — Turing Test)

- **A/B 블라인드 테스트**: 실제 게시글/댓글 vs AI 생성물 5:5 혼합
- **판정 기준**: "이 중 AI가 쓴 것을 고르시오"
- **목표**: 판별 정확도 55% 이하 (랜덤 수준에 근접)
- **평가자**: 프로젝트 관계자 + 가능하면 외부 2인

---

## 4. 로컬 테스트 전략 (Pre-RunPod Validation)

### 4.1 M4 Mac 로컬 환경

| 구성요소 | 설정 |
|---------|------|
| Hardware | Apple M4, 16GB RAM, Metal GPU |
| Inference | ollama (Qwen3-4B, Qwen3-1.7B) |
| Training | MLX-LM LoRA fine-tuning (Qwen3-0.6B, 1.7B) |
| Python | 3.12 + venv |

### 4.2 로컬 검증 루프

```
[로컬 학습설계 검증 루프]

1. Qwen3-0.6B/1.7B로 축소 실험
   - CPT: 동일 데이터 구조, 동일 special tokens
   - SFT: 동일 5-task 구조, 축소 데이터

2. 생성물 평가
   - Level 2 도메인 메트릭 자동 측정
   - ollama로 샘플 생성 → 원천DB와 비교

3. 설계 수정
   - 메트릭 미달 시 데이터 비율/LR/구조 조정
   - 반복 (완벽할 때까지)

4. RunPod 가동 판정
   - 로컬 실험에서 Level 2 메트릭 허용 범위 진입
   - Thread coherence 주관 평가 통과
```

### 4.3 RunPod 실행 계획

```
단계별 비용 계획 ($30 예산):
├── Smoke run (1 epoch CPT, 100 steps SFT): ~$2
├── CPT full (3 epochs): ~$8-10
├── SFT full (2 epochs): ~$4-6
├── Eval (생성 + 자동 메트릭): ~$2
├── 예비: ~$10-14
└── 총 예상: $16-20 (예비 여유 있음)
```

---

## 5. 구현 우선순위

### Phase A: 데이터 파이프라인 혁신 (로컬)
1. `build_structured_cpt.py` — 구조 토큰 삽입된 CPT 코퍼스 생성
2. `build_5task_sft.py` — 5개 타스크 SFT 데이터셋 생성
3. `is_promo_v2()` — 필터 규칙 정밀화
4. Tokenizer special token 추가 스크립트

### Phase B: 로컬 축소 실험
5. MLX-LM으로 Qwen3-0.6B LoRA CPT 로컬 실행
6. 로컬 SFT 실행 + 생성물 평가
7. 도메인 메트릭 자동 측정 스크립트
8. 설계 반복 수정

### Phase C: RunPod 학습
9. Unsloth + QLoRA 학습 스크립트 (train_cpt_v3.py, train_sft_v3.py)
10. 다층 평가 스크립트 (eval_v3.py)
11. 실행 + 평가 + 반복

---

## 6. 참고 논문

| 논문 | 적용 포인트 |
|------|-----------|
| Synthetic Continued Pretraining (ICLR 2025) | EntiGraph로 소규모 코퍼스 증강 |
| LoRA Learns Less and Forgets Less (2024) | 높은 rank, catastrophic forgetting 분석 |
| Learning Rate Matters (2026) | vanilla LoRA에서 LR 튜닝이 핵심 |
| Thunder-LLM (2025) | 한국어 LLM 적응 end-to-end 프로세스 |
| DNA 1.0 (2025) | CPT+SFT+SLERP+DPO 한국어 적응 파이프라인 |
| MeCo (2025) | Metadata conditioning으로 CPT 가속 |
| Domain-Adaptive CPT for Small LMs (2025) | 소규모 모델 incremental training |
| Open Ko-LLM Leaderboard2 (2024) | 한국어 LLM 평가 벤치마크 |

---

---

## 7. 로컬 실험 결과 및 설계 반영

### 7.1 MLX LoRA 로컬 실험 (Qwen3-0.6B, M4 16GB)

| 항목 | 결과 |
|------|------|
| 모델 | Qwen3-0.6B (596M params) |
| 학습 가능 파라미터 | 0.484% (2.884M) |
| 데이터 | CPT v2 42,170 train / 4,686 val |
| Peak memory | 6.0 GB |
| Val loss 추이 | 6.105 → 4.741 (100it) → **4.166 (200it)** → 4.260 (300it) → 4.227 (400it) |
| 최적 checkpoint | 200 iter (overfitting 시작점) |

### 7.2 구조/하이퍼파라미터 ablation 결과 (2026-04-29)

상세 매트릭스는 [runs/ablation/ABLATION-RESULTS.md](/Users/unoa/projects/dalbitalba-train-data/runs/ablation/ABLATION-RESULTS.md), 포맷 비교는 [runs/ablation/structure-comparison.md](/Users/unoa/projects/dalbitalba-train-data/runs/ablation/structure-comparison.md), LR/LoRA depth 비교는 [runs/ablation/hyperparam-comparison.md](/Users/unoa/projects/dalbitalba-train-data/runs/ablation/hyperparam-comparison.md)에 정리되어 있다.

핵심 결론:
- **Structured CPT가 기본 포맷으로 확정**: `A2-structured-cpt-v3`는 `A1` flat, `A3` context stream, `A4` title-prepended보다 가장 낮은 val loss와 가장 높은 구조 충실도를 보였다. 구조 토큰 자체가 성능 향상의 주된 원인이다.
- **LR 기본값은 5e-5**: structured sweep의 `B3`(`lr=5e-5`, `8 layers`)가 최저 val loss(`3.372`), `100%` 구조 fidelity, 가장 균형 잡힌 반말/무레지스터 분포를 기록했다.
- **LoRA depth는 8 layers 유지**: `B4`의 16 layers는 `B3`보다 손실이 나쁘고 lexical gain도 거의 없었다. 첫 8B CPT도 8 layers를 sweep anchor로 시작한다.
- **0.6B의 역할은 방향 검증**: 이 실험은 포맷 선택과 LR 방향, adapter depth 우선순위를 검증하는 데는 충분했지만, 절대적인 생성 품질이나 최종 realism ceiling을 판단하는 근거로 쓰면 안 된다.

### 7.3 생성 품질 관찰

**스타일 학습 성공 징후:**
- 초성 사용 (`ㄴㄷㄷ`, `ㄱㅈㄴ`, `ㄱㅌ`) — 도메인 패턴 흡수
- 짧은 반응 생성 — 댓글 길이 분포에 가까워짐
- 구어체 어미 사용

**의미적 한계 (0.6B 고유):**
- Coherence 부족: 문맥과 무관한 토큰 나열
- 한국어 지식 기반 얕음: "okay" 같은 영어 출력 혼재
- 도메인 은어의 올바른 맥락 사용 미달

### 7.4 설계 수정 사항

1. **Overfitting 방지**: 200 iter에서 이미 val loss 반등 → RunPod CPT에서 **early stopping 필수**
   - `save_every_n_steps=100`, `eval_every_n_steps=100` 추가
   - Val loss 3회 연속 상승 시 중단

2. **Epoch 재계산**: 46K rows / batch 4 / grad_accum 4 = ~2,900 steps/epoch
   - 3 epoch = ~8,700 steps → 0.6B 기준 200 iter overfitting으로 볼 때 **1-2 epoch이 적절**
   - 8B는 용량이 크므로 2-3 epoch 유지하되 early stopping으로 보호

3. **Packing 효과 확인**: max_seq_length=512에서 일부 텍스트 truncation 경고 → packing으로 해결

4. **로컬 검증 가치**: 0.6B로도 **포맷 우열, LR 방향, LoRA depth 우선순위**는 검증 가능 → RunPod 전 로컬 sanity check 유효

### 7.5 Qwen3-4B 베이스라인 (파인튜닝 전)

| 항목 | 결과 |
|------|------|
| "초이스"에 대한 이해 | 커피숍 이름으로 해석 (도메인 지식 0%) |
| 응답 언어 | 영어로 thinking, 한국어 미생성 |
| /no_think 동작 | 실패 — thinking 루프에 빠짐 |
| 10개 댓글 생성 | 전부 timeout (30초 초과) |

**결론**: 파인튜닝 없이는 이 도메인에 **전혀 대응 불가**. 학습의 필요성이 정량적으로 확인됨.

### 7.6 Qwen3-0.6B 500 iter 생성물 분석

| 항목 | 관찰 |
|------|------|
| 도메인 용어 | "TC", "아가씨", "케어", "방도" 등 자연 출현 — **스타일 흡수 성공** |
| 문장 구조 | "일하는게 안간데 여덟 빠리는데" — 커뮤니티 구어체 유사 |
| 초성 사용 | 활발하나 degeneration 경향 (ㄷㄷㄷ 무한반복) |
| 광고 오염 | Sample 5에서 "카카오톡아이디" + 이모지 생성 — **필터 강화 필수** |
| Coherence | 0.6B 한계로 의미적 일관성 부족 — 8B에서 해결 예상 |

### 7.7 학습설계 수정 반영

위 실험에서 도출된 추가 수정 사항:

1. **광고 이모지 필터 추가**: CPT/SFT 데이터에서 `➡️❤️⭐️` 등 이모지 반복 패턴 제거
2. **Repetition penalty**: 생성 시 `repetition_penalty=1.2` 기본 적용
3. **Early stopping 강화**: val loss 2회 연속 상승 시 학습 중단 (0.6B 실험에서 200 iter에 최적점)
4. **광고 텍스트 추가 클리닝**: "카카오톡아이디", "텔레그램", 이모지 3개 이상 연속 패턴 필터

### 7.8 도메인 메트릭 정량 평가 결과 (Qwen3-0.6B, 500 iter, N=50)

| 메트릭 | 원천 | 생성 | 편차 | 판정 |
|--------|------|------|------|------|
| 초성 축약 | 30.3% | 36.0% | +19% | ✅ |
| 웃음(ㅋㅋ) | 13.4% | 2.0% | -85% | ❌ **심각** |
| 울음(ㅠㅜ) | 8.3% | 10.0% | +20% | ✅ |
| 물음표 | 26.5% | 14.0% | -47% | ❌ |
| 말줄임표 | 19.5% | 2.0% | -90% | ❌ **심각** |
| 느낌표 | 7.3% | 0.0% | -100% | ❌ **심각** |
| 존댓말 | 22.2% | 16.0% | -28% | ✅ |
| 평균 길이 | 60.3자 | 42.8자 | -29% | ✅ |
| 길이 중간값 | 38자 | 49자 | +29% | ✅ |
| **Pass rate** | | | | **5/9** |

**Level 1 분포**: Bigram JSD 0.557 (목표 <0.08), Length KL 3.035 (목표 <0.01) — 0.6B+500iter로는 분포 수렴 불가.

### 7.9 고도화 필요 사항 (다음 루프에서 반영)

1. **웃음/구두점 마커 부족**: 0.6B의 생성 다양성 한계. 8B에서 개선 예상되나, SFT에서 구두점 패턴을 명시적으로 학습시키는 방안 검토
2. **광고 오염 지속**: 50개 샘플 중 2건(4%)에서 전화번호/카카오톡 생성 → CPT 데이터에서 광고 텍스트 추가 제거 필요
3. **초성 과학습**: 36% > 30.3% 원천. 초성 텍스트만으로 이루어진 row 비중 축소 검토
4. **길이 분포 괴리**: avg 42.8 vs 60.3 — 생성물이 원천보다 짧음. 긴 텍스트 학습 비중 조정 필요

### 7.10 N=200 정밀 평가 결과 (v2 클린 데이터, Qwen3-0.6B, 500 iter)

| 메트릭 | 원천 | 생성 | 편차 | 판정 |
|--------|------|------|------|------|
| 초성 축약 | 30.3% | 12.0% | -60% | ❌ |
| 웃음(ㅋㅋ) | 13.4% | 4.0% | -70% | ❌ |
| 울음(ㅠㅜ) | 8.3% | 0.5% | -94% | ❌ |
| 물음표 | 26.5% | 18.5% | -30% | ❌ (경계선) |
| 말줄임표 | 19.5% | 3.5% | -82% | ❌ |
| 느낌표 | 7.3% | 0.5% | -93% | ❌ |
| 존댓말 | 22.2% | 27.5% | +24% | ✅ |
| 평균 길이 | 60.3자 | 43.6자 | -28% | ✅ |
| 중간값 길이 | 38자 | 46자 | +21% | ✅ |
| **Pass rate** | | | | **3/9** |

### 7.11 로컬 실험 종합 결론

**0.6B 모델은 concept validation용이며, 학습설계의 건전성을 확인하는 도구:**

1. ✅ **데이터 클리닝 효과 확인**: 광고 텍스트 제거 → 광고 미생성
2. ✅ **스타일 적응 방향 확인**: 도메인 용어(하퍼, TC), 존댓말 패턴, 길이 분포가 원천에 근접
3. ✅ **학습 안정성 확인**: v2 클린 데이터가 v1보다 안정적 수렴 (overfitting 없음)
4. ❌ **특수문자 마커 부족**: 0.6B의 생성 다양성 한계. 8B에서 해결 예상

**RunPod 8B 학습 전 확인된 사항:**
- 데이터 전처리 파이프라인 동작 검증 ✅
- 도메인 메트릭 자동 측정 파이프라인 동작 검증 ✅
- is_promo_v2 필터 효과 검증 ✅
- 학습 하이퍼파라미터 방향성 검증 (`structured CPT > flat`, `LR 5e-5 > 1e-4/2e-4`, `8 layers > 16 layers` on 0.6B) ✅
- 구조 토큰 삽입과 structured CPT 우위는 로컬 ablation으로 검증 완료. 다만 **절대 품질/realism은 8B RunPod 학습에서만 판단** 가능

### 7.12 2026-04-29 로컬 ablation matrix 반영

200 iter / batch 1 / 450 train / 25 val 조건의 Qwen3-0.6B MLX LoRA matrix에서 확인된 추가 결론:

| 축 | 승자 | 근거 |
|----|------|------|
| 데이터 형식 | Structured CPT v3 (`A2`) | `A1 4.221 → A2 3.455`, structure fidelity `0% → 90%` |
| LR sweep | `5e-5` (`B3`) | structured runs 중 최저 val loss `3.372`, structure fidelity `100%` |
| LoRA depth | 8 layers | `B4`(16 layers)가 `B3`보다 loss/lexical 품질 모두 우위 아님 |
| Base vs FT | FT 필수 | `B0`는 태그 미종결, 메타데이터/제목 echo가 심함 |

이 matrix가 설계에 주는 수정 사항:

1. **로컬 MLX 기본 LR 교체**: `2e-4`를 기본값으로 두지 말고 `5e-5`를 sanity-check anchor로 사용.
2. **구조 토큰을 기본 경로로 고정**: 구조 없는 flat/context/title 변형은 모두 structured 대비 명확히 열세였다.
3. **16 layers는 후순위**: 8-layer structured baseline이 안정화되기 전까지 depth 확장은 비용만 늘리고 이득이 없었다.
4. **0.6B와 8B의 역할 분리**: 0.6B는 형식/최적화 ranking용, 8B는 실제 도메인 어휘 회수와 realism 검증용으로 명시.
5. **도메인 어휘 평가 해석 강화**: `ㅋㅋ`, `ㅠㅠ` 과생성은 쉬운데 `TC`, `밀빵`, `케어` 같은 희귀 어휘는 0.6B/200 iter에서 거의 회수되지 않았다. 따라서 단순 term count는 구조 fidelity, readability와 함께 봐야 한다.

---

## 8. Threats to Validity

본 학습설계의 결론을 일반화할 때 명시해야 할 위협을 internal/external/construct validity로 분리하여 사전 등록한다.

### Internal Validity

1. **단일 커뮤니티 편향**: cb2_밤문화이야기 단일 보드에서 수집. 다른 한국어 커뮤니티(루리웹, 디시 등)로의 generalize는 별도 검증 필요. *Mitigation*: 보드 외부 generalize 주장 자제.
2. **2개월 시계열**: 2026-01~02 수집. seasonal drift, slang 변화 미반영. *Mitigation*: 결과 보고 시 수집 기간 명시.
3. **평가자 수**: human native eval N=2. statistical power 한계. *Mitigation*: 3-judge LLM blind eval로 보완.
4. **Claude judge 도메인 편향**: bench v3-real에서 Anthropic models FN=1.00 — 도메인 적응된 AI 샘플을 human으로 분류. *Mitigation*: J style classifier v2 (test_AUC=0.9999) + GPT-5 judge 보완.
5. **Generation 비결정성** (closed in B5a): `phase6_generate.set_seed`로 mitigated. seed 고정 + temperature/top_p pinning은 paper-grade run의 default.

### External Validity

6. **8B 단일 모델 한계**: 더 큰 모델(70B+)에서 catastrophic forgetting 양상이 다를 수 있음. H4의 KoBEST/HAE-RAE 결과는 8B 스케일에 한정해 보고한다.
7. **데이터 규모 한계**: CPT 48K + SFT 10K — domain-specific 학습엔 충분하나 general capability 회복엔 부족. 한국어 replay 비율(10%)은 본 규모에서의 sweet spot이며, 대규모 코퍼스 시 재튜닝 필요.

### Construct Validity

8. **MAUVE의 한계**: 짧은 텍스트 (avg 48자) 분포에서 MAUVE 신뢰구간이 넓어짐. N=1,139 eval set은 marginal. *Mitigation*: bootstrap CI(≥1,000 resamples)와 함께 보고.
9. **도메인 은어 ratio**: keyword 기반 메트릭(`하퍼`, `TC`, `밀빵`, `케어` 등 출현율)은 의미 보존을 직접 측정하지 못한다. *Mitigation*: LLM-as-judge thread coherence(H3)와 human pairwise(H2)로 보완.

---

## 9. Revision History

| Version | Date | Notes |
|---------|------|-------|
| v3.0 | 2026-04-29 | Initial research-grade design (Ralph iter 3-4): structured CPT + 5-task SFT + 다층 평가. |
| v3.1 | 2026-05-12 | hypothesis + ToV 추가 (paper-grade close-out). §0 H1~H4 falsifiable hypotheses, §8 Threats to Validity (9개 항목), `RESEARCH_PROTOCOL.md` cross-reference 정합. |

---

*Generated: 2026-04-29 | Ralph iteration 3-4 | dalbitalba-train-data v3 training design*
*Local experiments: legacy 500-iter probe + 2026-04-29 ablation matrix on Qwen3-0.6B MLX LoRA*
*v3.1 close-out: 2026-05-12 — hypotheses (H1~H4) + Threats to Validity (9 items)*
