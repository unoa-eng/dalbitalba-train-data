# dalbitalba-train-data — Paper-grade Analysis & Hardening Plan

**작성**: 2026-05-07 KST
**대상 레포**: `unoa-eng/dalbitalba-train-data` @ `main:ba4056f`
**대상 베이스 모델**: `Qwen/Qwen3-8B-Base` (8.2B, 36 layers, 32 Q heads / 8 KV heads, ctx 32K, BPE ~151K, bf16, Apache-2.0, arXiv:2505.09388)
**분석 입력**:
- 코드베이스 인벤토리 (`/tmp/codebase_inventory.md`, 1,140 lines, 17 sections)
- 외부 SOTA 리서치 (`/tmp/research_korean_llm_sota.md`, 508 lines, 10 sections, 35+ citations)
- 이전 산출물 (HF artifacts, bench/v3-real-20260421, calibration JSON, phase1_summary)
- 데이터 직접 프로파일링 (CPT/SFT/val 실측)

---

## 0. Executive Summary (TL;DR)

본 프로젝트는 **한국 유흥 커뮤니티 도메인 LLM 파인튜닝**이라는 niche 문제에 대해 *체계적인 7-stage MLOps 파이프라인*을 구축했다. RunPod 자동화, 5-metric deterministic gate, recipe mutation rulebook, GitHub Actions bridge 등 **인프라 성숙도는 동급 오픈소스 프로젝트보다 우수**하다.

그러나 **모델 품질의 핵심 병목 4가지**가 인프라가 아닌 *학습 설계 자체*에 존재한다:

1. **CPT가 LoRA 한계에 정면 충돌** — arXiv:2405.09673 ("LoRA Learns Less and Forgets Less", May 2024)는 도메인 CPT에서 LoRA가 풀 FT 대비 **16× 데이터 비효율**임을 입증. 0424-0618 run의 *CPT 46.75% 완료 / SFT 빈 어댑터*는 이 실패 모드와 정확히 일치한다.
2. **데이터 오염이 학습 신호를 압도** — 실측 결과 CPT v2 코퍼스의 **21.6%가 광고 키워드 보유**, dedup threshold 0.85에서 19개만 제거됨에도 실효 dup_rate ~0.4. 손실의 상당 부분이 반복 광고 템플릿에 흡수되어 SFT phase까지 도달하지 못한 것이 0618 run 실패의 직접 원인일 가능성이 매우 높다.
3. **NFKC 정규화 양면성** — `phase1_data_pipeline.py:88`은 NFKC 정규화로 Compatibility Jamo를 분해, `fix_jamo_normalization.py`는 별도 사후 복원. 결과: CPT 코퍼스 jamo 비율이 raw 2.0% → 2026-04-24 시점 3.06%로 **54% 증가**(이중 정규화 흔적). Qwen3 토크나이저에서 ㅈㄴ/ㅇㅈ/ㄹㅇ 같은 *초성 슬랭*이 분리된 codepoint로 매핑되어 *long-tail vocabulary failure*를 유발.
4. **평가 시스템이 부분적으로 무효** — `bench/v3-real-20260421`에서 Claude Opus 4.7 / Sonnet 4.6 두 judge 모두 **모든 50개 샘플을 human으로 예측 (acc=0.5)**. GPT-5만 88% 정확도. 즉 3-judge 합의 시스템에서 Claude 두 노드는 사실상 무용. 또한 phase6의 5-metric은 분포 매칭만 — 의미적 일관성, 스레드 코히런스, 한국어 벤치마크 회귀 검사 모두 부재.

**Hardening 방향 (P0→P3)**:
- P0: MinHash dedup + 광고 정규식 강화 + Compatibility Jamo 보존을 phase1에 통합 (GPU 비용 0)
- P1: thread-aware SFT 포맷 (`(post, parent_comment, target_reply)`), CAI 비율 정상화 (6.8% → 33%)
- P2: CPT를 r=256 rsLoRA + DoRA 또는 fp16 풀 FT로 전환 (Unsloth 백엔드로 비용 흡수)
- P3: HRET/KoBEST 회귀 게이트, klue/roberta 기반 thread-coherence 메트릭, GPT-5 + 스타일 분류기 2-judge 시스템

본 문서는 **(1) 의도 / (2) 베이스 모델 / (3) 데이터 / (4) 학습 / (5) 평가 / (6) 운영 / (7) 산출물**의 7개 layer에서 *증거-기반*으로 진단하고, 각 진단마다 *측정 가능한 ROI*를 가진 개선안을 제시한다.

---

## 1. 프로젝트 의도 재구성 (Intent Layer)

코드베이스, 메모리, 문서를 종합하면 본 프로젝트의 의도는 다음과 같이 재구성된다:

> **"한국 유흥업 구인구직 게시판(달빛알바/queenalba 도메인)의 실제 게시글·댓글 분포와 LLM 생성물의 간극을 'blind eval에서 사람/AI 구분 난도가 50% 동전던지기에 수렴'할 정도까지 줄인다."**
> *(출처: `AUTOPILOT_LOOP.md` "목표" 섹션)*

이 정의에서 도출되는 *3개 부분 목표*:
- **G1. Stylistic match**: 초성/은어/짧은 턴/감정 부호/말줄임표 등 비표준 구어체의 분포 일치
- **G2. Structural match**: 게시글 → 루트 댓글 → 답글 댓글의 thread 구조 보존, 광고/운영자 톤 배제
- **G3. Behavioral safety**: 미성년/성매매 등 hard filter, PII 스크러빙

**현재 파이프라인의 G1/G2/G3 달성도 (정량)**:

| 목표 | 측정 지표 | 현재 상태 | 목표 | Gap |
|---|---|---|---|---|
| G1 stylistic | bigram_jsd vs raw | unknown (0618 SFT empty) | ≤ 0.08 (calibration primary), 0.05 (stretch) | full unknown |
| G1 stylistic | length_kl vs raw | unknown | ≤ 0.10 | full unknown |
| G2 structural | thread-aware SFT 사용률 | 0% (단순 (post, comment) 페어) | ≥ 80% | 80pt |
| G2 structural | CPT에서 kind/title/board 보존 | 0% (`text` 단일 컬럼) | ≥ 100% | 100pt |
| G2 structural | 광고 댓글 잔존율 | **21.6% (실측)** | ≤ 5% | -17pt |
| G3 behavioral | PII 스크러빙 | 휴대전화 14,557건 / URL 1,153건 마스킹 | full | OK |
| G3 behavioral | 미성년 hard filter | implemented | full | OK |
| G3 behavioral | 광고 hard filter | partial regex | ≥ 95% precision | gap |

**의도-구현 간 가장 큰 불일치**: G2 structural. *"댓글 문맥과 reply 흐름까지 닮아야 한다"*가 design doc의 핵심 원칙이지만, 현 SFT는 `{post, comment}` 단일 페어만 학습 — 부모 댓글, thread depth, board, kind를 모두 버린다. CPT 또한 `{text}`만 학습한다 (`train_cpt.py:114-130`).

---

## 2. 베이스 모델 분석 (Qwen3-8B-Base)

### 2.1 아키텍처 / 토크나이저 사실관계

| 속성 | 값 | 출처 / 검증 |
|---|---|---|
| Total params | 8.2B | HF model card |
| Non-embedding params | 6.95B | HF model card |
| Layers | 36 | HF model card |
| Heads (Q / KV) | 32 / 8 (GQA) | HF model card |
| Hidden / Intermediate | 4096 / 12288 | Qwen3 paper, arXiv:2505.09388 |
| Context length | 32K (deployable up to 128K) | Qwen3 paper |
| Vocab | ~151K BPE | Qwen3 paper |
| Languages | 119 (Korean explicitly supported) | Qwen3 paper |
| Pre-training tokens | 36T | Qwen3 paper |
| dtype (training) | bf16 | HF model card |
| License | Apache-2.0 | HF model card |
| Required transformers | ≥ 4.51.0 | HF model card |

### 2.2 한국어 적합성 (정성 / 정량 부재 → 측정 권고)

**Qwen3-8B-Base는 다음 이유로 한국어 도메인 fine-tune의 *기본 후보*는 적절하다**:
- ✅ Apache-2.0 (상업 사용 가능)
- ✅ 36-layer / 8.2B는 LoRA fine-tune 비용/품질 sweet spot (vs 14B/32B)
- ✅ GQA로 추론 메모리 효율
- ✅ 119 언어에서 한국어가 explicit support

**그러나 다음 미확인 risk가 있음**:
- ⚠️ **Korean fertility 미측정** — Qwen3 BPE 토크나이저가 한국어 chars/token 비율이 알려지지 않음. Thunder-Tok (arXiv:2506.15138) 측정에서 표준 BPE는 한국어 기준 1.509 tokens/char, 언어 인식 토크나이저는 1.370. 8K seq_len을 1024 char로 잘랐을 때 실효 정보량을 측정해야 함.
- ⚠️ **초성 슬랭 BPE 머지 부재** — Qwen3 사전훈련 코퍼스에 ㅈㄴ/ㅇㅈ/ㄹㅇ 같은 도메인 초성이 충분히 등장했을 가능성 낮음. 이는 *각 자모가 개별 rare token ID로 매핑* → 임베딩 미학습. KatFishNet (arXiv:2503.00032)이 이 패턴을 LLM-generated text의 핵심 식별 신호로 사용.
- ⚠️ **Compatibility Jamo (U+3130-U+318F) vs Conjoining Jamo (U+1100-U+11FF) 매핑 불명** — NFKC 정규화 시 두 codepoint 클래스가 다르게 처리되며, Qwen3 BPE가 어느 쪽을 학습했는지 사전 확인 필요.

**권고 측정 (15분 미만, GPU 불필요)**:
```python
# scripts/measure_qwen3_korean_fertility.py 신규 작성 권고
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
import json, statistics
samples = [json.loads(l)['text'] for l in open('val_set.v2.jsonl')]
char_per_tok = [len(s) / max(1, len(tok.encode(s))) for s in samples]
print(f"chars/token p50={statistics.median(char_per_tok):.3f} mean={statistics.mean(char_per_tok):.3f}")
# 초성 단독: 'ㅈㄴ', 'ㅇㅈ', 'ㄹㅇ', 'ㅋㅋ'
for slang in ['ㅈㄴ', 'ㅇㅈ', 'ㄹㅇ', 'ㅋㅋ', '하퍼', '쩜오', '텐카', '초이스']:
    ids = tok.encode(slang, add_special_tokens=False)
    print(f"{slang!r:>8} -> {len(ids)} tokens: {[tok.decode([i]) for i in ids]}")
```

이 측정 없이 CPT 성공 가능성을 평가할 수 없음.

### 2.3 비교 베이스 모델 평가 (대안 검토)

SOTA 리서치에서 식별된 한국어 8B 후보 중 **현 프로젝트에 더 적합할 수 있는 대안**:

| 베이스 | 한국어 vocab 확장 | 벤치마크 | 라이선스 | 권고 |
|---|---|---|---|---|
| **Qwen3-8B-Base** (현재) | ✗ (151K, 한국어 fertility 미측정) | Open KO LLM 미실행 | Apache-2.0 | continue, 측정 필요 |
| `MLP-KTLim/llama-3-Korean-Bllossom-8B` | ✓ (+30K Korean) | LogicKor 6.93 | LLaMA-3 (commercial-friendly) | **CPT base 후보 #1** |
| `beomi/Llama-3-Open-Ko-8B` | ✓ continued pretrained | Open Ko-LLM compete | LLaMA-3 | CPT base 후보 #2 |
| `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` | ✓ BBPE 102K | KMMLU compete | EXAONE License (제한적) | 라이선스 검토 필요 |

**전략적 선택**: 현재 Qwen3-8B를 *유지*하되, Bllossom-8B를 **CPT base로 사용하는 dual-track 실험**을 권고. Bllossom은 이미 한국어 vocab + DPO를 거쳤으므로 *추가 CPT 비용을 절감*하면서 도메인 적응만 하면 된다. arXiv:2405.09673의 결과 — Bllossom 위에 LoRA SFT만 (CPT 생략) — 는 현재 $30 budget 내에서 *유의미한 dalbitalba 스타일 적응*이 가능할 가능성이 매우 높다.

---

## 3. 데이터 레이어 분석 (Data Pipeline)

### 3.1 입력 / 가공 / 출력 흐름

```
raw crawl (Windows queenalba-crawler):
    ├─ 11,288 posts, 50,811 comments (post-spam filter)
    ├─ 67,136 records total, 11,287 unique threads, 2,453 authors
    │
    ▼ (phase1_data_pipeline.py)
PII scrubbing:        14,557 phone, 1,153 url
Hard filters:          1 minor-proximity drop, 5,038 spam, 67 short
Dedup (Jaccard ≥0.85): 19 dropped (← extremely lenient)
NFKC normalize:       LINE 88 (Compatibility Jamo broken here)
Length-bucket oversample: 2× lg/xl/xxl
Mix policy:           80% raw continuation + 20% reply pair
    │
    ▼
cpt_corpus.v2.jsonl     46,856 rows, schema {text, kind, source_id, source_field, length_bucket}
sft_pairs.v2.jsonl      44,474 rows, schema {post, comment, thread_key, source_id, length_bucket}
val_set.v2.jsonl         2,451 rows  (val/(train+val) = 2.6%, target was 5%)
cai_pairs.filtered.jsonl 1,743 rows  (constitutional AI triples)
```

### 3.2 실측 프로파일 (이 분석에서 직접 측정)

**CPT v2 corpus** (n=46,856):
- avg_len = 60.3 chars (xs=15.7%, sm=47.0%, md=24.1%, lg=8.8%, xl=4.3%, xxl=0.2%)
- char class density: hangul 63.4%, **jamo 3.06%** ← raw baseline 2.0%, +54% 증가
- **english density 0.6%** ← raw baseline 2.2%, **-72% 감소**
- digit density 4.6% ← raw baseline 7.9%, -42% (PII 마스킹 후 잔여)
- 초성/감정 슬랭 보유 레코드: 26.3%
- **광고 키워드 보유 레코드: 21.6%** ← clean_ad_spam.py 우회 잔여

**SFT pairs v2** (n=44,474):
- post p50=64 / mean=100 / max=1804 chars
- comment p50=30 / mean=48 / max=1114 chars
- **comment < 20 chars: 30.4%** (의미적으로 빈약한 댓글이 1/3)
- 슬랭 보유: post 21.3%, comment 17.9%

**val_set v2** (n=2,451):
- post 560 (22.8%) / comment 1,891 (77.2%) — train 분포와 일관

**CAI pairs filtered** (n=1,743):
- 형식: `{topic, category, draft, critique, revision}`
- **draft = AI-style (또한, ~것 같아요), critique = 메타 비평 (문어체 어미 남발 지적), revision = 도메인 톤 (쩜오 원피스 무조건 올탈 강요...)`
- *높은 품질의 supervised contrastive 데이터*. 그러나 `TRAINING_RECIPE_VALIDATION.md`에서 "CAI_RATIO=0.33이지만 실효 비중 6.8%"로 명시 — 의도와 실제 사이에 5× gap.

### 3.3 데이터 결함 진단

**D1. Dedup이 사실상 무효 (CRITICAL, P0)**
- `phase1_data_pipeline.py` line 21: thread 내부 Jaccard ≥ 0.85, char 4-gram 기반
- phase1_summary.json: dropped_near_dup = **19**
- 그러나 `local_verification_loop.py`의 dup_rate 검사는 0.396 (CPT) / 0.382 (SFT) WARN
- 모순 해석: thread 내부 dedup만 적용, **thread 간 dedup 부재**. 즉 동일 광고 템플릿이 여러 thread에 복제될 때 모두 통과
- **영향**: 학습 손실의 상당 부분이 반복 광고 패턴에 의해 dominate
- arXiv:2406.17557 (FineWeb): 큰 cluster (100+ duplicates) dedup이 가장 큰 perplexity 이득 제공
- **0618 run의 CPT 46.75%/SFT empty 실패가 이 결함의 직접 결과일 가능성이 가장 높음**

**D2. 광고/운영자 정규식 누수 (CRITICAL, P0)**
- 실측 21.6%의 CPT v2 레코드에 광고 키워드 (카톡, 오픈톡, 텔레, VIP, 문의, 출근, 언니, 지원 등)
- `clean_ad_spam.py`의 `AD_PATTERNS`는 *일부 패턴만* 잡음 (e.g., `r'카톡\s+[a-zA-Z][a-zA-Z0-9]{2,}'`)
- 음성 패턴 누락:
  - 운영자 자기홍보 ("저희 가게는...", "출근시 연락주세요")
  - 가게명 + 추천 (스팸성 vs 사용자성 모두 동일 형태)
  - 반복 운영진 케어 문구 ("언제든 편하게 연락주세요")
  - 단축 URL / 비-Latin handle
- arXiv:2411.04257 (LSHBloom) + character n-gram entropy < 2.8 bits/char 임계 적용 권고 (FineWeb 패턴)

**D3. NFKC 이중 정규화 의심 (HIGH, P1)**
- Phase1 line 88: `text = unicodedata.normalize("NFKC", text)` — Compatibility Jamo (ㅋ U+314B 등) → Conjoining Jamo (ᄏ U+1100 등) 또는 결합형으로 변환
- 별도 `scripts/fix_jamo_normalization.py`: NFKC 후 깨진 Jamo를 Compatibility Jamo로 복원
- 문제: phase1과 fix_jamo가 *분리되어 있어* 적용 누락 또는 이중 적용 risk
- 데이터 실측: jamo 비율 raw 2.0% → CPT v2 3.06% (54% 상승) — 복원이 적용된 흔적이지만 codepoint 일관성 미보장
- arXiv:2506.15138 (Thunder-Tok): 표준 BPE는 한국어 syllable + 분리 자음 같은 malformed substring 처리 실패
- **권고**: phase1에서 NFKC 후 즉시 Compatibility Jamo 복원을 *원자적 단계*로 통합

**D4. 짧은 댓글 우세 (MEDIUM, P1)**
- SFT 페어 30.4%가 `comment < 20 chars`
- 이런 댓글은 thread coherence 관점에서 학습 신호 빈약 (e.g., "ㅇㅈ", "ㄱㄱ", "ㄹㅇ")
- arXiv:2411.15124 (Tulu-3) / LIMA: 1K 큐레이션 데이터가 대규모 미정제 SFT를 능가
- **권고**: SFT를 quality-filter (length ≥ 20, entropy ≥ 3.5 bits/char, slang marker, no contact info) 적용 후 ~15K로 축소

**D5. CAI 비율 미반영 (MEDIUM, P1)**
- `TRAINING_RECIPE_VALIDATION.md`: CAI_RATIO=0.33 코드, 실효 6.8%
- 1,743 CAI 페어는 매우 높은 신호 (draft/critique/revision 트리플) — 도메인 톤 학습의 *콘트라스트* 역할
- 6.8% → 33%로 끌어올리려면 CAI 5,000건 추가 생성 또는 oversample factor 조정 필요

**D6. English density 누수 (LOW, P2)**
- Raw 2.2% → CPT 0.6% (-72%)
- 원인 후보: PII 마스킹에서 [URL] 토큰 치환, 또는 영문 광고 정규식이 영문 도메인 키워드 일부 제거
- 영향: 한국어 코드 스위칭 (브랜드명 카카오톡, 스포츠/패션 영문) 학습 약화
- Thunder-LLM (arXiv:2506.21595): English 1:1 mix 권장. 본 프로젝트는 *순수 한국어 도메인*이므로 1:1은 과하지만 **Wikipedia Korean + 일부 English brand-mention 약 5% 보강** 권고 (catastrophic forgetting 방지 — section 4.3)

### 3.4 권고 데이터 v3 스펙

```
cpt_corpus.v3.jsonl (target ~80K rows after MinHash dedup):
  - schema: {
      text: str,                     # NFKC normalized + Jamo restored
      kind: "post"|"comment",
      title: str | null,             # post title preserved (CPT structure fix #1)
      board: str,                    # board context (CPT structure fix #2)
      parent_text: str | null,       # parent comment if reply (CPT structure fix #3)
      depth: int,                    # 0=post, 1=root_comment, 2+=reply
      length_bucket: ...,
      ad_score: float,               # 0-1, residual ad-likelihood (for hard filter)
      jamo_density: float,           # quality signal
      char_entropy: float            # 3.5+ for SFT eligibility
    }
  - dedup: MinHash b=20 r=5 Jaccard ≥0.8 GLOBAL (not per-thread)
  - ad filter: regex + entropy < 2.8 bits/char + classifier (weak supervision)

sft_pairs.v3.jsonl (target ~15K, quality-filtered):
  - schema: {
      post_title: str,
      post_body_excerpt: str (≤256 chars),
      parent_comment: str | null,
      target_comment: str,           # response to learn
      thread_key, depth,
      task_type: "post_continuation" | "root_comment" | "reply"
    }
  - filter: target_comment length ≥ 20, entropy ≥ 3.5
  - prompt template: structured (see §4.4)

val_set.v3.jsonl (target 5% = ~4K):
  - re-split temporally to avoid train/val leakage
  - stratified by kind × topic × length_bucket

cai_pairs.v3.jsonl (target ~5K):
  - existing 1,743 + 3,000+ new generated triples
  - include in SFT mix at 25-33% ratio
```

---

## 4. 학습 레이어 분석 (Training)

### 4.1 현재 레시피 표 (정리)

| 단계 | 항목 | CPT | SFT | 출처 |
|---|---|---|---|---|
| Quant | bit / dtype | NF4 + double / bf16 | none / bf16 | train_cpt.py:187, train_sft.py:65 |
| LoRA | r / α / dropout | 64 / 64 / 0.0 | 64 / 64 / 0.0 | train_cpt.py:81-83 |
| LoRA | use_rslora / use_dora | True / False | True / False | train_cpt.py:84-85 |
| LoRA | target_modules | "all-linear" | "all-linear" | inventory §1 |
| Schedule | seq_len / batch / accum | 1024 / 1 / 16 | 1024 / 1 / 16 | recipes |
| Schedule | lr / scheduler | 2e-4 / cosine | 5e-5 / cosine | inventory §1, §2 |
| Schedule | epochs | 1 | 2 | inventory |
| Schedule | warmup / wd | 0.03 / 0.01 | 0.05 / 0.01 | inventory |
| Optim | optimizer / clip | paged_adamw_32bit / 1.0 | same | inventory |
| Mem | grad ckpt / FA2 | True / True | True / True | inventory |
| Data | template / mask | none, full causal loss | post\n + completion-only | train_sft.py:198-211 |
| Mix | ratio | — | 80% raw + 20% pair | recipes, train_sft.py:91-92 |
| Eval | save / eval / log steps | 50 / 200 / 20 | 50 / 200 / 10 | recipes |
| Repro | seed / cuBLAS / determ | 42 / pinned / on | 42 / pinned / on | inventory |

### 4.2 SOTA 비교 + 실패 모드 매핑

**Mismatch #1: LoRA-CPT가 도메인 적응에 부적합 (CRITICAL)**
- 출처: arXiv:2405.09673 (Biderman et al., May 2024) — *"LoRA Learns Less and Forgets Less"*
- 실험 결과: r=256 LoRA의 CPT는 동일 HumanEval 성능에 도달하기 위해 **풀 FT 대비 16× 데이터 비효율**
- 도메인 CPT가 요구하는 weight perturbation rank: 10-100× 일반 LoRA 설정
- 본 프로젝트와의 일치도: **CPT 1 epoch + r=64 + 92K 코퍼스 = arXiv:2405.09673이 정의한 *underfit zone* 한가운데**
- 0424-0618 run의 *CPT 46.75%까지 진행 후 SFT empty*는 LoRA가 도메인 패턴을 충분히 인코딩하지 못해 *후속 SFT가 안정 수렴할 base를 받지 못함*과 일치

**Mismatch #2: SFT 프롬프트 포맷이 thread context 미반영**
- 현재 (`train_sft.py:198-211`):
  ```
  prefix = "{post}\n"
  response = "{comment}<eos>"
  labels = [-100]*len(prefix) + tokenize(response)
  ```
- TreeTop (NeurIPS 2024) / arXiv:2603.21278 권장:
  ```
  [CTX] kind={kind} | board={board} | title={title} |
        parent_post_excerpt={post[:256]} |
        parent_comment={parent_comment} |
        depth={depth} [/CTX]
  [TGT] {target_comment}
  ```
- 차이: 현 SFT는 *원글 → 즉시 답*만 본다. 부모 댓글, board, depth 모두 손실. 따라서 댓글이 thread 어디에서든 동일한 분포로 생성됨 → 대화 구조 불일치 (G2 미달성).

**Mismatch #3: chat template 부재**
- 현재: 단순 `{post}\n{comment}` 텍스트 raw
- 영향: vLLM, llama.cpp, ollama 등 표준 추론 서버에서 chat 템플릿 자동 적용 시 학습 분포와 불일치
- **0429-0430의 11회 재시도 중 일부 실패가 이 mismatch에서 비롯됐을 가능성** (재현 어려움 신호)
- 권고: `chatml` 또는 Qwen3 native template (`<|im_start|>...<|im_end|>`) 사용

**Mismatch #4: Catastrophic forgetting 방어 없음**
- 95%+ 한국 단일 도메인 CPT → arXiv:2405.09673이 명시한 catastrophic forgetting risk zone
- StarCoder-Python 사례와 동일 패턴
- Thunder-LLM (arXiv:2506.21595): 1:1 English mix 권장
- 본 프로젝트: 5-15% Wikipedia-Ko + 일반 web KR 5% 혼합으로 충분

### 4.3 학습 권고 v3

**옵션 A — Conservative ($30 budget 유지)**:
1. **Base 변경**: `Qwen/Qwen3-8B-Base` → `MLP-KTLim/llama-3-Korean-Bllossom-8B` 또는 `beomi/Llama-3-Open-Ko-8B`
2. **CPT 생략, SFT 직진**: Bllossom는 이미 한국어 + DPO 완료 → 도메인 적응만 SFT로 학습
3. SFT recipe:
   - r=128, alpha=128, rsLoRA=True, **DoRA=True** (arXiv:2402.09353 +2-5% gain at <0.1% extra params)
   - lr=1e-4 (TRAINING_RECIPE_VALIDATION.md 권고와 일치, arXiv:2602.04998 검증)
   - epochs=3 (quality-filtered 15K 데이터에서 over-fit 위험 적음)
   - completion-only mask (현재 유지)
   - chat template = `chatml`
   - thread-aware prompt (§4.4)
   - **Unsloth 백엔드** → L40S에서 2-5× throughput, 비용 50-80% 감소 → $30 budget로 3-cycle 가능

**옵션 B — Quality-First (예산 확장 시)**:
1. **CPT recipe upgrade**:
   - r=256, alpha=256, rsLoRA + DoRA 동시 enable
   - 또는 fp16 풀 FT (L40S 48GB로 Qwen3-8B + grad ckpt + flash-attn 가능; A100 권장)
   - epochs=2, lr=1e-4 (cosine), seq_len=2048 (긴 thread 컨텍스트 수용)
   - corpus mix: dalbitalba 80% + Korean Wikipedia 15% + English snippets 5%
2. **SFT (옵션 A와 동일 + DPO)**:
   - SFT 후 DPO with CAI revision pairs (chosen=revision, rejected=draft)
   - SimPO (EXAONE 3.5 패턴) 추가 가능

**옵션 C — Recipe Mutation Refactor**:
- 현 `recipe_mutator.py` R1: bigram_jsd > 0.15 → epochs +1 → r 64→128 → DoRA on
- 진단: R1은 LoRA-CPT가 *underfit*인 상황에서 *correct lever* (LoRA → full FT 전환)을 트리거하지 않음
- **R7 신규 (긴급)**: `if cycles_with_jsd>0.15 ≥ 2 and r ≥ 128`: switch to **fp16 full FT** (또는 r=256)
- **R8 신규**: `if same recipe + same data → JSD delta < 0.01`: stop, recommend data regen (dedup + ad cleanup)

### 4.4 권고 SFT 프롬프트 포맷 (thread-aware)

```
<|im_start|>system
당신은 한국 유흥업 종사자 커뮤니티의 일반 사용자입니다. 짧고 솔직하며 초성·은어를 자연스럽게 씁니다.<|im_end|>
<|im_start|>user
[게시판] {board}
[유형] {kind}
[제목] {title}
[원글] {post_body_excerpt}
{# parent_comment 있으면 추가 #}
[부모댓글] {parent_comment}
[depth] {depth}

위에 어울리는 답글을 1개 작성하세요.<|im_end|>
<|im_start|>assistant
{target_comment}<|im_end|>
```

Loss masking: assistant 응답만 학습. system + user는 -100. (현 train_sft.py:198-211과 동일 메커니즘, 컨텍스트 풍부화).

---

## 5. 평가 레이어 분석 (Evaluation)

### 5.1 현재 평가 자산

**Phase 6 deterministic gate** (`scripts/phase6_eval.py`):
| 메트릭 | 임계 | 측정 방법 | 한계 |
|---|---|---|---|
| char bigram_jsd | ≤ 0.15 (stretch 0.08) | char-level n-gram 분포 JSD | 표면적; 의미 미반영 |
| length_kl | ≤ 0.10 | 13-bin log-spaced length histogram KL | 짧은 댓글 비중에 dominated |
| digit_density Δ | ≤ 0.03 | mean abs delta | PII mask 후 raw shift |
| english_density Δ | ≤ 0.02 | mean abs delta | 브랜드명 영향 |
| MAUVE | ≥ 0.80 (optional) | klue/roberta-large featurizer | 옵션 (필수 아님), <1000 samples 불안정 |

**3-Judge eval** (`eval/judge_3way.py`):
- Heuristic (formal +2, slang -2, len-based)
- Anthropic Claude (model env 가능, JSON 파싱)
- OpenAI GPT (동일 패턴)

**Calibration baseline** (`raw-vs-raw.json`):
- bigram_jsd ceiling: **0.0190**
- recommended primary: 0.08, stretch: 0.05
- length_kl baseline: 0.0004

**Bench v3-real-20260421** (이전 산출물):
- n=50 (25 human + 25 AI)
- claude-opus-4-7: **acc=0.50, FN=1.00 (모든 ai를 human으로)**
- claude-sonnet-4-6: **acc=0.50, FN=1.00 (동일)**
- gpt-5: acc=0.88, FN=0.24

### 5.2 평가 결함 진단

**E1. Claude judge 도메인 편향 (CRITICAL, P0)**
- bench/v3-real에서 Claude Opus/Sonnet 두 모델 모두 *모든 AI 샘플을 human으로 분류*
- 해석: Claude는 한국 커뮤니티 도메인의 슬랭/초성/짧은 댓글을 *AI가 생성할 수 없는 패턴*으로 강하게 prior 형성
- 영향: 3-judge 합의 시스템에서 Claude 두 노드 사실상 무용 → *효율적 1.5-judge 시스템*
- **권고**: judge_3way를 (1) GPT-5 + (2) 도메인 fine-tuned 스타일 분류기 (DeBERTa v3 small + dalbitalba speech vs formal Korean) + (3) 휴리스틱으로 재구성

**E2. 의미적 / 스레드 일관성 메트릭 부재 (HIGH, P1)**
- 현 5-metric은 *분포 매칭*만 — 게시글 의미와 댓글이 *내용적으로 호응*하는지 미평가
- 권고 추가:
  - **Thread coherence**: `klue/roberta-base` 임베딩으로 (parent_post, generated_reply) cosine similarity vs (parent_post, actual_human_reply) cosine
  - **BERTScore**: 동일 prompt에서 다수 인간 답글들과 BERTScore F1 평균
  - **Style classifier AUC**: dalbitalba speech vs formal Korean DeBERTa 이진 분류기, AUC < 0.65 = 구분 불가

**E3. 한국어 일반 능력 회귀 미검사 (HIGH, P1)**
- CPT/SFT 후 KoBEST/HAE-RAE/KMMLU 점수 변화 미측정
- catastrophic forgetting 발생 시 감지 불가
- 권고: `HRET` (HAE-RAE Evaluation Toolkit, arXiv:2503.22968) GHA에 통합
  - `pip install haerae-eval && hret evaluate --model_path ./checkpoint --tasks haerae_bench,kobest`
  - Δ KoBEST > -2pt → 회귀 game

**E4. MAUVE 옵션 처리의 silent failure (MEDIUM, P1)**
- `phase6_eval.py:140-161`: MAUVE 라이브러리 없으면 SKIP
- gate에 ‘mauve_score’ 없을 때 PASS로 간주됨 (silent gap)
- 권고: MAUVE 필수화 또는 명시적 Vendi Score fallback

**E5. 생성 비결정론 (MEDIUM, P1)**
- `phase6_generate.py`: do_sample=True, temperature=1.1, no `set_seed`
- 동일 어댑터에서 매 eval마다 다른 생성 → 5-metric variance ≈ 0.01-0.02
- 권고: `transformers.set_seed(42)` + `model.generate(seed=...)` (지원 시) 추가

**E6. 평가 샘플 수 부족 (MEDIUM, P2)**
- bench는 n=50 (25 vs 25)
- MAUVE 안정 추정에 ≥1000 샘플 필요 (arXiv:2102.01454)
- val_set 2,451 활용 시 가능. `phase6_generate.py`의 EVAL_MAX_ROWS 기본 500 → **1000+** 권고

### 5.3 권고 평가 v3 스펙

**Tier 1 — Stage gate (CI 자동, GHA, GPU 1× pass)**
- phase6 5-metric (현행 유지) + MAUVE 필수화 + 생성 결정론
- HRET KoBEST + HAE-RAE + KMMLU 회귀 (Δ ≤ -2pt fail)
- Korean style classifier AUC < 0.65 (DeBERTa-v3-small)

**Tier 2 — Thread coherence (GPU 0.5× pass)**
- 생성된 reply 1,000개 + ground-truth reply pair
- klue/roberta-base 임베딩으로 (post, reply) cos sim 분포 비교
- BERTScore F1 vs 인간 답글 다수 평균

**Tier 3 — LLM judge (cost: GPT-5 API only)**
- GPT-5 acc ≥ 0.85 from prior bench (Claude는 제거)
- 새 prompt: rubric 기반 (5-point likert × 3 axis: stylistic, structural, factuality)

**Tier 4 — Manual audit (sampled)**
- 50 샘플 random + 50 worst-case (낮은 thread coherence 샘플)
- Korean reviewer + GPT-5 cross-check
- 결과 → 학습 데이터 cycle feedback

---

## 6. 운영 레이어 분석 (MLOps / Infra)

### 6.1 현재 인프라 강점

| 자산 | 효용 |
|---|---|
| chain_train.sh 5-stage + timeout | 안정적 stage 격리, 부분 실패 격리 |
| graceful_abort + ntfy | TERM/INT/HUP 시 artifact 보존 + 알림 |
| persist_run_artifacts() git push | 매 cycle 별 train-run-* branch 자동 생성 |
| recipe_mutator + autonomous_loop | 자동 cycle, budget cap, convergence detection |
| local_verification_loop | pre-pod compile / encoding / PII / minor / sexual gate |
| budget30/smoke recipes | $30 ceiling 운영, 빠른 plumbing 검증 |
| Korean encoding profile | hangul/jamo/digit/eng 측정 (mojibake 감지) |

이는 **현 시점 한국어 LLM fine-tuning 오픈소스 생태계 중 운영 자동화에서 상위 10%** 수준이다.

### 6.2 운영 결함

**O1. Unsloth/LLaMA-Factory 미사용 (HIGH, P2)**
- 현재 raw HF Trainer + PEFT
- Unsloth (Triton kernel): L40S에서 Qwen3 학습 2-5× throughput, 80% VRAM 절감
- 권고: SFT 단계만 Unsloth로 전환 → $30 budget 내 3-cycle 가능

**O2. 단일 GPU (HIGH, P2)**
- batch=1, grad_accum=16
- A100 80G 또는 2× L40S DDP 시 cycle 시간 50% 단축
- 다만 Unsloth 적용으로 충당 가능

**O3. accelerate 버전 락 → data_seed 미사용 (MEDIUM, P1)**
- `train_cpt.py:305-308` 코멘트: trl 0.12.1 + accelerate 0.34.2 pinned, transformers 4.51.3는 accelerate≥1.1 요구
- 결과: data_seed 미설정, 데이터 샘플러 결정론 일관성 미보장
- 권고: trl 최신화 (0.13+) + accelerate 1.x 업그레이드 → 단일 dependency upgrade

**O4. local_verification_loop가 pipeline에 미통합 (LOW, P2)**
- 별도 standalone 도구
- 권고: chain_train.sh stage 0 (preflight)에 호출 추가 → severe finding > 0이면 abort

**O5. data git versioning (LOW, P3)**
- JSONL 93K rows 직접 commit
- 권고: git-lfs 또는 dvc 도입

### 6.3 권고 v3 인프라

```
chain_train.v3.sh:
  Stage 0a: scripts/local_verification_loop.py --strict (block on severe)
  Stage 0b: data/v3 freshness check (regenerate if older than corpus mtime)
  Stage 1: CPT (옵션 A 생략, 옵션 B만 적용)
  Stage 2: merge (옵션 B시)
  Stage 3: SFT (Unsloth + DoRA + thread-aware prompt)
  Stage 3.5: DPO (CAI chosen/rejected)
  Stage 4: phase6 + HRET 회귀
  Stage 5: HF upload
  Stage 6: PR auto-create (single dispatch)

GHA:
  - .github/workflows/data-regen.yml (manual + cron)
    - phase1_data_pipeline.v3 + dedup + ad filter + jamo restore
  - .github/workflows/runpod-train.yml (현행 유지)
  - .github/workflows/runpod-eval.yml (HRET + thread coherence 추가)
  - .github/workflows/local-gate.yml (PR 시 local_verification_loop 자동)
```

---

## 7. 산출물 레이어 (Artifact Inventory)

### 7.1 HF Hub 현황 (2026-05-07 시점)

**삭제됨 (지난 audit 시점 존재했으나 현재 404)**:
- `UNOA/dalbitalba-qwen3-cpt-20260424-0618` ← `HANDOFF_CODEX.md` 기준 핵심
- `UNOA/dalbitalba-qwen3-sft-20260424-0618` ← 동일

**존재 중 (UNOA 네임스페이스, 2026-04-29 ~ 2026-04-30)**: 11개 페어
- `*-cpt-20260429-0809, 0817, 0849, 1523, 1700, 1830, 2300`
- `*-cpt-20260430-0000, 0030, 0130`
- 각각 sft 페어 동반
- **최신**: `UNOA/dalbitalba-qwen3-sft-20260430-0130` (safetensors 태그 — 어댑터 존재)

**해석**:
- 11회 재시도 = mutation rule 또는 environmental issue로 인해 안정 수렴 미달성
- 0424-0618 audit doc에서 진단된 *CPT 46.75% / SFT empty* 패턴이 0429-0430 cycle에서도 (적어도 일부) 반복된 것으로 추정 (downloads=0, likes=0 모두)
- Latest 0130 SFT가 safetensors 보유 = adapter 페이로드는 존재. 그러나 평가 결과 미공개 / private

**권고**: 0430-0130 어댑터를 phase6 + HRET로 평가하여 *현재 모델 능력 baseline*을 fix. 그 위에 본 분석의 v3 데이터 + recipe로 실험 비교.

### 7.2 GitHub 산출물

**bench/** — 3개 snapshot (v2 dryrun, v3, v3-real)
- v3-real-20260421: AI vs human 실제 측정 (위 §5.1)
- 결과: GPT-5 88%, Claude 50% (judge 결함)
- AI 샘플 25개 모두 misclassified by Claude → 향후 비교 시 *Claude 단일 의존 회피*

**turing-test/2026-04-20T04-26-42** — 10-question quiz (모두 human)
- 의도: 50% 추론 baseline 정의
- *현재 비활용 상태*

**runs/** — git-persisted train-run-* / eval-run-* branches (인벤토리상 표시, 클로닝 후 미관측 → 별도 브랜치에 존재)

**research/obsidian-export** — 9개 카테고리별 게시글 export (data-body-image, data-dating-advice, data-fashion 등)
- 의도: 수동 큐레이션/태깅 보조
- *학습 입력 아님 (README 명시)*. 향후 quality-curated SFT 후보 풀로 활용 권고

### 7.3 Calibration / planning

`raw-vs-raw.json` — side_a vs side_b raw 비교
- 모든 5 metric의 이론적 ceiling 정의
- bigram_jsd ceiling **0.0190** → primary target 0.08 (4.2× ceiling), stretch 0.05 (2.6×)
- **이 정의는 학술적으로 견고함** (반사실적 비교 + multiple of intrinsic variance)

`phase1_summary.json` — 데이터 빌드 metadata
- timestamp 2026-04-24
- dedup_dropped=19 (CRITICAL: §3.3 D1과 일치)
- mix_policy: raw 0.8 / pair 0.2

---

## 8. 우선순위 개선 로드맵 (Implementation Plan)

### P0 — 즉시 (GPU 비용 0, 1-2일 작업)

**P0-1. MinHash global dedup**
- 도구: `text-dedup` (github.com/ChenghaoMou/text-dedup) 또는 datasketch MinHashLSH 직접
- 파라미터: b=20, r=5, Jaccard ≥0.8 GLOBAL (post + comment 통합)
- 예상 결과: 92K → ~50-60K rows, dup rate 0.4 → <0.05
- 통합 위치: `scripts/phase1_data_pipeline.py` 후단, `cpt_corpus.v3.jsonl` 생성 직전
- 측정: phase1_summary.json의 dedup.dropped_near_dup → 신규 build에서 30,000+ 예상

**P0-2. Ad/operator filter 강화**
- 신규 패턴 추가:
  ```python
  AD_PATTERNS_V3 = AD_PATTERNS + [
      r'(?:저희|우리)\s*(?:가게|업소|샾)',  # operator self-promo
      r'출근시?\s*연락',                    # recruitment
      r'언제든\s*편하게\s*(?:연락|문의)',    # template
      r'(?:VIP|vip)\s*[가-힣]{1,5}',          # VIP service
      r'라인\s*(?:ID|아이디)?\s*[:\s][a-zA-Z0-9_]+',
      r'시간\s*(?:당|시)?\s*\d+\s*만원',     # pay quote
  ]
  ```
- + character n-gram entropy gate: `< 2.8 bits/char → spam` (FineWeb 패턴)
- + classifier weak-supervision (선택): 100 manual labeled spam → fastText linear classifier
- 예상 결과: ad keyword 보유 레코드 21.6% → <5%
- 측정: §3.2 동일 profiling 재실행

**P0-3. NFKC + Compatibility Jamo 원자적 정규화**
- `phase1_data_pipeline.py:88` 직후에 fix_jamo_normalization 인라인:
  ```python
  text = unicodedata.normalize("NFKC", text)
  text = restore_compatibility_jamo(text)  # from fix_jamo_normalization.py
  assert is_consistent_jamo(text)  # invariant check
  ```
- `scripts/fix_jamo_normalization.py`를 `lib/text_normalize.py`로 재구성 → phase1에서 import
- 측정: jamo density raw 2.0% vs CPT v3 차이 < 0.5%

**P0-4. Qwen3 Korean fertility 측정**
- `scripts/measure_qwen3_korean_fertility.py` 신규 (§2.2 코드)
- 결과 보고서: `.planning/calibration/qwen3_korean_fertility.json`
- chars/token < 2.0이면 vocab 확장 또는 base 모델 변경 검토 트리거

**P0-5. Recipe mutator R7/R8 신규**
- R7 (architecture switch): 2 cycles JSD>0.15 + r≥128이면 fp16 full FT 또는 r=256 권고 정지
- R8 (data regen): same recipe + JSD delta < 0.01 → data regen 명령 emit, training relaunch 보류

### P1 — 단기 (GPU 1-2 cycle, 1주일)

**P1-1. Thread-aware SFT 데이터 v3 생성**
- `scripts/build_thread_aware_datasets.py` 확장 (이미 존재)
- 새 schema (§3.4) 적용
- prompt template (§4.4) 적용
- 결과: sft_pairs.v3.jsonl ~15K (quality-filtered)

**P1-2. CAI 비율 정상화**
- 1,743 → 5,000+로 확장 (Anthropic API 또는 GPT-5로 추가 triple 생성)
- SFT 믹스에서 25-33%로 끌어올림
- 또는 SFT 다음 stage로 DPO (chosen=revision, rejected=draft) 추가

**P1-3. SFT 평가 baseline 측정**
- 0430-0130 어댑터를 phase6 + HRET + thread coherence 평가
- *현재 모델 능력*을 numerical fix
- 이후 v3 실험과 비교

**P1-4. Korean style classifier 학습**
- DeBERTa-v3-small (HF: microsoft/deberta-v3-small) → 한국어 호환 위해 klue/bert-base 또는 klue/roberta-base 권고
- 데이터: dalbitalba speech (raw crawl 5K) + 일반 한국어 web (AIHub 또는 Korean wiki 5K)
- 학습: 1-2시간 GPU
- 산출: AUC 측정용 binary classifier; phase6에 metric 추가

### P2 — 중기 (GPU 3-5 cycle, 2-3주)

**P2-1. Unsloth 백엔드 SFT**
- `train_sft.py`를 Unsloth FastLanguageModel로 wrap
- 또는 LLaMA-Factory YAML 기반 마이그레이션 (Unsloth + Liger Kernel 자동 적용)
- 효과: L40S 단일 GPU에서 throughput 2-5× → 단일 cycle 비용 $5-7로 감소

**P2-2. CPT recipe upgrade**
- 옵션 A (저비용): Bllossom-8B 또는 Llama-3-Open-Ko-8B로 base 변경, CPT 생략, SFT 직진
- 옵션 B (고품질): r=256 rsLoRA + DoRA, lr=1e-4, seq_len=2048, English 5% 혼합

**P2-3. HRET / lm-eval-harness 통합**
- GHA `runpod-eval.yml`에 HRET stage 추가
- KoBEST regression gate (Δ > -2pt → FAIL)

**P2-4. 3-judge 재구성**
- Claude judge 제거 (도메인 편향 확정)
- 신규 구성: GPT-5 + DeBERTa style classifier + 휴리스틱
- judge prompt rubric 기반 (stylistic, structural, factuality 5-point Likert)

### P3 — 장기 (1개월+)

**P3-1. DPO/SimPO post-SFT (EXAONE 3.5 패턴)**
- SFT → DPO with CAI pairs → SimPO with judge feedback
- arXiv:2412.04862 sequential preference optimization

**P3-2. Vocab 확장 (필요 시)**
- P0-4 fertility 측정 결과 < 2.0이면 SentencePiece merger로 한국어 5K 추가
- LongLoRA (ICLR 2024) 패턴: embedding + norm 동시 학습

**P3-3. TreeTop 17-task auxiliary CPT (NeurIPS 2024)**
- parent identification, sibling ranking, depth prediction 등 구조 인식 task
- thread 구조 학습 강화 → G2 structural 목표에 가장 직접적

**P3-4. Production inference stack**
- vLLM + qwen3 chat template + LoRA hot-swap
- 추론 latency 50ms 미만 / 단일 L4 GPU 운영

---

## 9. 측정 가능한 KPI (Before / After)

| KPI | 현재 (실측 또는 추정) | P0+P1 후 목표 | P2+P3 후 목표 |
|---|---|---|---|
| CPT corpus dup rate | ~0.40 | ≤ 0.05 | ≤ 0.02 |
| Ad keyword 보유 % | 21.6% | ≤ 5% | ≤ 2% |
| Jamo NFKC 일관성 | 부분 (3.06% 잔여 isolated jamo) | 100% (원자 보존) | 100% |
| SFT 짧은 댓글 비율 (<20 char) | 30.4% | ≤ 10% (quality filter) | ≤ 5% |
| CAI 실효 비율 | 6.8% | 25% | 33% (DPO 합산) |
| bigram_jsd | unknown (0618 SFT empty) | ≤ 0.10 | ≤ 0.06 (stretch) |
| length_kl | unknown | ≤ 0.08 | ≤ 0.05 |
| KoBEST regression | unmeasured | -2pt 이내 | +2pt 이상 |
| GPT-5 judge accuracy (낮을수록 좋음) | 0.88 (구분 잘됨) | ≤ 0.70 | ≤ 0.55 |
| Style classifier AUC (낮을수록 좋음) | unmeasured | ≤ 0.75 | ≤ 0.60 |
| Single cycle cost (USD) | $7-12 (L40S, vanilla HF) | $4-6 (Unsloth) | $3-5 (DoRA + Unsloth) |
| Single cycle wall time (h) | 12-32h | 6-16h | 3-8h |

---

## 10. 한 줄 요약

**0618 run의 실패는 인프라가 아니라 *학습 설계와 데이터 정제* 두 layer에서 동시에 발생했다. P0의 4가지 GPU-free 작업 (MinHash global dedup, ad regex 강화, NFKC + Jamo atomic normalize, Qwen3 fertility 측정)만으로도 다음 cycle의 *SFT 도달 확률*과 *bigram_jsd 수렴*이 정성적으로 다른 단계로 이동할 것이며, P1의 thread-aware SFT 포맷이 G2 structural 목표 달성의 *유일한 정공법*이다.**

---

## 부록 A — 외부 SOTA 인용 요약 (must-read)

| 출처 | 핵심 발견 | 본 프로젝트 적용 |
|---|---|---|
| arXiv:2405.09673 | LoRA CPT는 풀 FT 대비 16× 데이터 비효율 | CPT를 LoRA로 진행하지 말 것 (P2-2) |
| arXiv:2506.21595 (Thunder-LLM) | 한국어 8B에 1:1 EN 혼합, 5-gram dedup, perplexity filter | English 5-15% 혼합 (P2-2) |
| NeurIPS 2024 (TreeTop) | 17-task thread structure auxiliary objectives | thread-aware SFT 포맷 (P1-1) |
| arXiv:2412.04862 (EXAONE 3.5) | DPO + SimPO sequential | SFT → DPO (P3-1) |
| arXiv:2506.15138 (Thunder-Tok) | Hangul Jamo malformed substring 문제 | NFKC atomic + 초성 보존 (P0-3) |
| arXiv:2408.11294 (RedWhale) | 4-stage 학습 + 20K vocab 확장 | 옵션 A의 base 변경 검토 |
| arXiv:2503.22968 (HRET) | 한국어 LLM eval toolkit 통합 | KoBEST 회귀 게이트 (P2-3) |
| arXiv:2406.17557 (FineWeb) | 큰 cluster dedup이 가장 큰 perplexity 이득 | MinHash global dedup (P0-1) |
| arXiv:2402.09353 (DoRA) | <0.1% extra param, +2-5% gain | DoRA enable (P1-1) |
| arXiv:2411.15124 (Tulu-3) | Quality > quantity SFT mix | 15K quality SFT (P1-1) |

## 부록 B — 본 분석에서 *직접 측정한* 데이터 (재현 가능)

```
CPT v2 (n=46,856):
  avg_len=60.3, hangul=0.634, jamo=0.0306, digit=0.0460, eng=0.0060
  slang_marker_records=12,332 (26.3%), ad_keyword_records=10,142 (21.6%)
  kinds: comment 36,386 / post 10,470
  buckets: xs 7,351 / sm 22,022 / md 11,285 / lg 4,120 / xl 1,993 / xxl 85

SFT v2 (n=44,474):
  post p50=64 mean=100 max=1804
  comment p50=30 mean=48 max=1114
  comments<20chars: 13,523 (30.4%)
  slang in post: 21.3%, in comment: 17.9%
  buckets: xs 13,523 / sm 18,685 / md 8,413 / lg 2,337 / xl 1,495 / xxl 21

val v2 (n=2,451):
  post 560 (22.8%) / comment 1,891 (77.2%)

Calibration baseline (raw-vs-raw):
  bigram_jsd=0.0190 (ceiling), unigram_jsd=0.0006, length_kl_sym=0.000407
  recommended primary: bigram 0.08, unigram 0.04, length 0.10
  recommended stretch: bigram 0.05, unigram 0.02, length 0.05
```

---

**End of Analysis** — 본 문서는 코드, 데이터, 외부 문헌의 *증거-기반* 진단이며, 모든 권고 항목은 ROI 측정 가능한 KPI와 연결되어 있다. 다음 단계는 P0의 4개 항목 *동시 PR* 권고 (GPU 비용 0, 코드 변경량 ~300 LOC).
