# tokenizer_v4 (Qwen3 + 호빠 도메인 어휘 확장)

**Generated**: 2026-05-04
**Base**: Qwen/Qwen3-8B-Base
**Method**: Kiwi 형태소 분석 → 명사/외래어/한자 (NNG/NNP/SL/SH/NR) 토큰 추출
            → freq>=20 필터 → BPE 분해 후보만 → 점수순 top 200 + 구조 마커 +
            CORE_DOMAIN_TERMS 강제 포함

## 결과

| | 이전 (Qwen3 base) | tokenizer_v4 |
|---|---|---|
| vocab size | 151,669 | **151,908** (+239) |
| 쩜오 | 2 tokens | **1** |
| 텐카 | 2 | **1** |
| 밀빵 | 2 | **1** |
| 호빠 | 2 | **1** |
| 마담 | 2 | **1** |
| 강남역 | 3 | **1** |
| 논현동 | 3 | **1** |
| **가라오케** | **4** | **1** |
| 엘리트 | 3 | **1** |
| 퍼블릭 | 3 | **1** |
| 스트레스 | 3 | **1** |
| 아가씨 | 3 | **1** |

## 한계 / 주의

- 신규 토큰의 임베딩은 **랜덤 초기화 상태** — Embedding warmup CPT 1 epoch
  필요 (V4 Recipe Stage 2 Phase A)
- Kiwi 사전이 일부 도메인 신어 (밀빵/풀묶/빠꾸/TC 등)을 알지 못해 freq 0 →
  CORE_DOMAIN_TERMS 강제 포함으로 보충
- "강하늘" 같은 도메인 외 고유명사도 일부 포함됨 (수동 큐레이션 시 제거 가능)
- **사용 전 임베딩 warmup 필수** — 그렇지 않으면 신규 토큰은 무의미한
  representation을 갖고 있어 모델 출력 악화 가능

## 사용법

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("./tokenizer_v4")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")
model.resize_token_embeddings(len(tok))   # 새 임베딩 슬롯 생성
# 그 다음 CPT로 신규 임베딩 학습 — 최소 1 epoch
```

## 재현

```bash
# Stage 0+1 audit + tokenizer 빌드 전체 재실행
source .venv-local/bin/activate
python scripts/audit/build_vocab_kiwi.py
python scripts/extend_tokenizer_v4.py \
  --vocab-candidates-top runs/audit/vocab_kiwi.json \
  --top-n 200 \
  --out-dir tokenizer_v4/
```

## 관련 문서

- `research/RECIPE_DESIGN_V4_MULTIDIM.md` — V4 8-stage 설계
- `research/STAGE1_AUDIT_FINDINGS.md` — Stage 0+1 실행 결과 + 한계 분석
- `research/ROUND_N_EVIDENCE_AUDIT.md` — evidence-based 기법 미적용 audit
