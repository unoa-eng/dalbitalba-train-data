# V4 Stage 0+1 Audit Findings (2026-05-04)

## 실행 결과 요약

`scripts/audit/run_all.sh` 1회 실행 (mac-mini, .venv-local).

| 산출물 | 규모 | 상태 |
|---|---|---|
| persona_index.jsonl | 11,515 authors | ✅ veteran 4명만 (>=50 docs) |
| dimension_baseline.json | 42,473 docs | ✅ |
| topic_labels.jsonl | 42,473 rows | 🟡 "기타" 54.6% (heuristic 한계) |
| tone_labels.jsonl | 42,473 rows | ✅ 4-class 분포 합리적 |
| vocab_candidates.jsonl | 4,546 ranked | ⚠️ **noise heavy** (아래 §1) |
| vocab_candidates_top.json | top 500 | ⚠️ **수동 큐레이션 필수** |

## §1. Vocab candidates 노이즈 — Stage 1 핵심 발견

**현상**: 상위 후보가 의미 있는 단어가 아닌 sentence fragment.

**예시 (top 10)**:
```
연락주세여    freq=262 bpe=5 score=66.70
마담눈치보    freq=258 bpe=5 score=66.68
스받지말고    freq=257 bpe=5 score=66.64
담눈치보느    freq=257 bpe=5 score=66.64
눈치보느라    freq=257 bpe=5 score=66.64
레스받지말    freq=257 bpe=5 score=66.64
게연락주세    freq=260 bpe=5 score=66.60
수위때문에    freq=259 bpe=5 score=66.56
트레스받지    freq=258 bpe=5 score=66.51
저희는어떠    freq=164 bpe=5 score=61.27
```

**원인 진단**:
1. 코퍼스에 띄어쓰기 거의 없음 (구어체 + 디시 스타일)
2. 알고리즘이 char n-gram 단순 추출 — 단어 경계 정보 없음
3. `스트레스받지말고` → `스받지말고`, `트레스받지`, `레스받지말`, ... 다중
   substring이 동일 빈도로 등장 → 모두 상위 진입

**필요한 후처리**:
- A. **Substring dedup**: 같은 frequency를 보이는 substring set 중 가장 긴
     것만 유지
- B. **Word-boundary detection**: KoNLPy / Mecab / KSS 등 한국어 tokenizer로
     형태소 분해 후 합성어만 후보화
- C. **Manual curation**: top 500을 사람이 검토해서 진짜 도메인 어휘
     200-300개 선별

**현재 상태**: 자동화된 top 500은 **그대로 토큰 추가에 사용 불가**.
`scripts/extend_tokenizer_v4.py`는 작성 완료 + 실행 가능하지만, 노이즈 후보를
그대로 추가하면 오히려 토큰 효율 악화 가능.

## §2. Persona pool 작음

**관측**: 11,515 저자 중 ≥50 docs 작성 저자가 **4명**.

**의미**:
- 페르소나-conditioned SFT (V4 Stage 5/6)의 ground truth가 매우 제한적
- 한 저자가 다양한 톤을 보일 수 있으므로 단순 source_id 기반 페르소나는
  취약. **글 스타일 임베딩 + 클러스터링** 필요할 수 있음
- 또는 베테랑 임계값 완화 (>=20 docs로 낮추면 ?명 확장 가능 — 향후 재분석)

## §3. Topic 라벨 "기타" 54.6%

**의미**: 키워드 기반 heuristic의 한계. 실제 토픽이 다양해서 단순 정규식
패턴으로는 분류 못 하는 글이 절반 이상.

**개선 옵션**:
- KoBigBird / klue-RoBERTa 등 한국어 토픽 분류 모델 fine-tune
- LDA / BERTopic 비지도 클러스터링 (10-30 토픽 자동 발견)
- 본격적 V4 Stage 4 (Topic Coverage)에서는 위 둘 중 하나 채택 필수

## §4. Stage 1 (Tokenizer Extension) 실제 진행 여부

**작성**: `scripts/extend_tokenizer_v4.py` 완성, syntax/구조 검증
**실행**: 미실행. **이유**: 자동 후보가 노이즈 heavy → 검증된 어휘 list 없이
실행하면 잘못된 토큰 임베딩 학습 위험.

**다음 단계 권장**:
1. 사람 검토 또는 KoNLPy 후처리로 vocab_candidates_top.json 정제
2. `scripts/extend_tokenizer_v4.py --top-n 200 --out-dir tokenizer_v4/` 실행
3. 신규 토큰의 임베딩 warmup CPT 1 epoch (RunPod, ~$2-5)
4. 결과 토큰화 효율 측정 → before/after 비교

## §5. 작업 흐름 다음 트랙 (Stage 2 진입 조건)

| 조건 | 현재 |
|---|---|
| Stage 0 audit 완료 | ✅ (heuristic 한계 명시) |
| vocab_candidates_top 검증 | ❌ (수동/형태소 후처리 미수행) |
| 토크나이저 v4 실제 생성 | ❌ (검증된 어휘 list 대기) |
| Embedding warmup CPT | ❌ (Stage 2-Phase A 진입) |

## 비용/시간 정산

- 본 audit: 무료 (mac-mini 로컬, 약 5분)
- 다음 단계 (수동 큐레이션 + warmup CPT): ~$2-5 + 1-2일

## 결론

**Stage 0 + Stage 1 자동화 부분은 완료**. 그러나 Stage 1을 실제로 적용
가능한 형태로 만들려면 **vocab 후처리 단계가 추가 필요**. 본 audit이 그
한계를 명시적으로 노출시킴 — 다음 라운드 의사결정에 이 한계를 반영해야 함.
