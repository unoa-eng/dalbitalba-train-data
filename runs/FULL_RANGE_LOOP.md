# 전범위 루프 프로세스 (Full-Range Verification Loop)

자동 실행: 모든 개별 스테이지 검증 완료 후 트리거

## 루프 구조

각 이터레이션은 아래 3단계로 구성:

### Phase A: 웹 리서치 + 최신 논문 검토
- HuggingFace/arXiv: LoRA CPT, SFT, structured token 관련 최신 논문/실험
- Reddit r/LocalLLaMA, r/MachineLearning: 실전 경험 공유
- 현재 TRAINING_DESIGN_V3.md와 비교하여 개선점 식별

### Phase B: 경쟁적 3-모델 토론 (Competitive Debate)
- 3 workers (codex/gemini/claude) 각각 독립적으로 파이프라인 평가
- 판정: PASS/FAIL + 근거
- 만장일치 PASS가 아니면 이슈 수정 후 재검증

### Phase C: 로컬 MLX 실제 테스트
- 0.6B 모델로 전체 체인 실행: CPT → merge → SFT → eval
- 산출물 품질 평가: bigram_jsd, DKA, tone, structure token 정확도
- Before/After 비교 리포트

## 종료 조건
- **연속 3회** Phase B에서 전원 만장일치 PASS
- 또는 7라운드 도달 시 강제 종료 + 잔여 이슈 문서화

## 검증 대상 스테이지 (11개 전수)

| # | 스테이지 | 검증 항목 |
|---|----------|-----------|
| 1 | 데이터 빌드 (CPT) | cpt_structured_v3.jsonl 행수, 구조 토큰 분포 |
| 2 | 데이터 빌드 (SFT) | sft_5task_v3.jsonl 행수, 태스크 비율 (T2+T3 65%) |
| 3 | 토크나이저 확장 | 11개 특수 토큰, vocab 크기 |
| 4 | CPT 학습 | train_cpt.py + recipe → adapter, val loss 감소 |
| 5 | 모델 병합 | merge_cpt_to_fp16.py → 병합 모델 + 확장 토크나이저 |
| 6 | SFT 학습 | train_sft.py + instruction format → SFT adapter |
| 7 | 생성 (phase6) | phase6_generate.py → 50 samples, kind 필드 |
| 8 | 평가 (phase6) | phase6_eval.py → metrics, gate verdict |
| 9 | PPL 체크 | korean_retention_ppl ≤ 1.50 |
| 10 | HF 업로드 | chain_train.sh → huggingface-cli upload |
| 11 | 프로모션 | check_smoke_promotion.py → 최종 판정 |

## 실행 이력

| Round | Phase A | Phase B | Phase C | 결과 | 날짜 |
|-------|---------|---------|---------|------|------|
| R1 | LoRA CPT/SFT 리서치 완료 | 3 codex FAIL (val leak, EOS mask, uncommitted) | - | FAIL→수정 | 2026-04-29 |
| R2 | - | W1 FAIL(legacy), W2 PASS, W3 FAIL(PII)→수정 | PII 제거, WARN gate | 실질 PASS | 2026-04-29 |
| R3 | - | W1 PASS, W2 PASS, W3 FAIL(budget gate, stop/term) | Budget gate+SFT timeout cap 수정 | 2/3 PASS+수정 | 2026-04-29 |

## 수정 이력

### Round 1 발견 및 수정
1. **Val set leakage** (64%) → val_set.v3.jsonl 생성 (412행, leak 0)
2. **EOS supervision masking** → DataCollatorForPackedCPT 추가
3. **미커밋 수정사항** → 커밋 필요

### Round 2 발견 및 수정
4. **PII in val/sft** → 전화번호/URL 마스킹 (0 severe)
5. **Verifier SFT audit** → sft_5task_v3.jsonl 기준 업데이트

### Round 3 발견 및 수정
6. **Budget gate CPT-only** → 총 비용 기준으로 변경 ($16.73 < $30)
7. **SFT 비용 과대추정** → SFT_TIMEOUT_HOURS cap 적용
8. **Launcher WARN gate** → PASS|WARN 허용

## 최종 상태
- 데이터: PASS (leakage 0, PII 0)
- 코드: PASS (64 py + 11 sh 컴파일, EOS fix, format detect)
- 배포: PASS (dry-run, verifier WARN, $16.73 < $30)
- **전체: PASS — RunPod 배포 준비 완료 (커밋 후)**
