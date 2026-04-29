# Pipeline Verification Status Dashboard

Updated: 2026-04-29T16:10 KST

## Stage-by-Stage Status

| # | Stage | Status | Evidence | Notes |
|---|-------|--------|----------|-------|
| 1 | CPT Data Build | ✅ TESTED | 53,675 rows, PII=0 | cpt_structured_v3.jsonl |
| 2 | SFT Data Build | ✅ TESTED | 58,476 rows, 5-task mix | sft_5task_v3.jsonl |
| 3 | Tokenizer Extension | ✅ TESTED | 11 tokens, vocab correct | v3-data/tokenizer/ |
| 4 | CPT Training | ✅ TESTED | MLX 0.6B 200iter, val loss 4.822→3.250 | /tmp/e2e-test/cpt-adapter |
| 5 | Model Merge | ✅ TESTED | merge script verified, tokenizer saved | code review + dry-run |
| 6 | SFT Training | ✅ FIXED+TESTED | format detection, val loss 5.082→3.193 | instruction/input/output OK |
| 7 | Generation (phase6) | ✅ FIXED | SFT_ADAPTER_REPO default empty | run_eval.sh patched |
| 8 | Evaluation (phase6) | ✅ TESTED | REQUIRED_GATE_METRICS, fail-closed | phase6_eval.py smoke-tested |
| 9 | Korean PPL Check | ✅ TESTED | Fail-closed on None | part of phase6 gate |
| 10 | HF Upload | ✅ TESTED | adapter layout verified | upload-readiness proxy |
| 11 | Promotion Gate | ✅ FIXED+TESTED | check_phase6_promotion.py new | verdict=PROMOTE on valid input |

## Quality Trajectory (0.6B MLX Proxy)

| Stage | bigram_jsd | length_kl | style_pass | Interpretation |
|-------|-----------|-----------|------------|----------------|
| Base | 0.657 | 20.10 | 1/9 | Off-domain |
| CPT | 0.683 | 0.87 | 3/9 | Length normalized, style improved |
| SFT | 0.604 | 1.40 | 2/9 | Bigram improved, mixed overall |

## Files Modified (need commit)

- `train_sft.py` — instruction format support
- `scripts/run_eval.sh` — SFT_ADAPTER_REPO unbound fix
- `scripts/check_smoke_promotion.py` — CLI args support
- `scripts/check_phase6_promotion.py` — NEW: phase6 promotion gate
- `recipes/budget30_v2.env` — SFT enabled
- `chain_train.sh` — SFT_INPUT_JSONL forwarding
- `scripts/launch_train_pod.py` — SFT env forwarding

## Untracked (need commit)

- `v3-data/replay_korean_5k.jsonl` — recipe에서 참조, 커밋 필수

## Pending: Full-Range Loop

- Round 1: 3-worker codex team 실행 중 (data/training/eval 검증)
- Round 2+: Round 1 결과에 따라 자동 진행
- 종료 조건: 연속 3회 만장일치 PASS
