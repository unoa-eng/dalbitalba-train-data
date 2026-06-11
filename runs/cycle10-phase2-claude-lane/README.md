# Phase 2 — Claude lane (canonical, 2026-06-11)

사이트 시뮬레이션 본질 목표(2026-06-09 user redirect) 기준의 Phase 2:
seed 3,204 threads (`runs/cycle10-phase1-claude-direct/threads_v3.jsonl`)의
댓글 10,008개를 교체한다. 63k 1:1 expansion이 아니다.

## 분담 (user 지시 2026-06-10)
- **게시판 초기 (2024-10 ~ 2025-03, 555개): Claude 직접 작성** →
  `dalbit_phase2_early_merged.jsonl` (uniqueness 100%, 평균 36.3자,
  4-병렬 writer + 검증 완료)
- **bulk (2025-04+, 9,453개): boost-4 로컬 생성** →
  `dalbit_phase2_src_bulk.jsonl` 소스로
  `dalbit_phase2_full_launch.sh` (500/chunk, resume: `bash ... <start_chunk>`)
  출력: `/tmp/dalbit_phase2_full/chunk_*/`
  setup: cycle9 ORPO `pref_targeted_artifact_v4_0000200` + qwen3-8b-mlx-4bit,
  temp 0.35 / top_p 0.9 / rep 1.3 / min 60 / max 200 + 16 text-bias (boost-4)

## 재조립
`dalbit_phase2_reassemble.py <samples.jsonl(연결)> <src_bulk.jsonl> <threads_v4.jsonl>`
- cleanup v2 (winning recipe — v3 회귀 금지), [N] 인덱스 재번호, 실패 시 seed fallback
- early 555 merged 반영은 reassemble 확장 필요 (root_id+comment_index 기준 동일)

## 폐기된 lane — 실행 금지
`phase2_prompt_pool.jsonl` / `phase2_production_driver.sh` /
`phase2_autonomous_full_run.sh` (commits 3c0ecf9, 63e4342, d00f0f5)
— 2026-06-09 user redirect와 모순되는 63k 1:1 expansion. 2026-06-10 세션
충돌(상호 프로세스 kill) 후 일원화하며 중단됨. 재개하지 말 것.

## 완료 기준
- AUC audit (HashingVectorizer char_wb(2,4) + LogReg 5-fold, multi-seed) ≤ 0.55~0.58
- 댓글 uniqueness > 95%
- threads_v4.jsonl push
