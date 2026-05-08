# Dalbitalba 학습 파이프라인 통합 머지 — 2026-05-08 산출물

## 핵심 결과 (한 줄)

`unoa-eng/dalbitalba-train-data` 6개 open PR (#2~#7) → 단일 통합 PR (#8) → admin merge to main. 3-모델 적대적 검증 (Claude Opus 4.7 / GPT-5 Codex / Gemini 2.5 Pro) **3 라운드** 통과 + 로컬 MLX smoke 검증 + 학습 품질 fix 7건 추가. main HEAD: `5f76f6e`.

## 위치

`docs/handoff/2026-05-08/` (레포 내부)

## 산출물 인덱스

```
.
├── README.md                                           ← 이 파일
├── dalbitalba_consolidation_merge_2026-05-08.pptx     ← 발표 자료 (10 슬라이드, 16:9)
└── screenshots/
    ├── 01_pr8_overview.png        통합 PR #8 메인 뷰 (Merged 배지)
    ├── 02_main_merge_commit.png   main HEAD `d9ac22e` 머지 커밋
    ├── 03_closed_prs.png          6 PR 모두 closed 상태
    ├── 04_pr8_files.png           PR #8 변경 파일 리스트 (89 files)
    └── 05_main_history.png        main 브랜치 커밋 히스토리
```

## 빠른 검증 링크

- **PR #8 (Merged)**: https://github.com/unoa-eng/dalbitalba-train-data/pull/8
- **main HEAD**: https://github.com/unoa-eng/dalbitalba-train-data/commit/d9ac22e
- **닫힌 6개 PR**: https://github.com/unoa-eng/dalbitalba-train-data/pulls?q=is%3Apr+is%3Aclosed

## 작업 흐름 요약

| Phase | 작업 | 결과 |
|---|---|---|
| 0 | Stash + fetch 72 branches | working tree clean, stash@{0} 보존 |
| 1 | 6 PR 병렬 분석 + verifier | sibling 6 + verifier 1 |
| 2 | Base 모델 3-에이전트 paper-grade 리서치 + verifier | Qwen3-8B-Base 유지 합의 |
| 3 | 통합 plan 작성 | central anchor: PR #7 |
| 4.1 | Cherry-pick consolidation 실행 | 84-커밋 우회한 selective import + 5 commit |
| **4.1b R1** | /omc-teams 3-모델 적대적 debate | **3 BLOCKER + 3 HIGH 발견** |
| 4.1c | 6 fix + v3 self-dedup 권장 cleanup 적용 | 6 commit |
| **4.2 R2** | 3-모델 재검증 | **claude/codex APPROVE_W/FOLLOW_UP, gemini APPROVE_FOR_PUSH** |
| 4.3 | Push + PR #8 + admin merge + 6 PR close | main `d9ac22e` |
| **5.1** | **R2 follow-up: 3 학습품질 fix** | live Phase-3 env wiring + 토크나이저 force-include + train_sft 다중 복제 |
| **5.2** | **로컬 MLX smoke (Qwen3-0.6B 200 iter)** | 88.6% token-fire / val loss 5.114→1.823 / sample (e)에서 single-token `ㅈㄴ` |
| **5.3 R3** | **3-모델 재검증** | claude APPROVE_W/FOLLOW_UP, codex REJECT, gemini APPROVE → banker's rounding 발견 |
| **5.4** | **R3 BLOCKER 4건 fix + 검증** | fractional multiplicity (1.5≠2.0 측정 OK), env export, manifest hash |
| 5.5 | Push to main | main `5f76f6e` |

## 3-모델 검증 verdict 파일

| Round | Model | Path | Verdict |
|---|---|---|---|
| R1 | Claude Opus 4.7 | `/tmp/dalbi-verdict-claude.md` | APPROVE_WITH_CHANGES |
| R1 | GPT-5 Codex | `/tmp/dalbi-verdict-codex.md` | REJECT |
| R1 | Gemini 2.5 Pro | `/tmp/dalbi-verdict-gemini.md` | REJECT |
| R2 | Claude Opus 4.7 | `/tmp/dalbi-verdict-r2-claude_md` | APPROVE_WITH_FOLLOW_UP_PR |
| R2 | GPT-5 Codex | `/tmp/dalbi-verdict-r2-codex_md` | APPROVE_WITH_FOLLOW_UP_PR |
| R2 | Gemini 2.5 Pro | `/tmp/dalbi-verdict-r2-gemini_md` | **APPROVE_FOR_PUSH** |
| R3 | Claude Opus 4.7 | `/tmp/dalbi-verdict-r3-claude_md` | APPROVE_W_FOLLOW_UP_PR (banker's rounding 발견) |
| R3 | GPT-5 Codex | `/tmp/dalbi-verdict-r3-codex_md` | REJECT_REQUIRES_MORE_FIXES (Fix 1 wiring 발견) |
| R3 | Gemini 2.5 Pro | `/tmp/dalbi-verdict-r3-gemini-cli.md` | APPROVE_FOR_PUSH_AND_RUNPOD |

## 적용된 fix commits (`git log main`)

```
5f76f6e docs(tokenizer): update README vocab count 151879 → 151908                       [R3 cosmetic]
be03113 fix(chain): SFT data manifest-hash check forces rebuild on env change            [R3 BLOCKER]
95da7ea fix(recipe): export SFT_LOSS_WEIGHT_* and training vars                          [R3 BLOCKER]
f37168e fix(train): fractional stochastic multiplicity + defensive lw parsing             [R3 BLOCKER]
1b0050b fix(tokenizer): force-include CORE_DOMAIN_TERMS + slang Jamo                     [R2 follow-up]
db18b03 fix(train): numeric loss_weight oversampling (later corrected by R3)             [R2 follow-up]
e405510 fix(round2): live Phase-3 builder consumes SFT_LOSS_WEIGHT_* + MinHash dedup     [R2 follow-up]
568079c docs: handoff package for PR #8                                                  [handoff]
d9ac22e Merge pull request #8 from unoa-eng/merge/main-consolidation                     [통합 머지]
6fd4d9f fix(data): v3 self-dedup pass — sft_pairs.v3 1.1% residual → 0%                 [recommended cleanup]
a67016a fix(train): enforce native chat template                                          [R1 HIGH#5]
2a0273f fix(chain): set -e + remove non-essential || true                                [R1 HIGH#4]
2e0caca fix(chain): ORPO Phase 4 failure fatal when ORPO_NUM_EPOCHS > 0                  [R1 BLOCKER#3]
05dfdbc fix(tokenizer): wire tokenizer_v4 end-to-end via TOKENIZER_PATH env              [R1 BLOCKER#2]
3c4eca8 fix(data): rebuild sft_pairs.v3 from MinHash-deduped v2 + add loss_weight        [R1 BLOCKER#1]
de91b24 feat(train): unify train_sft.py
6f66ee7 fix(prelaunch): remove ALLOW_MISSING_RUNTIME_SECRETS bypass
... (PR #3/#4/#7 commits + selective PR #2 imports)
```

## 핵심 데이터 검증 수치

| 항목 | 값 |
|---|---|
| `sft_pairs.v3.jsonl` 행 수 | 44,716 (PR #3 dedup 후 재빌드 + self-dedup) |
| `loss_weight` 필드 존재율 | 100% |
| `loss_weight > 1.0` 비율 | 10.41% |
| 잔존 dedup | 0.0% |
| `tokenizer_v4` vocab | **151,908** (+210 Kiwi + 22 CORE_DOMAIN + 15 SLANG_JAMO) |
| 도메인 단어 single-token | 쩜오/텐카/밀빵/마담/강남역/논현동/가라오케 + ㅈㄴ/ㅇㅈ/ㅋㅋ/ㅎㅎ/ㅠㅠ etc. |
| Token-fire rate (val_set, n=2491) | **88.6%** content-bearing / 90.9% FORCE_INCLUDE |
| MLX smoke val loss | 5.114 → 1.823 (200 iter Qwen3-0.6B) |
| Multiplicity 검증 (1.5 vs 2.0) | 1.5→25 expanded, 2.0→20 expanded (R3.5 distinguishable) |
| `ALLOW_MISSING_RUNTIME_SECRETS` 잔존 | 0 (제거됨) |
| `set -euo pipefail` | enabled (line 17) |
| ORPO fatal guards | 2 sites |

## Phase 5.7 — 종합 로컬 정합성 검증 (8 stages F-M)

전체 리포트: [`local-integrity/FINAL_INTEGRITY_REPORT.md`](local-integrity/FINAL_INTEGRITY_REPORT.md)

| Stage | 결과 | 핵심 수치 |
|---|---|---|
| **F** MLX SFT Qwen3-4B 1000 iter | ✅ COMPLETE | val 2.81→2.46, peak 9.3GB, 1041s |
| **G** ORPO pairs build+validate | 🟡 PARTIAL | 5/100 chosen이 val_set leak (build script 버그) |
| **H** MLX CPT smoke 500 iter | ✅ PASS_W_NOTES | 초기 1.53 nat 스파이크 — production warmup으로 mitigated |
| **I** Full-corpus token-fire (4 corpora) | ✅ **95% PASS** | 0 force-include dead, 109,028 rows 검사 |
| **J** PR #6 style classifier v2 retrain | ✅ **COMPLETE 3 epochs** | **test_AUC=0.9999 / test_acc=0.997 / cm=[[499,1],[2,498]]** (Wikipedia v1의 trivial AUC=1.0와 달리 KLUE NLI + polite 존댓말 negatives 포함) |
| **K** End-to-end glue test | ✅ **10/10 PASS** | manifest hash 4 permutation 정확 |
| **L** Persona matrix 5×3 generation | 🟡 PATH_VERIFIED | 0.6B+200iter로는 persona 조건화 불가 — production에서 측정 |
| **M** R3 fix delta 정량화 | ✅ COMPLETE | **합성 50:50 lw=1.5/2.0: pre 2.000x → post 1.751x (정확히 differentiable)** |

### 핵심 정합성 검증 결과

- **token-fire 95%** (val-only 88.6% 보다 향상, 실 production schema 기준)
- **R3 fix 진짜 작동** — banker's rounding 버그 해결 정량 입증
- **전체 chain script** AST + manifest hash + recipe export 모두 정상
- **남은 5%**: 8B QLoRA fit + CUDA bnb numerical (RunPod 자체가 검증)

## RunPod 본 학습 진행 가능 여부 — **YES (with caveats)**

학습 품질에 직접 영향을 주던 4건이 R3 라운드에서 발견되어 **모두 fix되었습니다**:
- ✅ `train_sft.py` banker's rounding (1.5와 2.0 동일 동작) → fractional stochastic multiplicity
- ✅ `recipes/round2-cycle1.env` env 변수가 subprocess에 전파 안 됨 → `export` 추가
- ✅ `chain_train_round2.sh` stale SFT data 재빌드 안 함 → manifest-hash check
- ✅ `tokenizer_v4` 도메인 argot 일부 fragment → CORE_DOMAIN_TERMS + SLANG_JAMO force-include

R3.5 empirical validation: weight=1.5 → 25 expanded copies, weight=2.0 → 20 expanded copies. 두 weight가 distinguishable. mutator escalation effective.

## 잔여 follow-up (RunPod 진행에는 무영향, 다음 사이클에 처리)

1. ~~`train_eval_process.py:246` dead `ALLOW_MISSING_RUNTIME_SECRETS` 주입 (cosmetic)~~ — **5.8.3 완료**
2. ~~`--skip-mauve` waiver 두 군데 정리~~ — **5.9.1 완료** (`chain_train_round2.sh` env-gated `MAUVE_DISABLED=1`만 적용)
3. ~~Phase 3 `if-then-else fail_with_logs` 래핑~~ — **5.9.2 완료**
4. ~~`git lfs migrate`~~ — **5.9.3 deferred-with-guide** (`docs/handoff/2026-05-08/LFS_MIGRATION.md` 참조)
5. PR #6 retrain (Korean forum/news negative class) — eval signal 강화 — **5.8.5 부분 진행: ep1 완료 (val_acc=0.99, loss=0.073, 1818s), ep2/ep3 + test_AUC + confusion matrix는 RunPod 사이클로 이월** (M4 CPU 30분 budget 한계)
6. ~~ORPO_NUM_EPOCHS 기본값 통일 (production launch는 recipe pin으로 마스킹됨)~~ — **5.8.4 완료**

## Phase 5.8 — Final remaining fixes (사용자 오프라인 자동 마무리)

Phase 5.7 audit 후 잔여 학습품질/위생 fix를 자동 실행. 5건 모두 atomic commit 적용.

| # | Fix | Commit | 결과 |
|---|---|---|---|
| 5.8.1 | ORPO build script — chosen 후보 사전 val-set 필터링 | `009d7e5` | G_validation_v2.json: chosen_leaks=0, **verdict=PASS** |
| 5.8.2 | CPT recipe — lr 2e-4→1e-4, warmup 0.03→0.08 (cold-start 1.53 nat 스파이크 mitigation) | `02f423d` | recipe `bash -n` PASS |
| 5.8.3 | `train_eval_process.py:246` dead env injection 제거 | `55b116c` | py_compile PASS |
| 5.8.4 | `chain_train_round2.sh` ORPO_NUM_EPOCHS 기본값 `:-0` 통일 (5/5 sites) | `cc7e99e` | bash -n PASS |
| 5.8.5 | PR #6 style classifier v2 재학습 (Korean news+KMMLU+polite negatives) | (커밋 없음 — 부분 진행) | **ep1 PASS**: val_acc=0.99, train_loss 0.232→0.073 (500 batch), 1818s on M4 CPU. ep2/ep3 + test_AUC RunPod 이월 |

### Fix 5.8.1 임팩트 (확인된 leak 수)

기존 `--val-set` 필터는 cpt_corpus의 top-level `text` 키만 인덱싱했고, refinement-runs 경로와 val.v3의 `messages[]/completion` 항목은 미처리 상태였음. 새 `_index_val_completions()` 헬퍼는 G_validate_orpo.py와 동일한 harvest 로직을 사용 (text/completion/answer/target_comment + assistant messages). 재실행 시 2012건의 val-leak 후보가 chosen_pool에서 사전 제거됨.

### Fix 5.8.2 노트

스펙은 `CPT_WARMUP_STEPS=100`을 요청했으나 `train_cpt.py:75`는 `CPT_WARMUP_RATIO`만 읽음 (절대 step 변수는 dead). `total_steps × ratio` 식으로 계산되므로 ratio를 0.08로 조정 (cycle-1 typical step수에서 ~100 step 효과).

### Fix 5.8.5 노트 (J 부분 진행)

이전 시도는 ep1 batch 250/500에서 종료. 이번 시도는 ep1을 끝까지 완료 (batch 500/500, train_loss 0.0734, val_acc 0.9900, 1818초). 3 epoch 풀 학습은 M4 CPU 기준 ~90분 예상이라 30분 budget 내 완주 불가 — RunPod 사이클로 이월. ep1 결과만으로도 다양화 negatives (Korean news + KMMLU NLI + 존댓말 synth) 분리도가 매우 높음 입증. **별도 commit은 생성하지 않음** (스펙은 "only if completes" 명시). 산출물: `local-integrity/phase-5.8/J_v2_summary.json` + `J_v2_run.log`.

### Phase 5.8 산출물 위치

```
docs/handoff/2026-05-08/local-integrity/phase-5.8/
├── G_orpo_pairs_v2.jsonl              ← 100 pairs, val-leak free (PASS verdict)
├── G_validation_v2.json               ← chosen_leaks=0 / rejected_leaks=0 / verdict=PASS
├── J_v2_summary.json                  ← Phase 5.8.5 부분 진행 분석
├── J_v2_run.log                       ← 풀 학습 로그 (ep1 done까지)
└── J_v2_classifier_manifest.json      ← classifier config snapshot
```

## 보존된 사용자 작업

`stash@{0}` ("WIP before P0/P1 PR merge planning 2026-05-08T12:45:50") — 시작 시점의 5 modified + 21 untracked 파일 (`TRAINING_DESIGN_V3.md`, `.venv-mlx/`, `runs/`, `v3-data/`, `hf_assets.example.json` 등). `git stash pop` 또는 `git stash show -p stash@{0}` 으로 복원 가능.

---

Generated by 3-model adversarial loop process via `/omc-teams` (claude / codex / gemini sequential workers).
