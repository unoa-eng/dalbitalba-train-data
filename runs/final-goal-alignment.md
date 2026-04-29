# Final Goal Alignment Audit

- Timestamp: `2026-04-28`
- Scope: `AUTOPILOT_LOOP.md` goals + principles, live artifacts (`cpt_corpus.v3.jsonl`, `cpt_context_stream.jsonl`, `sft_pairs.v2.jsonl`, `val_set.v2.jsonl`), and active launch/eval code paths.
- Overall verdict: `FAIL`

The repo has real MLOps safety work in place, but the active `budget30_v2` path still misses key parts of the stated acceptance target. The biggest gaps are: thread-aware reply conditioning exists as an artifact but is not the paid training input, default eval is not blind and not kind-stratified in practice, and promo/operator language still survives in the live CPT corpus at non-trivial rates.

## Required Worker Checks

- `FAIL` 1. Comment reply flow
  Evidence: `cpt_context_stream.jsonl` contains `12,014` `context_comment` rows, with `3,939` reply rows carrying `parent_comment_key` (`32.79%` of context rows, `8.20%` of the full CPT stream). This proves the artifact can encode parent-child flow. But the live launch path does not use it: `recipes/budget30_v2.env` is CPT-only (`SKIP_SFT=1`, `SFT_NUM_EPOCHS=0`), and `chain_train.sh` / `scripts/launch_train_pod.py` default `TRAIN_CPT_JSONL` to `cpt_corpus.v3.jsonl`, not `cpt_context_stream.jsonl`. If SFT were enabled, `train_sft.py` still trains only on `{post}\n{comment}` with masked labels, and `sft_pairs.v2.jsonl` has no parent comment field. Surface reply tags survive, but actual reply conditioning is not in the active paid path.

- `PASS` 2. 초성 / emotive marker learnability
  Evidence: in `cpt_corpus.v3.jsonl` (`48,054` rows), row-level marker counts are high enough to learn from: `ㅋㅋ 5,717` (`11.90%`), `ㅠㅠ 2,616` (`5.44%`), `ㄹㅇ 712` (`1.48%`), `ㅇㅈ 687` (`1.43%`), `ㅎㅎ 642` (`1.34%`). Raw crawl rates are in the same ballpark: `ㅋㅋ 9.73%`, `ㅠㅠ 4.74%`, `ㄹㅇ 1.22%`, `ㅇㅈ 1.27%`, `ㅎㅎ 1.87%`. The marker family is materially present in both posts and comments, so the model can learn it.

- `FAIL` 3. Ad vs user speech separation / promo contamination
  Evidence: live `cpt_corpus.v3.jsonl` still contains `2,392 / 48,054` rows (`4.98%`) matching phone/kakao/recruiter heuristics. By kind, that is `1,993 / 37,584` comments (`5.30%`) and `399 / 10,470` posts (`3.81%`). Examples still include explicit operator templates like `강남 퍼펙트 "동생이사"`, `문의 환영`, contact handles, and roster-style recruiter spam. `scripts/build_context_cpt.py` skipped `10,221` promo-like context candidates, but the active CPT corpus still learns a meaningful amount of operator voice.

- `FAIL` 4. Kind-stratified eval
  Evidence: `scripts/refinement_loop.py` is kind-aware (`KIND_ORDER = ("post", "comment")`) and `eval/make_eval_samples.py` does stratified blind sampling by `kind`. But the default paid eval path is `EVAL_MODE=phase6` (`recipes/budget30_v2.env`), and `scripts/phase6_generate.py` writes only `{"text","seed"}` without `kind`. `scripts/phase6_eval.py` only emits per-kind breakdowns when both sides carry kind metadata, so the default phase6 run cannot actually stratify generated outputs by kind in practice.

- `FAIL` 5. Short turns and question/reply patterns
  Evidence: short turns and question rates are preserved well at the surface level. In comments, `xs+sm` rows are `25,939 / 37,584` (`69.02%`) in `v3` vs `36,165 / 55,849` (`64.75%`) in raw. Comment question rate is `17.76%` in `v3` vs `16.15%` in raw; post question rate is `54.90%` vs `54.14%`. Reply-tag share also survives: `11,387 / 37,584` comment rows (`30.30%`) in `v3` vs `13,995 / 55,849` (`25.06%`) in raw, and `sft_pairs.v2.jsonl` contains `12,931` reply-tagged comment rows. But because parent context is not used in the active training recipe, the full “question/reply pattern” objective is only surface-preserved, not behaviorally taught.

- `FAIL` 6. Avoiding the 0618-style meaningless GPU burn
  Evidence: there are strong mechanical gates, listed below, but they are not sufficient to block the current goal-misaligned paid recipe. The pipeline can still spend GPU on `cpt_corpus.v3.jsonl` CPT-only training without thread-aware reply conditioning and without default blind/kind-stratified eval. That is safer than the old SOLAR failure mode, but still not enough to call “meaningless burn avoided” for this project goal.

## AUTOPILOT Goal Bullets

- `FAIL` Blind eval should make human/AI discrimination difficult.
  Evidence: the blind-eval toolchain exists (`eval/make_eval_samples.py`, `eval/judge_3way.py`, `eval/native_eval_kit.py`), but the active recipe defaults to deterministic `phase6` metrics, not blind eval. There is no default paid gate that requires a blind human-vs-AI indistinguishability check before promotion.

- `FAIL` Raw-vs-generated distributions should be close on length, punctuation, initials, slang, emotion tone, and comment structure.
  Evidence: some pieces are addressed, but not the whole target. `scripts/refinement_loop.py` compares term, punctuation, emotive markers, tone, and structure by kind, but that is an offline diagnostic, not a mandatory launch/eval gate. More importantly, live training still ignores `cpt_context_stream.jsonl`, so comment structure / reply behavior is not actually taught in the active budget30 run.

- `FAIL` Ad/operator noise should not be learned as human speech.
  Evidence: `4.98%` of the live CPT corpus still hits the repo’s own promo/operator heuristics, including clear recruiter templates and contact calls to action.

## Current Core Principles

- `FAIL` “게시글만 닮으면 안 되고 댓글 문맥과 reply 흐름까지 닮아야 한다.”
  Evidence: the thread-aware corpus exists, but the active paid path does not train on it.

- `FAIL` “초성, 감정 기호, 짧은 턴, 질문/반문, 생략 부호까지 같이 맞춰야 한다.”
  Evidence: initials/emotive markers and short turns are present, but reply conditioning is absent from the active recipe and there is no mandatory gate on punctuation / ellipsis alignment in the paid loop.

- `FAIL` “운영자 홍보 댓글, 전화번호, 오픈카톡, 반복 광고 템플릿은 최대한 제거하거나 문맥용으로만 제한한다.”
  Evidence: live `v3` still has `2,392` promo-flagged rows and obvious operator templates.

- `FAIL` “eval 은 kind 기준으로 층화한다.”
  Evidence: supported in code, not enforced in the default `phase6` generation/eval run because generated rows have no `kind`.

- `FAIL` “개선이라고 보고할 때는 항상 이전 실행 대비 근거를 같이 남긴다.”
  Evidence: `scripts/cycle_report.py` writes a current-run report, and `scripts/recipe_mutator.py` tracks history, but there is no hard requirement in the active report path to emit previous-run diffs alongside claimed improvements.

- `PASS` “GPU 는 비싸므로 raw 분석, 데이터 재생성, 정합성 검증, 프롬프트/샘플링 설계는 가능한 한 로컬에서 먼저 끝낸다.”
  Evidence: the repo has local-first machinery: `scripts/profile_raw_crawl.py`, `scripts/build_thread_aware_datasets.py`, `scripts/local_verification_loop.py`, `scripts/check_l40s_availability.py`, dry-run launch modes, and `launch_train_pod.py` blocks `budget30` unless `runs/latest-local-verification.json` is `PASS`. The current latest local verification file is `PASS`.

- `FAIL` “RunPod 는 지금 GPU 를 켜야만 다음 증거를 얻을 수 있을 때만 켠다.”
  Evidence: this is not encoded as a hard gate. A `budget30` launch can still proceed after a mechanical verifier `PASS` even when the current recipe skips thread-aware reply data and default blind eval, which means spend can still happen before the right evidence path is in place.

## Live Gate Inventory

These gates are real and worth keeping, but they do not fully solve goal alignment:

- `scripts/launch_train_pod.py`
  `assert_verifier_pass_for_budget30()` refuses launch unless `runs/latest-local-verification.json` has `verdict=PASS`, unless `FORCE_LAUNCH=1`.

- `scripts/local_verification_loop.py`
  Fails on malformed JSONL, missing required keys, direct PII, mojibake, catastrophic duplicate rate `> 0.50`, strict budget overrun for `budget30`, missing scripts, and unreadable/incomplete HF artifacts.

- `chain_train.sh`
  Hard-refuses non-Qwen3 base models unless `FORCE_BASE_MODEL=1`; checks required env and dataset files; runs a preflight smoke test for CUDA, `bitsandbytes`, HF auth, Qwen3 config/tokenizer access; applies per-stage timeouts; traps signals; stops the pod on failure and after completion.

- `scripts/check_smoke_promotion.py`
  Blocks promotion unless the latest train status is `done_ok`, the CPT adapter exists, `trainer_state.json` is readable, and `global_step >= max_steps`.

- `scripts/run_eval.sh`
  Applies eval-stage timeouts, traps abort signals, persists artifacts, and stops the pod after eval.

- `scripts/autonomous_loop.sh`
  Enforces a single-pod wait/relaunch pattern and reacts to eval gate verdicts rather than blindly relaunching.

## Bottom Line

If the acceptance bar is “generated text should be indistinguishable from real dalbit forum posts/comments,” the current repo is not there yet.

The minimum fixes before calling this aligned are:

- Switch the paid CPT input from `cpt_corpus.v3.jsonl` to `cpt_context_stream.jsonl`, or otherwise make parent-comment conditioning part of the active training path.
- Make the default eval path actually kind-stratified in practice by emitting `kind` from `phase6_generate.py`, and restore blind eval as a real promotion gate instead of a legacy side path.
- Reduce promo/operator contamination in the live CPT corpus below the current `4.98%` flagged rate, with explicit rejection of recruiter-template carryover.
