# Re-audit Goal Alignment

- Date: `2026-04-28`
- Repo: `/Users/unoa/projects/dalbitalba-train-data`
- Source: `/Users/unoa/Downloads/crawled-data-v2`
- Active CPT input audited: `cpt_context_stream.jsonl`
- Live CPT rows: `43,635` = `10,063 post` + `21,558 comment` + `12,014 context_comment`
- Overall verdict: `FAIL`

Status legend:

- `PASS`: aligned now, with current evidence.
- `FIXED`: was a prior fail in `runs/final-goal-alignment.md`, now fixed.
- `STILL-FAIL`: was a prior fail and still is not aligned.

Evidence base used for this re-audit:

- `AUTOPILOT_LOOP.md`
- `recipes/budget30_v2.env`
- `chain_train.sh`
- `scripts/phase6_generate.py`
- `scripts/phase6_eval.py`
- `scripts/run_eval.sh`
- `scripts/local_verification_loop.py`
- `runs/final-goal-alignment.md`
- `runs/final-eval-readiness.md`
- `runs/final-data-fidelity.md`
- fresh `python3 scripts/refinement_loop.py --source-dir /Users/unoa/Downloads/crawled-data-v2 --cycle 1 --cpt-jsonl cpt_context_stream.jsonl` -> `runs/refinement-20260428-171444/cycle-1/diagnostic.json`
- direct spot checks on `cpt_context_stream.jsonl`, `val_set.v2.jsonl`, and `git status`

## Requested Checks

| Check | Status | One-line evidence |
| --- | --- | --- |
| Reply flow teachability | `FIXED` | The active paid recipe now points `TRAIN_CPT_JSONL` to `cpt_context_stream.jsonl`, and that file contains `12,014` `context_comment` rows with `댓글` / `부모댓글` / `답글` structure, so reply context is finally in the live CPT path. |
| 초성 / emotive learnability | `PASS` | The live CPT stream still has enough marker density to learn from (`comment+context_comment`: `ㅋㅋ 8.52%`, `ㅠㅠ 2.73%`, `ㅇㅈ 0.91%`, `ㄹㅇ 0.99%`), even though exact distribution matching is still off. |
| Promo contamination | `STILL-FAIL` | The live CPT stream still hits repo promo heuristics on `1,908 / 43,635` rows (`4.37%`), including `585` plain comments and `1,120` `context_comment` rows. |
| Kind-stratified eval | `FIXED` | `phase6_generate.py` now emits `kind`, `phase6_eval.py` consumes explicit kind metadata, and `val_set.v2.jsonl` rows already carry `kind`, so the default phase6 path can stratify by kind now. |
| Short turn preservation | `STILL-FAIL` | Once `context_comment` rows are counted as the active comment-side training signal, comment short-turn rate drops to `64.12%` vs raw `73.80%`, and question rate rises to `35.59%` vs raw `16.15%`. |
| Meaningful GPU spend gates | `STILL-FAIL` | The latest verifier `PASS` is not validating the live CPT file: `scripts/local_verification_loop.py` defaults to `cpt_corpus.v3.jsonl`, while the live CPT file is untracked `cpt_context_stream.jsonl` (`43,635` rows) and stale reports disagree on row count (`42,473`, `43,635`, `45,893`). |
| Blind eval path | `STILL-FAIL` | The default eval mode is still `phase6`, and the blind-judge branch in `scripts/run_eval.sh` is still the legacy path that builds its blind set from `cpt_corpus.jsonl`, not the active context stream. |
| Raw distribution matching | `STILL-FAIL` | The fresh refinement run on `2026-04-28 17:14:44` still reports `17` moderate gaps, including comment `ㅋㅋ` overuse (`2.1262x`) and comment `avg_sentences_per_text` `2.06` vs raw `4.38`. |

## AUTOPILOT Goals

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| blind eval 에서 사람/AI 구분 난도가 높아질 것 | `STILL-FAIL` | Blind eval exists, but it is not the default paid path, and the default path is still deterministic `phase6` rather than the blind judge flow. |
| raw 대비 길이, 문장부호, 초성, 은어, 감정톤, 댓글 구조 분포가 가까워질 것 | `STILL-FAIL` | The fresh refinement run still found `17` moderate gaps, and active comment-side training text is materially more scaffolded than raw (`...` rate `13.53%` vs raw `4.30%`). |
| 광고성/운영자성 노이즈를 사람 말투로 오인해 학습하지 않을 것 | `STILL-FAIL` | The live CPT stream still contains `1,908` promo-hit rows by repo heuristics, including plain comments like `언니 카톡 알려줘` and scaffolded context rows with recruiter/operator language. |

## Data Priorities

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 1차 원천: raw crawl | `PASS` | All current fidelity and gap checks are still anchored to `/Users/unoa/Downloads/crawled-data-v2`, and the active context stream was regenerated from that raw source. |
| 2차 보조: `research/obsidian-export` | `PASS` | The active train/eval path does not depend on Obsidian content, so it remains an auxiliary layer rather than a hidden training source. |
| 3차 산출: `cpt_corpus.jsonl`, `sft_pairs_v2.jsonl`, `.state/thread_aware/*` | `PASS` | The repo still uses generated JSONL artifacts as training/eval inputs, with `cpt_context_stream.jsonl` now acting as the promoted CPT artifact. |
| raw 가 진실원본 / snapshot 이 raw 와 멀어지면 재생성 | `FIXED` | The active CPT pointer moved off `cpt_corpus.v3.jsonl` and onto regenerated `cpt_context_stream.jsonl`, which is exactly the “regen when snapshot drifts” behavior the doc calls for. |

## Current Core Principles

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 게시글만 닮으면 안 되고 댓글 문맥과 reply 흐름까지 닮아야 한다 | `FIXED` | The active paid path finally trains on `context_comment` rows, and sampled rows clearly preserve root-comment and parent-reply structure. |
| 초성, 감정 기호, 짧은 턴, 질문/반문, 생략 부호까지 같이 맞춰야 한다 | `STILL-FAIL` | Marker families are present, but the active comment-side signal is still structurally off from raw: short turns `64.12%` vs `73.80%`, question rate `35.59%` vs `16.15%`, ellipsis `13.53%` vs `4.30%`. |
| 운영자 홍보 댓글, 전화번호, 오픈카톡, 반복 광고 템플릿은 최대한 제거하거나 문맥용으로만 제한한다 | `STILL-FAIL` | Promo/operator residue is still too high at `4.37%` of the live CPT stream, and `context_comment` rows are the worst bucket at `9.32%`. |
| eval 은 kind 기준으로 층화한다 | `FIXED` | The old `phase6_generate` missing-kind failure is fixed; both generation and evaluation now handle `kind` explicitly. |
| 개선이라고 보고할 때는 항상 이전 실행 대비 근거를 같이 남긴다 | `STILL-FAIL` | The repo’s current evidence is internally inconsistent across artifacts (`42,473` vs `43,635` vs `45,893` CPT rows), so the “what changed vs last run” story is still not durably clean. |
| GPU 는 비싸므로 raw 분석, 데이터 재생성, 정합성 검증, 프롬프트/샘플링 설계는 가능한 한 로컬에서 먼저 끝낸다 | `PASS` | The repo still has strong local-first machinery (`profile_raw_crawl.py`, `build_thread_aware_datasets.py`, `refinement_loop.py`, `local_verification_loop.py`) and this re-audit stayed entirely local. |
| RunPod 는 지금 GPU 를 켜야만 다음 증거를 얻을 수 있을 때만 켠다 | `STILL-FAIL` | The spend gate is not tied to the live CPT file, and the default eval loop still skips blind eval, so GPU can still be spent before the right evidence path is in place. |

## Autopilot Loop

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 1. raw crawl 상태를 프로파일링한다 | `PASS` | `runs/raw-profile-20260427.json` exists and was produced by the expected raw profiler. |
| `scripts/profile_raw_crawl.py` | `PASS` | The script exists and its output already covers the intended raw crawl profile dimensions. |
| 초성/은어/감정/길이/댓글깊이/광고오염도 확인 | `PASS` | `runs/raw-profile-20260427.json` includes `slang_markers`, `chosung_top`, `emotion_top`, length histograms, `comment_depth`, and `promo`. |
| 2. thread-aware 데이터셋을 재생성한다 | `FIXED` | The active CPT path is now the regenerated thread-aware stream instead of the old snapshot corpus. |
| `scripts/build_thread_aware_datasets.py` | `PASS` | The thread-aware dataset builder is present and the repo now depends on its output indirectly through `cpt_context_stream.jsonl`. |
| root 댓글, reply 댓글, 게시글 맥락이 모두 살아 있는지 확인 | `PASS` | `runs/final-data-fidelity.md` reports `10/10` context reconstruction passes, and the sampled `context_comment` rows clearly show `제목`, `원글`, `댓글` or `부모댓글` + `답글`. |
| 3. 새 데이터셋이 더 맞으면 학습 입력 승격 | `FIXED` | `recipes/budget30_v2.env` now sets `TRAIN_CPT_JSONL=cpt_context_stream.jsonl`, and `chain_train.sh` prefers that file when present. |
| 필요 시 기존 학습 입력 교체 / branch 반영 | `STILL-FAIL` | The active CPT file and recipe are currently untracked (`git status`: `?? cpt_context_stream.jsonl`, `?? recipes/budget30_v2.env`), so branch durability is not complete yet. |
| 변경 이유와 기대 효과를 진행 보고에 남긴다 | `PASS` | `runs/final-eval-readiness.md` and `runs/context-cpt-stats.json` explain why the context stream was promoted and what it was meant to fix. |
| 4. RunPod 에서 학습을 실행한다 | `PASS` | The train-launch path is operational enough for dry-run (`launch_train_pod.py` is green per prior readiness artifacts), even though this audit did not launch a pod. |
| 현재 branch 기준으로 pod 를 띄운다 | `PASS` | The launch path is branch-based and tied to the current repo checkout; no contrary evidence was found. |
| 실패 pod 는 방치하지 말고 원인을 추적해 수정 후 재실행 | `PASS` | `chain_train.sh` and `scripts/autonomous_loop.sh` both have explicit failure capture, timeout, and stop/retry logic. |
| 5. 학습 산출물을 Hugging Face 에 내구성 있게 올린다 | `STILL-FAIL` | The upload automation exists, but the local env still does not pin a current adapter repo for eval continuity, so durable “latest adapter” tracking is incomplete. |
| adapter repo id 를 확인한다 | `STILL-FAIL` | The local no-go audit still found `HF_ADAPTER_REPO` / `SFT_ADAPTER_REPO` missing for the current phase6 eval chain. |
| 필요 시 `.env.local` 의 `HF_ADAPTER_REPO` 갱신 | `STILL-FAIL` | That pointer update still has not happened according to `runs/final-runpod-go-nogo.md`. |
| 6. stratified blind eval 을 실행한다 | `STILL-FAIL` | Kind stratification is fixed, but the default loop is still not the blind-eval path. |
| `scripts/generate_samples.py` | `PASS` | The blind-eval generator script exists and still produces kind-labeled AI samples when given an adapter repo. |
| `eval/make_eval_samples.py` | `PASS` | The blind-set builder exists and samples AI/HUMAN rows stratified by `kind` when both sides provide it. |
| `eval/judge_3way.py` | `PASS` | The blind-judge script is still present as the intended judge stage. |
| `eval/native_eval_kit.py` | `PASS` | The report renderer for blind eval is still present. |
| 7. 생성물과 raw 분포를 비교한다 | `STILL-FAIL` | The repo can compare some dimensions, but the default paid eval still does not make blind misclassification evidence and reply-structure similarity first-class promotion gates. |
| 게시글/댓글 비율 | `PASS` | The refinement diagnostics and kind-aware scripts can report post/comment counts. |
| 길이 분포 | `PASS` | `phase6_eval.py` has `length_kl`, and refinement diagnostics emit length histograms. |
| 질문/감탄/말줄임표 비율 | `PASS` | `refinement_loop.py` explicitly audits punctuation features including `?`, `!`, and `...`. |
| 초성/은어/감정톤 빈도 | `PASS` | `refinement_loop.py` explicitly audits domain slang, 초성, and emotive marker frequency by kind. |
| reply 구조 유사성 | `STILL-FAIL` | The thread-aware corpus preserves reply structure, but the default generated-output gate still does not score generated reply structure against raw. |
| judge 오판/불일치 샘플 | `STILL-FAIL` | That evidence only exists in the legacy blind path, not in the default phase6 loop. |
| 8. 차이가 크면 원인을 분류하고 바로 다음 반복으로 넘긴다 | `PASS` | The fresh refinement run classified `17` moderate gaps and emitted concrete next recommendations immediately. |
| 데이터 필터 문제 | `PASS` | The current top findings explicitly point to promo filtering tradeoffs as likely causes for some under-represented comment terms. |
| 프롬프트/seed 문제 | `PASS` | The loop has a place to attribute prompt/seed issues, even though this re-audit did not isolate one. |
| eval 샘플링 문제 | `PASS` | The loop has the machinery to separate sampling issues, and the blind-path mismatch is already visible as a concrete issue. |
| 학습 파라미터 문제 | `PASS` | The loop can still attribute gaps to training parameters after the distribution diagnostics. |
| 광고/중복 오염 문제 | `PASS` | Promo contamination remains directly measurable and is still one of the real root-cause buckets. |

## RunPod Cost Control

### GPU 시작 전 게이트

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 로컬 raw 분석과 데이터셋 생성이 끝났다 | `PASS` | Both raw profiling and context-stream regeneration have already happened locally. |
| 기존 실행 대비 무엇이 달라졌는지 설명할 수 있다 | `STILL-FAIL` | The repo has stale and conflicting row-count artifacts, so the delta story is still noisier than it should be. |
| 직전 실패 원인을 수정했다 | `STILL-FAIL` | Some important prior failures are fixed (`context stream`, `kind`), but blind eval defaulting and promo contamination are still unresolved. |
| 이미 살아 있는 pod 가 없다 | `PASS` | This audit found no contrary evidence of a live pod, and no local automation artifact claims one is active. |
| 이번 실행이 목표 수렴에 실제로 기여한다 | `STILL-FAIL` | Because the spend gate does not validate the live CPT file and blind eval is still off the default path, a new paid run can still be misaligned with the stated acceptance target. |

### local-first 원칙

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| raw 프로파일링 | `PASS` | Already done locally in `runs/raw-profile-20260427.json`. |
| 정합성 검증 | `PASS` | The repo already contains multiple local fidelity/verification reports, and this audit added another local-only pass. |
| 데이터 필터링/재생성 | `PASS` | `cpt_context_stream.jsonl` is itself the result of local filtering/regeneration work. |
| 샘플 몇 건의 포맷 검토 | `PASS` | This re-audit manually sampled `context_comment` rows and `val_set.v2.jsonl` rows locally. |
| env 검증 | `PASS` | The repo has local env/verifier tooling, even though the eval checker contract is still imperfect. |
| GitHub/HF 상태 확인 | `PASS` | Prior readiness artifacts already checked repo/env readiness without launching a pod. |
| 새 데이터셋으로 CPT/SFT 재학습 | `PASS` | This remains a legitimate GPU-only step; it is not something the repo tries to fake locally. |
| adapter 를 얹은 생성 샘플 대량 생산 | `PASS` | The adapter-backed generation paths are still GPU-bound as intended. |
| blind eval 용 생성 단계 | `PASS` | Blind-eval generation is still correctly treated as a generation step rather than a local text transform. |

### 자동 중지 조건

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 컨테이너 restart loop | `PASS` | The pod control scripts are explicitly built to detect failure/timeouts and stop rather than spin forever. |
| 로그 무출력 상태가 비정상적으로 길다 | `PASS` | `chain_train.sh` / `run_eval.sh` both enforce timeouts and failure handling around stalled stages. |
| 잘못 설계된 eval/train 으로 판명됐다 | `STILL-FAIL` | The code does not hard-stop on goal-misaligned design issues like “blind eval not default” or “live CPT file not locally verified”; this audit had to catch those manually. |
| 결과 업로드가 끝났고 더 돌릴 이유가 없다 | `PASS` | Both train and eval shell chains explicitly stop the pod after completion. |
| 다음 단계가 GPU 없이 가능한 단계로 넘어갔다 | `PASS` | The shell chains are designed to stop at stage completion instead of idling into later local-only work. |

### 목표 달성 후 정책

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 최신 adapter 와 eval 결과가 Hugging Face / GitHub 에 내구성 있게 남아 있다 | `STILL-FAIL` | The current local env still cannot point phase6 eval at a stable adapter repo, so the “durably pinned latest adapter” story is not closed. |
| blind eval 과 raw 분포 비교에서 현재 목표 수준에 도달했다 | `STILL-FAIL` | The repo has not met blind-eval-default or raw-distribution-close criteria yet. |
| 다음 반복을 열어도 기대 이득보다 비용이 큰 상태다 | `STILL-FAIL` | There are still unresolved alignment gaps large enough to justify another iteration once the gates are corrected. |

## 승인 없이 계속 진행하는 범위

### 승인 없이 계속 진행

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| raw 분석 | `PASS` | Current tooling and this audit both do this locally without any external approval step. |
| 데이터셋 재생성 | `PASS` | The repo already regenerated and promoted a new context stream without waiting on a separate approval gate. |
| train/eval pod 재실행 | `PASS` | Launch scripts are built for repeated dry-run/real-run execution without a manual policy wall in the repo itself. |
| 실패 pod 중지 및 재시도 | `PASS` | `chain_train.sh` and `autonomous_loop.sh` both encode stop/retry behavior directly. |
| Hugging Face repo 반영 | `PASS` | The training shell still includes HF upload automation. |
| GitHub branch/pointer 업데이트 | `PASS` | The train chain still persists run artifacts back to GitHub when credentials are present. |

### 외부 하드 블로커가 생길 때만 보고

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 인증 만료 | `PASS` | No auth-expiry blocker was encountered in this local re-audit. |
| quota/결제 문제 | `PASS` | No quota/payment blocker was encountered in this local re-audit. |
| 외부 서비스 장애 | `PASS` | No external outage blocker was encountered in this local re-audit. |
| 회복 불가능한 데이터 손실 징후 | `PASS` | No irreversible data-loss signal was found in the current local artifacts. |

## 보고 원칙

| Bullet | Status | One-line evidence |
| --- | --- | --- |
| 진행 보고는 상태 변화가 있을 때마다 남긴다 | `PASS` | The repo has a strong habit of writing run-local reports under `runs/` whenever state changes. |
| "잘 되고 있다" 대신 무엇이 끝났고 무엇이 병목인지 구체적으로 적는다 | `PASS` | The better reports in this repo are concrete and stage-specific, and this re-audit follows that rule. |
| 성능 향상 주장은 항상 수치 또는 샘플 근거와 같이 적는다 | `STILL-FAIL` | The repo still has stale/inconsistent “latest” artifacts, so not every current improvement claim is anchored to one coherent dataset snapshot. |

## Bottom Line

The repo did fix the two biggest pathing failures from the prior audit:

- the active paid recipe now trains on `cpt_context_stream.jsonl`
- the default phase6 eval path can now stratify by `kind`

But the repo is still not goal-aligned enough to call this `PASS`:

- blind eval is still not the default paid evidence path
- promo/operator contamination is still materially present in the live CPT stream
- the active comment-side training signal is still distributionally far from raw because `context_comment` scaffolding is now `35.79%` of comment-side rows
- the current GPU gate is still validating stale/default datasets rather than the live CPT file actually selected by `budget30_v2`

If you want a single brutal sentence: the repo now teaches reply flow better than before, but it still does not enforce the right evidence path tightly enough to justify an unqualified “goal aligned” verdict.
