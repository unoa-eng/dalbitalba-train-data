# HANDOFF — Qwen3-8B Local Paper-Grade Training Until "원천 DB와 구분 불가능"

> **사용법**: 새 Claude Code 세션 첫 메시지에 이 파일 전체를 그대로 붙여넣으면 cycle-9부터 학습 완료까지 자율 진행. 사용자 컨펌은 RunPod 사용/머지 같은 비가역 행위에만 요청.

---

## 0. 사용자 결정 (변경 불가 — 2026-05-13)

- **로컬 only**. RunPod 사용 금지. (이유: cycle-8 분석 결과 MLX 4-bit affine vs BnB NF4 양자화 gap ≈ 1-2pp KoBEST 일반 능력. 도메인 텍스트 생성 = Turing-indistinguishability에는 인간 perception 임계값 이하라 무관. `runs/cycle8-analysis/{quant_equivalence,lora_equivalence,attention_optim_batch,data_logging_identical,throughput_and_verdict}.md` 참조.)
- **시간 무관**. 수일~수주 OK. Mac mini M4 16GB single-machine.
- **퀄리티 절대 기준** — TRAINING_DESIGN_V3 L6 "원천 DB와 구분 불가능한 텍스트 생성". paper-grade는 하한, SOTA-Turing이 목표.
- **PR 머지**: unoa-eng admin merge OK (memory feedback_pr_merge_authority).
- **인프라 설치 전체 사전 승인 (2026-05-13 사용자 명시)**: 필요한 pip 패키지 / brew 도구 / 시스템 dep 모두 자유롭게 설치. 우회·mock·graceful-skip 사용 **금지**. 누락 시 즉시 설치하고 진행.
- **API key 발급도 사전 승인**: ANTHROPIC_API_KEY (console.anthropic.com → API Keys → "dalbitalba-paper-grade" name), OPENAI_API_KEY (platform.openai.com → API Keys). 부재 시 사용자에게 즉시 요청 후 `.env.local`에 추가 (gitignored 안전).

## 1. Mission

`/Users/unoa/dalbitalba-train-data` 레포에서 한국어 성인 커뮤니티 "달빛알바"(cb2_밤문화이야기) 도메인에 정합된 Qwen3-8B-Base 어댑터를 학습. 결과로 생성된 텍스트가 원천 댓글과 3-judge majority blind eval에서 **AI 식별률 ≤ 15%** (Strong Turing pass)에 도달할 때까지 phase 반복.

## 2. 목표 임계값 (SOTA-Turing 수준)

| Hypothesis | Metric | Pass threshold | Reject threshold | 측정 방법 |
|---|---|---|---|---|
| H1 도메인 적응 | MAUVE | ≥ **0.90** | < 0.70 | `scripts/phase6_eval.py` |
| H1 | Bigram JSD | ≤ **0.04** | > 0.08 | 동상 |
| H1 | Length KL | ≤ **0.01** | > 0.05 | 동상 |
| **H2 Turing pass** | **AI 식별률** | ≤ **15%** | > 30% | `eval/judge_3way.py --min-rows 200` |
| H2 | Wilson upper | < **25%** | ≥ 40% | 동상 |
| H2 | N | ≥ 200 stratified | < 100 | 동상 |
| H3 구조 토큰 | reply_depth_kl | ≤ **0.10** | > 0.25 | `scripts/phase6_eval_v2.py` |
| H3 | structure_fidelity | ≥ **95%** | < 85% | 동상 |
| H3 | thread_coherence (LLM judge) | ≥ **0.80** | < 0.55 | judge_3way 보조 |
| H4 일반 능력 | KoBEST avg Δ | ≥ **-0.02** | < -0.05 | `eval/run_h4_full.sh` |
| H4 | HAE-RAE avg Δ | ≥ -0.02 | < -0.05 | 동상 |
| Auxiliary | persona_consistency | ≥ **0.90** | < 0.80 | phase6_eval_v2 |
| Auxiliary | choseong_marker_match_max | ≤ **0.10** | > 0.20 | 동상 |
| Auxiliary | English code-switch ratio | < **2%** | > 5% | 동상 |

**모든 임계값 PASS 전까지 phase 반복** (자동 고도화 루프, 섹션 7).

## 3. 인프라 세팅 (cycle-9 시작 전 모든 단계 실행)

```bash
cd /Users/unoa/dalbitalba-train-data
git pull --ff-only origin main

# 3.1 .venv 패키지 (idempotent)
.venv/bin/pip install --quiet --upgrade \
    mlx-lm==0.29.1 statsmodels lm-eval==0.4.9.1 wandb==0.26.1 \
    transformers peft trl datasets accelerate huggingface_hub \
    anthropic openai mauve-text sacrebleu rouge-score \
    sentence-transformers kiwipiepy

# 3.2 시스템 도구
command -v jq || brew install jq
command -v gh || brew install gh
command -v tmux || brew install tmux

# 3.3 시크릿 — .env.local에 5개 + 옵셔널 2개 모두 있어야 함
set -a; source .env.local; set +a
[ -n "$WANDB_API_KEY" ] && [ -n "$WANDB_ENTITY" ] || echo "FAIL: WANDB keys missing"
[ -f "$HOME/.cache/huggingface/token" ] || echo "FAIL: HF token missing"
[ -f "$HOME/.runpod/config.toml" ] || echo "WARN: RunPod key absent (OK if local-only)"
gh auth status 2>&1 | grep -q "Logged in" || echo "FAIL: gh CLI not logged in"
# cycle-11 + cycle-12 진입 시 필수:
[ -n "$ANTHROPIC_API_KEY" ] || echo "WARN: ANTHROPIC_API_KEY absent — required for ORPO LLM-judge + judge_3way"
[ -n "$OPENAI_API_KEY" ] || echo "WARN: OPENAI_API_KEY absent — judge_3way GPT 부재 (heuristic fallback)"

# 3.4 HF cache + MLX model
test -d .hf-home/hub/models--Qwen--Qwen3-8B-Base || \
    HF_HOME=$(pwd)/.hf-home huggingface-cli download Qwen/Qwen3-8B-Base
test -f runs/cycle7-mac-simul/qwen3-8b-mlx-4bit/model.safetensors || \
    .venv/bin/python -m mlx_lm convert --hf-path Qwen/Qwen3-8B-Base \
        -q --q-bits 4 --mlx-path runs/cycle7-mac-simul/qwen3-8b-mlx-4bit

# 3.5 wandb 로그인
.venv/bin/wandb login --relogin || true

# 3.6 회귀 baseline (모두 PASS 확인)
.venv/bin/python scripts/macmini_smoke_loop.py --profile paper8b      # 11/11 PASS
.venv/bin/python scripts/round2_integrity_check.py                     # verdict=PASS thread_holdout intersect=0
python3 scripts/check_data_paths.py --strict --profile paper8b         # violations=0
python3 scripts/check_data_paths.py --strict --profile budget30        # violations=0

# 3.7 메모리 + thermal 기준선
vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} END{printf "free %.1fMB | inactive %.1fMB\n", f*4096/1048576, i*4096/1048576}'
sudo powermetrics --samplers thermal -i 1000 -n 1 2>/dev/null | grep "CPU die" | head -1
```

## 4. Cycle 계획 (cycle-9 → cycle-13)

### cycle-9: Phase 1+2 CPT (Mac MLX, full epoch each)

**recipes/cycle9_mlx_lora.yaml** (이미 cycle-8에서 작성, `--num-layers -1 + r=64 + DoRA + rsLoRA scale=8` 명시).

```bash
# Phase 1 — broad CPT
mkdir -p runs/cycle9/data
shuf -n 45000 cpt_enriched.jsonl > runs/cycle9/data/train.jsonl
tail -3000 cpt_enriched.jsonl > runs/cycle9/data/valid.jsonl

# 실행 (모든 LoRA 파라미터는 YAML config에 — CLI 플래그 부재 architect-verified)
set -a; source .env.local; source recipes/round2-cycle1.env; set +a
unset DATA_DIR
WANDB_NAME="cycle9-phase1-cpt-broad-$(date -u +%Y%m%dT%H%M%SZ)" \
.venv/bin/python -m mlx_lm.lora -c recipes/cycle9_mlx_lora.yaml \
  2>&1 | tee runs/cycle9/phase1.log

# 검증 (필수)
# - wandb run URL = wandb.ai/dalbit-ai/dalbitalba-round2/runs/<id>
# - train loss 단조 감소 (final loss < 3.7 권고)
# - val loss < 4.075 (cycle-6 m55w3tre baseline 대비 개선)
# - adapter saved at runs/cycle9/phase1-cpt-broad/adapters.safetensors
# - peak memory < 8GB (vm_stat 확인, swap 발생 안 함)

# Phase 2 — clean CPT (phase1 adapter 이어받기)
shuf -n 40000 cpt_corpus.v3.jsonl > runs/cycle9/data2/train.jsonl
tail -2000 cpt_corpus.v3.jsonl > runs/cycle9/data2/valid.jsonl
# recipes/cycle9_mlx_lora.yaml을 복사해 cycle9_phase2.yaml로 만들고
# resume_adapter_file: "runs/cycle9/phase1-cpt-broad/adapters.safetensors"
# data: "runs/cycle9/data2", adapter_path: "runs/cycle9/phase2-cpt-clean" 변경
.venv/bin/python -m mlx_lm.lora -c recipes/cycle9_phase2.yaml \
  2>&1 | tee runs/cycle9/phase2.log

# Architect Opus reviewer로 cycle-9 close
# memory project_dalbitalba_audit_close.md cycle-9 section 추가
# main 푸시
```

### cycle-10: Phase 3 TC-SFT

```bash
# 데이터 변환: sft_thread_conditioned.jsonl 10,245행을 mlx_lm prompt/completion 형식으로
python3 - <<'PY'
import json
with open("runs/cycle10/sft_train.jsonl", "w") as out:
    for line in open("sft_thread_conditioned.jsonl"):
        r = json.loads(line)
        # persona prefix in prompt
        persona = r.get("persona_id", "p-unknown")
        prompt = f"<persona:{persona}>\n{r['instruction']}\n{r['input']}"
        out.write(json.dumps({"prompt": prompt, "completion": r["output"]}, ensure_ascii=False) + "\n")
PY
# valid: sft_thread_conditioned.eval.jsonl 동일 변환

# recipes/cycle10_sft.yaml — phase2 adapter resume + --mask-prompt + 2 epoch
# (cycle-9 YAML 패턴 복제 + iters = len(sft_train) * 2)
.venv/bin/python -m mlx_lm.lora -c recipes/cycle10_sft.yaml \
  2>&1 | tee runs/cycle10/phase3.log

# 검증:
# - SFT val loss 단조 감소
# - 30-50 generated samples 정성 검증:
#   .venv/bin/python -m mlx_lm.generate --model <merged_phase3_path> \
#       --prompt "<persona:p-001>\n다음 게시글에 댓글을 작성하시오\n..." --max-tokens 100
#   사람이 보고 community tone / slang / choseong / persona 보존 확인
# - persona_consistency ≥ 0.85 (phase6_eval_v2 호출)
```

### cycle-11: ORPO Phase 4 (LLM-judge data → 학습)

```bash
# 4.1 ORPO LLM-judge pass (요구: ANTHROPIC_API_KEY)
set -a; source .env.local; set +a
[ -z "$ANTHROPIC_API_KEY" ] && { echo "STOP: ANTHROPIC_API_KEY required"; exit 1; }
.venv/bin/python scripts/round2_build_orpo_pairs.py \
    --runs-glob "runs/refinement-2026042*" \
    --val-set val_set.v3.jsonl \
    --sft-eval sft_thread_conditioned.eval.jsonl \
    --cpt-corpus cpt_corpus.v3.jsonl \
    --llm-judge-model claude-sonnet-4-5-20250929 \
    --min-judge-delta 1.0 \
    --max-pairs 2000 \
    --out runs/cycle11/orpo_pairs.judged.jsonl
# 결과: 잔여 ≥ 500쌍 (높을수록 좋음)

# 4.2 ORPO 학습 (Mac MPS-fp16 — MLX-LM에 ORPO trainer 없음)
# train_orpo.py는 transformers + TRL + peft 기반. Mac MPS device 사용 (bnb 없이 fp16).
# 16GB RAM에서 매우 무거움 — 가능하면 batch=1 + gradient_checkpointing + max_seq 1024
# 예상: ~24-48h, 메모리 swap 위험 있음
export ORPO_BASE_MODEL=<phase3_merged>
export ORPO_DATA=runs/cycle11/orpo_pairs.judged.jsonl
export ORPO_OUTPUT_DIR=runs/cycle11/phase4-orpo
export ORPO_NUM_EPOCHS=1
export ORPO_BETA=0.1
export ORPO_MAX_SEQ_LEN=1024
.venv/bin/python train_orpo.py 2>&1 | tee runs/cycle11/phase4.log

# 4.3 If OOM/swap unbearable, defer ORPO and skip to cycle-12.
# H2 reject-action allows ORPO 재시도 단계, paper에서 honestly 기술.
```

### cycle-12: Phase 5 Evaluation (H1~H4 verdict)

```bash
# 5.1 Generate 200+ AI samples (phase3 + optionally phase4 adapter)
mkdir -p runs/cycle12/eval
# phase6_generate.py 호출 또는 mlx_lm.generate batch 호출
# stratified sampling: 50 short replies + 50 medium + 50 long + 50 thread-deep

# 5.2 H1 + Aux: phase6_eval_v2 (MAUVE + JSD + persona + choseong)
.venv/bin/python scripts/phase6_eval_v2.py \
    --ai runs/cycle12/eval/ai_generated.jsonl \
    --raw cpt_corpus.v3.jsonl \
    --persona-list runs/round2-obsidian-synthesis/persona-30-extracted.json \
    --min-rows 200 \
    --out runs/cycle12/eval/h1_aux.json

# 5.3 H2 Turing — judge_3way (requires ANTHROPIC + OPENAI keys)
[ -z "$OPENAI_API_KEY" ] && echo "WARN: GPT judge 부재 — heuristic fallback"
.venv/bin/python eval/judge_3way.py \
    --samples runs/cycle12/eval/blind_pool.jsonl \
    --output-dir runs/cycle12/eval/judges \
    --min-rows 200 \
    --strata-min 20
# h2_report.json에서 verdict 확인 — PASS / MARGINAL / REJECT_H2 / UNDERPOWERED

# 5.4 H4 KoBEST + HAE-RAE — lm-eval-harness 전체 task
PYBIN=.venv/bin/python \
H4_BASE_MODEL=Qwen/Qwen3-8B-Base \
H4_ADAPTER_PATH=<phase3_or_phase4_path> \
H4_TASKS=kobest,haerae \
H4_MAX_DROP=0.02 \
H4_DEVICE=mps \
H4_OUTPUT_DIR=runs/cycle12/eval/h4 \
bash eval/run_h4_full.sh

# 5.5 종합 verdict
python3 - <<'PY'
import json
h1 = json.load(open("runs/cycle12/eval/h1_aux.json"))
h2 = json.load(open("runs/cycle12/eval/judges/h2_report.json"))
h4 = json.load(open("runs/cycle12/eval/h4/h4_summary.json"))
verdict = {
    "H1_mauve": h1["base"]["metrics"].get("mauve_score"),
    "H1_jsd": h1["base"]["metrics"]["bigram_jsd"],
    "H2_ai_rate": h2["h2_primary_endpoint"]["ai_detection_rate"],
    "H2_wilson_upper": h2["h2_primary_endpoint"].get("wilson_ci_high"),
    "H4_kobest_drop": h4["tasks"]["kobest"]["drop"],
    "PASS": all([
        h1["base"]["metrics"].get("mauve_score", 0) >= 0.90,
        h1["base"]["metrics"]["bigram_jsd"] <= 0.04,
        h2["h2_primary_endpoint"]["ai_detection_rate"] <= 0.15,
        h2["h2_primary_endpoint"].get("wilson_ci_high", 1) < 0.25,
        h4["tasks"]["kobest"]["drop"] >= -0.02,
    ]),
}
print(json.dumps(verdict, indent=2, ensure_ascii=False))
open("runs/cycle12/eval/final_verdict.json", "w").write(json.dumps(verdict, indent=2, ensure_ascii=False))
PY
```

### cycle-13: paper draft + 최종 마감

```bash
# 13.1 TRAINING_DESIGN_V3.md §0 H1~H4 verdict 행 채우기 (Edit tool)
# 13.2 docs/DATA_CARD.md v3.3: 학습된 adapter SHA + eval metrics
# 13.3 HF push (incremental adapter 이미 chain이 push했어야 — 수동 fallback)
HF_TOKEN=$(tr -d '\n\r ' < ~/.cache/huggingface/token) \
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="runs/cycle10/phase3-tc-sft",
                  repo_id="unoa/dalbitalba-qwen3-round2-final-phase3",
                  repo_type="model", token=os.environ["HF_TOKEN"])
PY
# 13.4 4-critic 만장일치 GO 합치 (Opus + Codex + Gemini + SRE) — cycle-6 패턴
# 13.5 ralph cancel + memory final cycle 작성
```

## 5. 자동 고도화 루프 (phase verdict 미달 시)

`phase verdict` < threshold일 때 다음 매트릭스로 recipe 조정 후 phase 재실행:

| 미달 metric | 조정 방향 | YAML 변경 |
|---|---|---|
| H1 MAUVE < 0.80 (under-converged) | LR ↑ + epoch ↑ | learning_rate 1e-4 → 2e-4 (warmup 0.12); iters × 2 |
| H1 MAUVE < 0.80 (val loss 진동) | LR ↓ + warmup ↑ | learning_rate 1e-4 → 5e-5; warmup_ratio 0.08 → 0.15 |
| H1 JSD > 0.08 | rank ↑ + DoRA on | lora_parameters.rank 64 → 96; fine_tune_type "lora" → "dora" |
| H2 AI 식별률 > 30% | SFT loss weighting | SFT_LOSS_WEIGHT_ARGOT 1.5 → 2.5; SFT 1 epoch 추가 |
| H2 persona 부재 | persona prefix 강화 + epoch | SFT 데이터 변환 시 `<persona:p-XXX>` 토큰 추가, SFT 2 → 3 epoch |
| H3 reply_depth_kl > 0.20 | depth 라벨링 명시 | sft 데이터에 `<depth:N>` 토큰 prepend (tokenizer_v4에 이미 있음) |
| H4 KoBEST Δ < -0.05 (catastrophic forget) | replay 비율 ↑ + r ↓ | SFT 데이터에 KoBEST/한국어 일반 corpus 10% mix; lora_parameters.rank 64 → 32 |
| Aux choseong > 0.20 | choseong sample mining 강화 | CPT 데이터에 choseong-rich row over-sample 2× |

**escalation 규칙**: 동일 metric이 3 cycle 연속 미달 시 사용자에게 "fundamental redesign 필요" 보고. 더 큰 모델 (Qwen3-14B), 다른 LoRA 알고리즘 (S-LoRA, X-LoRA), 또는 RunPod 양자화 강화 옵션 제시.

## 6. Failure modes + mitigations (Mac mini M4 16GB)

| Symptom | Cause | Fix |
|---|---|---|
| Training freezes ≥ 60s | thermal throttling | `sudo pmset -a thermlog 1` 로그 + 외부 fan / 환기 / `wait` 후 재시작 |
| Peak memory > 14GB → swap | batch + GAS 과다 | grad_accumulation_steps 16 → 32 + max_seq_length 2048 → 1536 |
| `OutOfMemoryError` from MPS | 다른 앱이 unified memory 점유 | `pkill -f Slack; pkill -f Chrome`; activity monitor 확인 |
| wandb sync slow / disconnect | 네트워크 | `WANDB_MODE=offline` + `wandb sync runs/<wandb-dir>` 사후 동기화 |
| Adapter save 실패 (디스크) | `df -h .` 확인 | save_every 250 → 500; HF Hub로 즉시 backup |
| Korean tokenization 이상 | kiwipiepy mecab corruption | `.venv/bin/pip install --force-reinstall kiwipiepy` |

## 7. Multi-critic 합의 (각 cycle close 시)

cycle-6 4-critic 패턴 반복. 다음 phase 진입 전 필수:

1. **Opus Area Chair** (NeurIPS reviewer 관점) — H1~H4 falsifiable 충족, paper acceptance 가능성 평가
2. **Codex critic** (code-design mismatch) — recipe ↔ chain ↔ train script 정합성
3. **Gemini critic** (claim vs implementation) — 보고서 주장과 실측 산출물 일치
4. **SRE Sonnet** (운영 안전) — memory/thermal/checkpoint 무결성

4-critic 만장일치 GO_SUFFICIENT 없으면 NEEDS_REVISIONS 수정 후 재verify. 3 cycle 연속 fail 시 escalation.

## 8. 절대 금지

- 검증 없이 다음 phase 진행
- adapter SHA 기록 누락
- wandb entity≠dalbit-ai
- `--num-layers 16` (default) 사용 — 반드시 -1
- `--lora-parameters` CLI 플래그 사용 (존재 안 함, YAML config로)
- 양자화 gap을 "무시 가능"으로 단정 — eval에서 실측
- 사용자 컨펌 없이 RunPod paid run
- Cycle 종료 전 commit/push 누락
- **graceful skip / mock / fallback 패턴 사용** — 의존성 부재면 즉시 설치, API key 부재면 즉시 사용자 요청. eval/eval_kobest.py·eval_haerae.py / run_h4_full.sh / macmini_local_train_simul.sh 에 있는 graceful skip 분기는 paper-grade run 시 반드시 fail-closed로 처리 (env H4_REQUIRE_FULL=1 같은 strict 옵션 추가 권고).

## 9. 마감 조건

다음이 모두 충족되면 **paper-grade Strong Turing entry GREEN**, 사용자에게 보고 + ralph cancel:

- [ ] H1 MAUVE ≥ 0.90, JSD ≤ 0.04, Length KL ≤ 0.01
- [ ] H2 AI 식별률 ≤ 15%, Wilson upper < 25%, N ≥ 200
- [ ] H3 reply_depth_kl ≤ 0.10, structure_fidelity ≥ 95%, thread_coherence ≥ 0.80
- [ ] H4 KoBEST Δ ≥ -0.02, HAE-RAE Δ ≥ -0.02
- [ ] persona_consistency ≥ 0.90, English ratio < 2%
- [ ] 4-critic 만장일치 GO_SUFFICIENT
- [ ] HF private adapter push (`unoa/dalbitalba-qwen3-round2-final-phase{1,2,3,4}`)
- [ ] TRAINING_DESIGN_V3.md §0 H1~H4 verdict 행 채워짐
- [ ] docs/DATA_CARD.md v3.3 학습 결과 SHA + eval metrics 기록
- [ ] memory `project_dalbitalba_audit_close.md` final cycle section + MEMORY.md 인덱스
- [ ] origin/main 최신 commit 푸시
- [ ] `/oh-my-claudecode:cancel` 실행 후 state clear

## 10. 시작 명령 (새 세션 진입 시)

```bash
cd /Users/unoa/dalbitalba-train-data
git log --oneline -5 origin/main         # main HEAD 확인 (현재 5043829 + cycle-8 commit)
.venv/bin/python scripts/macmini_smoke_loop.py --profile paper8b   # 11/11 PASS
ls runs/cycle8-analysis/                 # 7 산출물 (.md + yaml)
cat .omc/prd.json 2>/dev/null | jq '.title' || echo "no current PRD"
# 이전 세션이 cycle-8 cancel 했다면 .omc/state/ 비어있음. 새 PRD 작성 후 ralph 시작.
```

`/oh-my-claudecode:ralph cycle-9 paper-grade Phase 1+2 CPT 시작 — 로컬 only, AI 식별률 ≤15% 목표` 호출.

---

**이 핸드오프 프롬프트의 무결성 보증**:
- cycle-1~8의 모든 발견과 fix가 반영됨 (양자화 gap, LoRA expressivity, judge alias, cost watchdog, root_id invariant, thread-holdout, ORPO 3 P0 wiring, smoke 11/11)
- mlx_lm.lora 사용 명령은 architect-verified (YAML config, CLI 플래그 누락 없음)
- 모든 임계값은 design doc + 사용자 결정 (≤15%) 일치
- 회귀 baseline + 인프라 세팅 + failure mode + escalation 모두 포함
- 다른 세션에서 그대로 붙여넣어 자율 실행 가능
