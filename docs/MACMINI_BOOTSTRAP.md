# Mac mini Bootstrap

이 문서는 Mac mini에서 paid-GPU 직전까지의 검증과 RunPod launch 절차를 재현하기 위한 단계입니다.

## 0. 사전 조건 + Mac mini 의 역할 명시

**Mac mini 는 운영 콘솔이지 학습/추론 머신이 아니다.**
- 학습·생성·평가 inference 는 모두 RunPod L40S 1장에서 실행. Mac mini 에서 base model (Qwen3-8B) 적재 시도 X.
- Mac mini 가 다루는 모델 관련 작업은 **safetensors 헤더 파싱, adapter_config.json 검증, HF API 호출** 까지만. 16GB RAM 이면 충분.
- "로컬 모델"을 굴려서 비교하지 않는 이유: ① QLoRA 4bit 학습은 NVIDIA bitsandbytes 의존, ② Apple Silicon 에서 같은 모델을 재현 추론해도 RunPod 런타임 디버깅에 의미 없음, ③ 비용 절감 목적과 충돌.

전제:
- macOS 14 이상 (Apple Silicon Mac mini 권장, 16GB RAM 이상)
- 인터넷 연결 (HuggingFace, GitHub, RunPod API 호출용)
- 디스크 여유 30GB+ (HF 토큰 캐시·adapter snapshot)
- 본 레포 GitHub PAT, HuggingFace write token, RunPod API key

## 1. 시스템 도구 설치

```bash
# 1) Homebrew (없으면)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2) 핵심 CLI
brew install git python@3.11 gh jq tmux
brew install --cask docker  # 옵션, 향후 컨테이너 검증용

# 3) GitHub CLI 로그인
gh auth login   # PAT 또는 device flow
```

## 2. 레포 클론 + 브랜치 체크아웃

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/unoa-eng/dalbitalba-train-data.git
cd dalbitalba-train-data
git checkout main
git pull
```

## 3. Python 가상환경

```bash
cd ~/projects/dalbitalba-train-data
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 로컬 검증에 필요한 최소 패키지 (GPU 학습용 패키지는 불필요)
pip install \
  "huggingface_hub>=0.30,<1.0" \
  "transformers>=4.51.3,<5" \
  "tokenizers>=0.21,<0.22" \
  "datasets>=2.21" \
  "safetensors>=0.4.3"
```

> Mac mini 에서 직접 학습은 안함. GPU pod 발사 전 검증만.
> `transformers`/`tokenizers` 는 토큰 분석을 위해 필요.

## 4. 환경변수

```bash
cp .env.local.example .env.local
# 또는 새로 생성:
cat > .env.local <<'EOF'
RUNPOD_API_KEY=...
HF_TOKEN=hf_...
HF_USERNAME=UNOA
GITHUB_TOKEN=ghp_...
GITHUB_REPO=unoa-eng/dalbitalba-train-data
NTFY_TOPIC=dalbit_ai_alert
WANDB_API_KEY=
WANDB_PROJECT=dalbitalba-v2
BASE_MODEL=Qwen/Qwen3-8B-Base
GPU_TYPE="NVIDIA L40S,NVIDIA A100 80GB PCIe,NVIDIA RTX 6000 Ada Generation"
CONTAINER_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
EOF
chmod 600 .env.local
```

`.env.local` 은 gitignored. 절대 commit/push 금지.

## 5. 데이터 무결성 (Mac mini 측에서 한 번 더 확인)

```bash
ls -la cpt_corpus.v2.jsonl sft_pairs.v2.jsonl val_set.v2.jsonl
md5sum *.v2.jsonl   # 또는 macOS: md5 *.v2.jsonl
```

기대 값:
- cpt_corpus.v2.jsonl  92,407 줄
- sft_pairs.v2.jsonl   78,119 줄
- val_set.v2.jsonl      3,101 줄

## 6. Local Verifier — 무료 게이트

```bash
python scripts/local_verification_loop.py --strict --profile paper8b
```

기대 결과:
- `Verdict: PASS`
- `Severe: 0`
- `Warnings: 0`
- local smoke 포함: `tokenizer_v4` 기준 SFT 포맷/마스킹 smoke PASS

이 local smoke 는 tokenizer-only control-plane 검증이다.
- 8B base model 로드 안 함
- 로컬 학습 안 함
- RunPod paid smoke 대체 아님

Obsidian 범위 정책은 `docs/OBSIDIAN_SCOPE_POLICY.md` 를 따른다. 로컬 verifier 는
`research/obsidian-ref`, `research/obsidian-export`,
`runs/round2-obsidian-synthesis/persona-30-extracted.json` 의 존재와 persona-30
coverage 를 함께 확인한다.

Mac mini에서 전체 control-plane smoke 루프를 한 번에 남기려면:

```bash
python scripts/macmini_smoke_loop.py --profile paper8b
```

산출물은 `runs/macmini-smoke-<timestamp>/summary.json` 및 `report.md`에 기록된다.

```bash
# 추가 — 기존 0618 HF artifact 검증 (release-worthy 아님 확정)
python scripts/local_verification_loop.py --strict \
  --hf-cpt-repo UNOA/dalbitalba-qwen3-cpt-20260424-0618 \
  --hf-sft-repo UNOA/dalbitalba-qwen3-sft-20260424-0618
```

## 7. Dry-run 으로 paid pod payload 검증 (실 발사 X)

```bash
# smoke.env 검증 (CPT/SFT 각 20step)
set -a; source recipes/smoke.env; set +a
python scripts/launch_train_pod.py --dry-run | jq '.imageName, .gpuTypeIds, .env | keys' | head -40
unset $(grep -oE '^[A-Z_]+=' recipes/smoke.env | tr -d '=')

# paper8b full run 검증 (CPT 2단계 + SFT + phase6 eval)
set -a; source recipes/round2-cycle1.env; set +a
python scripts/launch_train_pod.py --dry-run | jq '.imageName, .gpuTypeIds, .env.BUDGET_PROFILE, .env.CPT_PHASE_1_DATA, .env.CPT_PHASE_2_DATA, .env.SFT_NUM_EPOCHS, .env.CPT_TIMEOUT_HOURS, .env.SFT_TIMEOUT_HOURS'
```

기대:
- `imageName` = `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- `gpuTypeIds[0]` = `NVIDIA L40S`
- paper8b: `BUDGET_PROFILE=paper8b`, `CPT_PHASE_1_DATA=cpt_enriched.jsonl`, `CPT_PHASE_2_DATA=cpt_corpus.v3.jsonl`, `SFT_NUM_EPOCHS=2`, `BUDGET_CAP_USD=60`
- smoke: `CPT_LIMIT_ROWS=512`, `CPT_MAX_STEPS=20`, `SFT_MAX_STEPS=20`, `MAUVE_DISABLED=1`

## 8. 첫 paid run — smoke (배관 검증)

`docs/LOCAL_VERIFICATION_LOOP.md` 의 Promotion Rule 통과 후에만:

실제 RunPod launch 는 선택한 Git ref 를 클론한다. `scripts/launch_train_pod.py`
와 `scripts/launch_eval_pod.py` 는 런타임 중요 파일이 로컬에서 dirty 이면
실 launch 를 거부한다. 로컬 검증이 통과한 변경은 먼저 commit/push 후 실행한다.

```bash
set -a; source recipes/smoke.env; set +a
python scripts/launch_train_pod.py
# 폰 ntfy 로 진행 이벤트 수신 — preflight OK / CPT start / SFT done / HF upload
# 예상 시간: ~30-40분, 비용 ~$0.5 (worst-case ~$5.5; 그 이상이면 즉시 §12 비상 정지)
```

smoke 가 끝나면 `runs/train-run-<stamp>/` 브랜치에 `DONE.txt`, `manifest.json`,
`*.log` 가 푸시된다. round2 chain 은 `runs/latest-round2-train.json`, classic chain 은
`runs/latest-train.json` 포인터를 업데이트한다.

## 8.5. PROMOTION 게이트 (smoke → paper8b 진입 전 필수)

```bash
git fetch origin
git checkout origin/main -- runs/latest-train.json runs/latest-round2-train.json
python3 scripts/check_smoke_promotion.py --require-sft
# 종료코드 0 = PROMOTE, 1 = HOLD, 2 = USAGE
```

기대:
- classic chain: `latest-train status=done_ok`, `hf_cpt`, `hf_sft`
- round2 chain: `latest-round2-train status=done_ok`, `hf_repo_round2`, `eval/phase5-eval-v2.json`
- `DONE.txt = done_ok`, `manifest.json present`

`HOLD` 가 떨어지면 paper8b 절대 launch 금지. 실패 사유 모두 해결 후 재시도.

## 9. 본 run — paper8b full pipeline

8.5 PROMOTE 통과 후에만:

```bash
set -a; source recipes/round2-cycle1.env; set +a
python scripts/launch_train_pod.py
# 예상 train 시간: ~35h, 예상 train 비용: ~$28
# timeout hard-cap exposure: ~61h / ~$48 under BUDGET_CAP_USD=60
# 기능 축소 없음: CPT phase1/2 + SFT + integrated phase6 eval 유지
```

## 10. 평가 (eval pod)

CPT 완료 후:

```bash
export SFT_ADAPTER_REPO=UNOA/dalbitalba-qwen3-sft-<stamp>
export CPT_MERGED_REPO=UNOA/dalbitalba-qwen3-cpt-<stamp>
python scripts/launch_eval_pod.py
```

## 11. 트러블슈팅

| 증상 | 해결 |
|------|------|
| `python: command not found` | `python3` 사용. venv 활성화 확인 |
| `gh: not authenticated` | `gh auth login` 재실행 |
| `Permission denied` for `.env.local` | `chmod 600 .env.local` |
| pod 발사 실패 `RunPod API HTTP 401` | `RUNPOD_API_KEY` 재발급 |
| pod 발사 실패 `HTTP 400 gpuType...` | `GPU_TYPE` 가용성 확인, 다른 community pool |
| HF upload 실패 | `HF_TOKEN` write 권한 확인 |
| 로컬 verifier 가 PII 신호 보고 | 데이터 재정제 필요 |

## 11.5. Hard ceiling 매트릭스 (0618 식 burn 재발 방지)

신설계의 **단일 사이클 최대 손실** (수학적 상한, GPU=L40S $0.79/hr):

| 단계 | hard cap | 누적 |
|---|---|---|
| smoke (worst-case 7h) | $5.53 | $5.53 |
| **§8.5 PROMOTION 게이트 — HOLD 시 paper8b launch 거부** | — | — |
| paper8b CPT phase1+2 40h | $31.60 | $37.13 |
| paper8b merge 3h | $2.37 | $39.50 |
| paper8b SFT 10h | $7.90 | $47.40 |
| paper8b integrated eval 6h | $4.74 | $52.14 |
| paper8b HF upload 2h | $1.58 | **$53.72 ceiling** |

`BUDGET_CAP_USD=60` 아래에서 기능 손실 없이 full pipeline 을 실행한다. 현재
추정 train 비용은 약 `$28`, timeout hard-cap exposure 는 약 `$48` 이다.

만약 paper8b 의 5-metric gate 가 fail 하면 **§14 rollback playbook 진입 — 같은 recipe 로 재시도 절대 금지.**

## 12. Stop / 비상 정지

```bash
# 모든 활성 pod 정지
python3 -c "
import os, json, urllib.request
for line in open('.env.local'):
    if '=' in line and not line.startswith('#'):
        k,_,v=line.partition('='); os.environ.setdefault(k.strip(), v.strip().strip('\"'))
key=os.environ['RUNPOD_API_KEY']
r=urllib.request.urlopen(urllib.request.Request('https://rest.runpod.io/v1/pods', headers={'Authorization': f'Bearer {key}'}), timeout=30)
for p in json.loads(r.read()):
    if p.get('desiredStatus')=='RUNNING':
        pid=p['id']
        urllib.request.urlopen(urllib.request.Request(f'https://rest.runpod.io/v1/pods/{pid}/stop', method='POST', headers={'Authorization': f'Bearer {key}'}), timeout=15)
        print('stopped', pid)
"
```

## 13. 작업 종료 체크리스트

- [ ] local verifier `PASS`, `Severe: 0`, `Warnings: 0` (`--profile paper8b`)
- [ ] dry-run payload 정상
- [ ] L40S 가용성 + RunPod 잔고 확인 (`check_l40s_availability.py`)
- [ ] smoke run DONE.txt + HF adapter 확인
- [ ] PROMOTION 게이트 PROMOTE (`check_smoke_promotion.py --require-sft`)
- [ ] adapter 무결성 OK (`check_adapter_integrity.py`)
- [ ] paper8b run DONE.txt + HF adapter 확인
- [ ] eval pod `eval/phase5-eval-v2.json` 또는 classic `metrics.json` 생성
- [ ] 5-metric gate PASS (jsd ≤0.15, length_kl ≤0.10, digit/eng delta, mauve ≥0.80)
- [ ] PR 생성 (`scripts/create_final_pr.sh`)
- [ ] 비활성 pod 정리 (위 12번 스크립트)

## 14. paper8b 의 5-metric gate FAIL 시 rollback playbook

**같은 recipe 로 절대 재시도 금지.** 동일 데이터·동일 hparam 으로 두 번째 시도해도 결과는 동일하고 또 ~$25 burn 한다 (0618 → 0424 식 누적 burn 의 패턴).

세 가지 분기 중 하나 선택:

### 분기 A — 데이터 재정제 후 재시도
적용 조건: eval gate fail 의 원인이 **데이터 품질** (high duplicate rate, 짧은 행 비율 과다) 일 때
```bash
# 1) 원본 크롤에서 더 엄격한 정제로 재생성
python3 scripts/phase1_data_pipeline.py --min-chars 50 --dedup-threshold 0.85
# 2) verifier 재실행 — duplicate_rate < 0.30 이어야 함
python3 scripts/local_verification_loop.py --strict --profile paper8b
# 3) smoke + paper8b 재실행
```

### 분기 B — 학습 강도 상향 후 재시도
적용 조건: gate fail 의 원인이 **under-fit** (loss 감소 추세 좋았지만 1 epoch 부족) 일 때
```bash
# 사용자 동의 필수 — budget cap 상향이므로 명시적 결정
cp recipes/round2-cycle1.env recipes/paper8b-strong.env
sed -i.bak -e 's/SFT_NUM_EPOCHS=2/SFT_NUM_EPOCHS=3/' \
           -e 's/BUDGET_CAP_USD=60/BUDGET_CAP_USD=90/' recipes/paper8b-strong.env
set -a; source recipes/paper8b-strong.env; export BUDGET_PROFILE=paper8b; set +a
python3 scripts/launch_train_pod.py
```

### 분기 C — partial 0618 adapter 채택 후 종료 ($0 추가 spend)
적용 조건: 시간/예산 더 못 쓰는 상황. paper8b 부분 adapter 라도 가져가서 향후 재학습용 baseline 으로 사용
```bash
# HF adapter 보존, 메모/PR 작성
gh issue create --title "dalbitalba paper8b cycle 1 partial — eval failed" \
  --body "5-metric gate fail. adapter @ UNOA/dalbitalba-qwen3-cpt-<stamp>. next cycle to be funded separately."
```

**판단 기준 (자동 추정):**
- duplicate_rate > 0.40 → 분기 A 우선
- training loss 가 step 5775 도달 + plateau 안 됨 → 분기 B 우선
- 위 둘 다 아닌 미스터리 fail → 분기 C 권장 (디버깅 비용 > 재학습 비용 가능성)

## 15. 자율 루프 / recipe mutator 범위 명시

이 부트스트랩은 **단발성 manual 운영** 전용이다. `scripts/autonomous_loop.sh` 와 `scripts/recipe_mutator.py` 는 본 가이드 사용 시 **호출하지 않는다.**
- 이유: autonomous_loop 의 budget cap 은 오래된 `$25 static estimate` 기준이라 paper8b full pipeline 의 실제 비용/timeout cap 을 반영하지 못한다.
- 자율 루프는 **$90+ 예산 프로파일** 이 별도 정의될 때만 의미 있음.
- 비개발자 운영자: §1~§14 만 따른다. autonomous_loop.sh 호출 금지.

## 16. 기계적 방어선 요약 (0618 식 의미없는 학습 재발 방지)

**구조적으로 차단된 사고 패턴:**

| 사고 패턴 | 차단 메커니즘 | 우회 가능? |
|---|---|---|
| SOLAR / 비-Qwen3 base 로 학습 | `chain_train.sh:46-65` BASE_MODEL_CPT case 검사 → exit 2 | `FORCE_BASE_MODEL=1` 명시적 override 시에만 |
| paper8b + dup rate > 0.50 | `local_verification_loop.py` SEVERE → verdict=FAIL → launcher 거부 | `FORCE_LAUNCH=1` 명시적 override |
| paper8b + 비용 추정 > cap | verifier `--profile paper8b` SEVERE | 같은 override |
| paper8b launch 시 verifier 미수행 | `launch_train_pod.py` gated-profile verifier check → exit 1 | 같은 override |
| A100 ($1.19/hr) 폴백 silent overspend | `recipes/round2-cycle1.env: GPU_TYPE="NVIDIA L40S"` 단일화 | recipe 직접 편집 시에만 |
| Eval 이 비어있는 SFT repo 호출 | `check_smoke_promotion.py` HOLD → exit 1 | 사용자가 무시하고 launch_eval_pod 직접 실행 시 |
| Adapter 깨졌는데 PROMOTE | `check_adapter_integrity.py` safetensors 헤더 파싱 실패 시 FAIL | 무시 시 |
| trainer 가 step 50/5775 에서 멈춤 → PROMOTE | `check_smoke_promotion.py` global_step ≥ max_steps 검증 → HOLD | 무시 시 |
| Wrapper kill (TERM/INT) 시 pod 누수 | `chain_train.sh` graceful_abort trap → stop_pod | 시그널 우회 (kill -9) 시에만 |
| Eval pod hang | `run_eval.sh` run_timeout + trap | timeout binary 부재 시 fallback warn |
| 동일 recipe 재시도로 누적 burn | §14 rollback playbook 3분기 명시 | 사용자가 playbook 무시 시 |

**우회 가능한 모든 항목은 `FORCE_*=1` 환경변수로만 가능 — 즉, 운영자가 "지금 우회한다"는 명시적 의도를 표명해야 함. 모든 default 는 fail-closed.**

## 17. Abandoned Pod Sweeper

`abandoned-pod-sweeper`는 GitHub Actions cron(6시간마다)으로 자동 실행되며, 36시간 초과 RUNNING 상태 pod 중 `.state/train_pod_state.json`에 등록되지 않은 것을 stop한다. `NTFY_TOPIC` 시크릿이 설정되어 있으면 stale pod 발견 시 ntfy 알림도 전송된다.

수동 실행:

```bash
# 후보 확인만 (실제 stop 없음)
python3 scripts/abandoned_pod_sweeper.py --dry-run

# 24시간 기준으로 즉시 stop
python3 scripts/abandoned_pod_sweeper.py --max-age-hours 24 --verbose
```

GitHub Actions에서 수동 트리거(workflow_dispatch)로 `dry_run=true` / `max_age_hours` 값을 지정해 실행할 수도 있다. `RUNPOD_API_KEY` secret이 없으면 즉시 에러로 종료(exit 2).

**우회 시도 자체도 paid spend 0 인 단계 (verifier, promotion, integrity) 에서만 가능** — 우회 후 launch 해도 GPU 시간은 chain_train.sh / run_eval.sh 의 timeout 에 또 막힌다.
