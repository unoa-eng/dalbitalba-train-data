# Mac mini Bootstrap

이 문서는 Windows/WSL 환경에서 codex/budget30-macmini-loop 브랜치 작업이 끝난 뒤, Mac mini로 옮겨서 paid-GPU 직전까지의 모든 검증을 다시 한 번 수행하기 위한 단계입니다.

## 0. 사전 조건

- macOS 14 이상 (Apple Silicon Mac mini 권장)
- 인터넷 연결 (HuggingFace, GitHub, RunPod API 호출용)
- 디스크 여유 30GB+ (HF 토큰 캐시·모델 메타데이터용)
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
git checkout codex/budget30-macmini-loop
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
python scripts/local_verification_loop.py --strict
```

기대 결과:
- `Severe: 0`
- 5건의 WARN (duplication, short rows, $30 budget exceed) — 정상

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

# budget30.env 검증 (CPT only, $30 ceiling)
set -a; source recipes/budget30.env; set +a
python scripts/launch_train_pod.py --dry-run | jq '.imageName, .env.SKIP_SFT, .env.CPT_TIMEOUT_HOURS'
```

기대:
- `imageName` = `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- `gpuTypeIds[0]` = `NVIDIA L40S`
- budget30: `SKIP_SFT=1`, `CPT_TIMEOUT_HOURS=32`, `CPT_NUM_EPOCHS=1`, `SFT_NUM_EPOCHS=0`
- smoke: `CPT_LIMIT_ROWS=512`, `CPT_MAX_STEPS=20`, `SFT_MAX_STEPS=20`

## 8. 첫 paid run — smoke (배관 검증)

`docs/LOCAL_VERIFICATION_LOOP.md` 의 Promotion Rule 통과 후에만:

```bash
set -a; source recipes/smoke.env; set +a
python scripts/launch_train_pod.py
# 폰 ntfy 로 진행 이벤트 수신 — preflight OK / CPT start / SFT done / HF upload
# 예상 시간: ~30-40분, 비용 ~$0.5
```

smoke 에서 다음이 모두 푸시되면 합격:
- `runs/train-run-<stamp>/` 브랜치에 `DONE.txt`, `manifest.json`, `*.log`
- HF 측 `UNOA/dalbitalba-qwen3-cpt-<stamp>` 와 `*-sft-<stamp>` 에 adapter 파일

## 9. 본 run — budget30 CPT-only

smoke 합격 시:

```bash
set -a; source recipes/budget30.env; set +a
python scripts/launch_train_pod.py
# 예상 시간: ~30시간, 비용 ~$23.36
# Stage timeout 자동 작동 (CPT 32h cap)
```

## 10. 평가 (eval pod)

CPT 완료 후:

```bash
export SFT_ADAPTER_REPO=UNOA/dalbitalba-qwen3-sft-<stamp>
# budget30 은 SFT 미수행이라 generate_samples 는 base+CPT만 사용
# CPT_MERGED_REPO 또는 CPT_MERGED_PATH 를 명시적으로 export
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

- [ ] local verifier `Severe: 0`
- [ ] dry-run payload 정상
- [ ] smoke run DONE.txt + HF adapter 확인
- [ ] budget30 run DONE.txt + HF adapter 확인
- [ ] eval pod metrics.json 생성
- [ ] PR 생성 (`scripts/create_final_pr.sh`)
- [ ] 비활성 pod 정리 (위 12번 스크립트)
