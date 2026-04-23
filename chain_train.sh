#!/usr/bin/env bash
# chain_train.sh — dalbitalba 무인 CPT→SFT→업로드→종료 파이프라인
#
# 실행 위치: RunPod pod 내부 (nohup 또는 startup script 로 실행)
# 로그: /workspace/logs/chain.log (tee 로 stdout 동시 출력)
#
# 필수 환경변수 (pod env 에 주입):
#   HF_TOKEN       — HuggingFace write token
#   HF_USERNAME    — HuggingFace username (예: unoa-labs)
#   RUNPOD_POD_ID  — RunPod pod ID (RunPod 자동 주입)
#   NTFY_TOPIC     — (선택) ntfy.sh topic, 비워두면 알림 skip
#   GITHUB_TOKEN   — (선택) train repo branch push token
#   GITHUB_REPO    — (선택) train repo name, 기본값 unoa-eng/dalbitalba-train-data
#
# 장애 전략: graceful degradation
#   - CPT 실패 → DONE.txt 에 cpt_failed 기록, exit 1
#   - SFT 실패 → CPT 결과만 HF 업로드, DONE.txt 에 sft_failed 기록
#   - HF 업로드 실패 → 3회 재시도, 그래도 실패하면 volume 보존 후 종료

set -uo pipefail
# set -e 제거: 개별 단계별로 오류 처리하여 graceful degradation 구현

# ── 디렉토리 및 로그 설정 ─────────────────────────────────────────────────────
WORKSPACE="/workspace"
LOG_DIR="${WORKSPACE}/logs"
LOG_FILE="${LOG_DIR}/chain.log"
OUT_DIR="${WORKSPACE}/out"
DATA_DIR="${WORKSPACE}/data"
CPT_OUT="${OUT_DIR}/cpt-lora"
SFT_OUT="${OUT_DIR}/sft-lora"
DONE_FILE="${OUT_DIR}/DONE.txt"
SCRIPTS_DIR="${WORKSPACE}/scripts"
REPO_CLONE_DIR="${WORKSPACE}/repo"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_DIR}"

# ── 로그 함수 (stdout + 파일 동시 기록, HF_TOKEN 마스킹) ─────────────────────
log() {
    local msg="$1"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "${ts} ${msg}" | tee -a "${LOG_FILE}"
}

# HF_TOKEN 이 로그에 노출되지 않도록 env 출력 억제
log "=========================================="
log "dalbitalba chain_train.sh 시작"
log "  POD_ID: ${RUNPOD_POD_ID:-unknown}"
log "  HF_USERNAME: ${HF_USERNAME:-unset}"
log "  NTFY_TOPIC: ${NTFY_TOPIC:-(없음, 알림 skip)}"
log "=========================================="

# ── ntfy 알림 함수 ────────────────────────────────────────────────────────────
notify() {
    local msg="$1"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -s -X POST \
            -H "Content-Type: text/plain; charset=utf-8" \
            --data-binary "${msg}" \
            "https://ntfy.sh/${NTFY_TOPIC}" \
            >> "${LOG_FILE}" 2>&1 || true
        log "[ntfy] 알림 전송: ${msg}"
    fi
}

# ── 종료 처리: pod 중지 (볼륨 보존) ─────────────────────────────────────────
stop_pod() {
    local reason="$1"
    log "[종료] 이유: ${reason}"
    notify "dalbitalba chain ${reason} — pod 중지"
    if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        log "[종료] runpodctl stop pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}" >> "${LOG_FILE}" 2>&1 || true
    else
        log "[종료] runpodctl 없음 또는 RUNPOD_POD_ID 미설정 — 수동 종료 필요"
    fi
}

# ── DONE.txt 기록 함수 ───────────────────────────────────────────────────────
write_done() {
    local status="$1"
    local extra="${2:-}"
    {
        echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "status=${status}"
        [ -n "${extra}" ] && echo "detail=${extra}"
        echo "cpt_out=${CPT_OUT}"
        echo "sft_out=${SFT_OUT}"
        echo "hf_repo=${HF_REPO:-not_uploaded}"
    } > "${DONE_FILE}"
    log "[DONE] ${DONE_FILE} 기록 완료: status=${status}"
}

persist_run_artifacts() {
    local final_status="$1"
    local stamp
    local branch
    local run_dir
    local repo_url

    if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_REPO:-}" ]; then
        log "[runs] GITHUB_TOKEN/GITHUB_REPO 미설정 — 원격 push skip"
        return 0
    fi
    if [ ! -d "${REPO_CLONE_DIR}/.git" ]; then
        log "[runs] repo clone 미존재 — 원격 push skip"
        return 0
    fi

    stamp="$(date -u '+%Y%m%d-%H%M%S')"
    branch="train-run-${stamp}"
    run_dir="${REPO_CLONE_DIR}/runs/${branch}"
    mkdir -p "${run_dir}"

    cp "${DONE_FILE}" "${run_dir}/DONE.txt"
    [ -f "${LOG_FILE}" ] && cp "${LOG_FILE}" "${run_dir}/chain.log"
    [ -f "${WORKSPACE}/train_cpt.log" ] && cp "${WORKSPACE}/train_cpt.log" "${run_dir}/train_cpt.log"
    [ -f "${WORKSPACE}/train_sft.log" ] && cp "${WORKSPACE}/train_sft.log" "${run_dir}/train_sft.log"

    cat > "${run_dir}/manifest.json" <<EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "status": "${final_status}",
  "hf_repo": "${HF_REPO:-}",
  "hf_status": "${HF_STATUS:-}",
  "pod_id": "${RUNPOD_POD_ID:-unknown}",
  "cpt_elapsed_min": "${CPT_ELAPSED:-}",
  "sft_elapsed_min": "${SFT_ELAPSED:-}",
  "total_elapsed_min": "${CHAIN_TOTAL:-}",
  "source_repo": "${GITHUB_REPO}"
}
EOF

    mkdir -p "${REPO_CLONE_DIR}/runs"
    cat > "${REPO_CLONE_DIR}/runs/latest-train.json" <<EOF
{
  "branch": "${branch}",
  "status": "${final_status}",
  "hf_repo": "${HF_REPO:-}",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

    repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
    (
        cd "${REPO_CLONE_DIR}"
        git checkout -b "${branch}" >/dev/null 2>&1
        git add "runs/${branch}" "runs/latest-train.json"
        git commit -m "train: ${branch}" >/dev/null 2>&1 || exit 0
        git push "${repo_url}" "${branch}" >/dev/null 2>&1
    )

    if [ $? -eq 0 ]; then
        log "[runs] 원격 branch push 완료: ${branch}"
        notify "dalbitalba train artifact push 완료: ${branch}"
        echo "${branch}" > "${OUT_DIR}/RUN_BRANCH.txt"
    else
        log "[runs] 원격 branch push 실패: ${branch}"
    fi
}

# ── STEP 0: 필수 환경변수 확인 ───────────────────────────────────────────────
log "[0/5] 환경 변수 확인..."
if [ -z "${HF_TOKEN:-}" ]; then
    log "[ERROR] HF_TOKEN 이 설정되지 않았습니다. HF 업로드 불가."
    write_done "env_error" "HF_TOKEN missing"
    stop_pod "env_error"
    exit 1
fi
if [ -z "${HF_USERNAME:-}" ]; then
    log "[ERROR] HF_USERNAME 이 설정되지 않았습니다."
    write_done "env_error" "HF_USERNAME missing"
    stop_pod "env_error"
    exit 1
fi

# 데이터 파일 존재 확인
for f in cpt_corpus.jsonl sft_pairs_v2.jsonl; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        log "[ERROR] 필수 데이터 파일 없음: ${DATA_DIR}/${f}"
        write_done "data_missing" "${f}"
        stop_pod "data_missing"
        exit 1
    fi
done
log "[0/5] 환경 확인 완료. 데이터 파일 확인 완료."

# ── STEP 1: 패키지 설치 ──────────────────────────────────────────────────────
log "[1/5] 필수 패키지 설치..."
pip install -q --no-cache-dir --upgrade \
    "transformers==4.44.2" \
    "peft==0.12.0" \
    "bitsandbytes==0.43.3" \
    "datasets==2.21.0" \
    "accelerate==0.33.0" \
    "huggingface_hub>=0.24.0" \
    "safetensors>=0.4.3" \
    "sentencepiece" \
    "tokenizers>=0.19" \
    "protobuf" \
    >> "${LOG_FILE}" 2>&1

if [ $? -ne 0 ]; then
    log "[ERROR] 패키지 설치 실패"
    write_done "install_failed"
    stop_pod "install_failed"
    exit 1
fi
log "[1/5] 패키지 설치 완료."

# ── train scripts 확인 및 복사 ───────────────────────────────────────────────
# launch_chain.py 가 scripts/ 를 함께 업로드했다고 가정
# 없으면 SCRIPTS_DIR 에서 찾아 WORKSPACE 루트로 복사
python - <<'EOF' >> "${LOG_FILE}" 2>&1
import torch
import transformers
import peft
import bitsandbytes
import datasets

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("datasets", datasets.__version__)
EOF

if [ $? -ne 0 ]; then
    log "[ERROR] import smoke test failed"
    write_done "import_failed"
    stop_pod "import_failed"
    exit 1
fi

for script in train_cpt.py train_sft.py; do
    if [ ! -f "${WORKSPACE}/${script}" ]; then
        if [ -f "${SCRIPTS_DIR}/${script}" ]; then
            cp "${SCRIPTS_DIR}/${script}" "${WORKSPACE}/${script}"
            log "  복사: ${SCRIPTS_DIR}/${script} → ${WORKSPACE}/${script}"
        else
            log "[ERROR] ${script} 를 찾을 수 없습니다. (${WORKSPACE}/ 또는 ${SCRIPTS_DIR}/)"
            write_done "script_missing" "${script}"
            stop_pod "script_missing"
            exit 1
        fi
    fi
done

# CAI 파일은 선택적 (없으면 SFT 전용으로 진행)
if [ ! -f "${DATA_DIR}/cai_pairs.filtered.jsonl" ]; then
    log "[WARN] cai_pairs.filtered.jsonl 없음 — SFT 전용으로 진행"
    # train_sft.py 는 CAI 파일 없으면 SFT만 사용 (graceful)
else
    # train_sft.py 가 cai_pairs.jsonl 을 찾으므로 symlink 생성
    ln -sf "${DATA_DIR}/cai_pairs.filtered.jsonl" "${DATA_DIR}/cai_pairs.jsonl" 2>/dev/null || true
fi

# ── STEP 2: CPT 학습 ─────────────────────────────────────────────────────────
log "[2/5] CPT 학습 시작 (예상 ~18시간)..."
notify "dalbitalba CPT 학습 시작 — pod ${RUNPOD_POD_ID:-unknown}"

CPT_START=$(date +%s)
python "${WORKSPACE}/train_cpt.py" >> "${LOG_FILE}" 2>&1
CPT_EXIT=$?
CPT_END=$(date +%s)
CPT_ELAPSED=$(( (CPT_END - CPT_START) / 60 ))

if [ ${CPT_EXIT} -ne 0 ]; then
    log "[ERROR] CPT 학습 실패 (exit ${CPT_EXIT}, ${CPT_ELAPSED}분 경과)"
    write_done "cpt_failed" "exit_code=${CPT_EXIT} elapsed_min=${CPT_ELAPSED}"
    notify "dalbitalba CPT 실패 — exit ${CPT_EXIT}"
    stop_pod "cpt_failed"
    exit 1
fi

# CPT 최종 loss 추출 (train_cpt.log 에서 마지막 loss 행)
CPT_LOSS=$(grep -E "loss" "${WORKSPACE}/train_cpt.log" 2>/dev/null | tail -1 || echo "loss=N/A")
log "[2/5] CPT 완료: ${CPT_ELAPSED}분 소요 | ${CPT_LOSS}"
notify "dalbitalba CPT 완료 (${CPT_ELAPSED}min) — SFT 시작"

# ── STEP 3: SFT 학습 ─────────────────────────────────────────────────────────
log "[3/5] SFT 학습 시작 (예상 ~5시간)..."

SFT_START=$(date +%s)
python "${WORKSPACE}/train_sft.py" >> "${LOG_FILE}" 2>&1
SFT_EXIT=$?
SFT_END=$(date +%s)
SFT_ELAPSED=$(( (SFT_END - SFT_START) / 60 ))

SFT_LOSS=$(grep -E "loss" "${WORKSPACE}/train_sft.log" 2>/dev/null | tail -1 || echo "loss=N/A")

if [ ${SFT_EXIT} -ne 0 ]; then
    log "[WARN] SFT 학습 실패 (exit ${SFT_EXIT}, ${SFT_ELAPSED}분 경과). CPT 결과만 업로드 진행."
    SFT_STATUS="sft_failed"
    notify "dalbitalba SFT 실패 — CPT adapter 만 업로드"
else
    log "[3/5] SFT 완료: ${SFT_ELAPSED}분 소요 | ${SFT_LOSS}"
    SFT_STATUS="ok"
    notify "dalbitalba SFT 완료 (${SFT_ELAPSED}min) — HF 업로드 시작"
fi

# ── STEP 4: HuggingFace Hub 업로드 ──────────────────────────────────────────
log "[4/5] HuggingFace Hub 업로드..."

TIMESTAMP=$(date -u '+%Y%m%d-%H%M')
HF_REPO="${HF_USERNAME}/dalbitalba-solar-cpt-sft-${TIMESTAMP}"
export HF_REPO

# 업로드 Python 스크립트 (인라인)
# HF_TOKEN 은 환경변수로 전달 — 로그에 직접 출력하지 않음
HF_UPLOAD_SCRIPT=$(cat << 'PYEOF'
import os
import sys
import time
from huggingface_hub import HfApi, create_repo

api = HfApi()
token = os.environ["HF_TOKEN"]
repo_id = os.environ["HF_REPO"]
cpt_out = os.environ.get("CPT_OUT", "/workspace/out/cpt-lora")
sft_out = os.environ.get("SFT_OUT", "/workspace/out/sft-lora")
sft_status = os.environ.get("SFT_STATUS", "ok")

print(f"[hf_upload] repo: {repo_id}")

# repo 생성 (private)
try:
    create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
    print(f"[hf_upload] repo 생성/확인 완료")
except Exception as e:
    print(f"[hf_upload] repo 생성 오류: {e}")
    sys.exit(1)

def upload_folder_with_retry(local_path, path_in_repo, retries=3):
    for attempt in range(1, retries + 1):
        try:
            api.upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                token=token,
                repo_type="model",
            )
            print(f"[hf_upload] 업로드 완료: {local_path} → {path_in_repo}")
            return True
        except Exception as e:
            print(f"[hf_upload] attempt {attempt}/{retries} 실패: {e}")
            if attempt < retries:
                time.sleep(30 * attempt)
    return False

# CPT adapter 업로드
if os.path.isdir(cpt_out):
    ok = upload_folder_with_retry(cpt_out, "cpt-lora")
    if not ok:
        print("[hf_upload] CPT 업로드 최종 실패")
        sys.exit(2)
else:
    print(f"[hf_upload] CPT 디렉토리 없음: {cpt_out}")
    sys.exit(2)

# SFT adapter 업로드 (sft 성공 시만)
if sft_status == "ok" and os.path.isdir(sft_out):
    ok = upload_folder_with_retry(sft_out, "sft-lora")
    if not ok:
        print("[hf_upload] SFT 업로드 최종 실패 (CPT 는 업로드됨)")
        sys.exit(3)
else:
    print(f"[hf_upload] SFT skip (status={sft_status})")

print(f"[hf_upload] 전체 완료: https://huggingface.co/{repo_id}")
PYEOF
)

HF_UPLOAD_EXIT=0
CPT_OUT="${CPT_OUT}" SFT_OUT="${SFT_OUT}" SFT_STATUS="${SFT_STATUS}" \
    python - << EOF >> "${LOG_FILE}" 2>&1
${HF_UPLOAD_SCRIPT}
EOF
HF_UPLOAD_EXIT=$?

# 업로드 결과 처리
case ${HF_UPLOAD_EXIT} in
    0)
        log "[4/5] HF 업로드 완료: https://huggingface.co/${HF_REPO}"
        notify "dalbitalba HF 업로드 완료 — https://huggingface.co/${HF_REPO}"
        HF_STATUS="uploaded"
        ;;
    2)
        log "[ERROR] HF CPT 업로드 실패 (3회 시도). adapter 는 volume 에 보존됨."
        notify "dalbitalba HF 업로드 실패 — volume 보존"
        HF_STATUS="upload_failed"
        ;;
    3)
        log "[WARN] HF SFT 업로드 실패. CPT 는 업로드됨."
        HF_STATUS="sft_upload_failed"
        ;;
    *)
        log "[ERROR] HF 업로드 알 수 없는 오류 (exit ${HF_UPLOAD_EXIT})"
        HF_STATUS="upload_error"
        ;;
esac

# ── STEP 5: DONE.txt 기록 ────────────────────────────────────────────────────
log "[5/5] 완료 기록..."

CHAIN_END=$(date +%s)
CHAIN_TOTAL=$(( (CHAIN_END - CPT_START) / 60 ))

{
    echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "total_elapsed_min=${CHAIN_TOTAL}"
    echo "cpt_elapsed_min=${CPT_ELAPSED}"
    echo "cpt_loss=${CPT_LOSS}"
    echo "sft_status=${SFT_STATUS}"
    echo "sft_elapsed_min=${SFT_ELAPSED}"
    echo "sft_loss=${SFT_LOSS}"
    echo "hf_status=${HF_STATUS}"
    echo "hf_repo=${HF_REPO}"
    echo "cpt_out=${CPT_OUT}"
    echo "sft_out=${SFT_OUT}"
    echo "pod_id=${RUNPOD_POD_ID:-unknown}"
} > "${DONE_FILE}"
log "[DONE] ${DONE_FILE}"
cat "${DONE_FILE}" | tee -a "${LOG_FILE}"

# ── 최종 종료 ─────────────────────────────────────────────────────────────────
FINAL_STATUS="done_ok"
if [ "${SFT_STATUS}" = "sft_failed" ]; then
    FINAL_STATUS="done_cpt_only"
fi
if [ "${HF_STATUS}" = "upload_failed" ]; then
    FINAL_STATUS="done_no_upload"
fi

log "=========================================="
log "chain_train.sh 완료: ${FINAL_STATUS} (총 ${CHAIN_TOTAL}분)"
log "=========================================="

notify "dalbitalba chain 완료: ${FINAL_STATUS} (${CHAIN_TOTAL}min) | HF: ${HF_REPO}"
persist_run_artifacts "${FINAL_STATUS}"

stop_pod "${FINAL_STATUS}"
