#!/usr/bin/env bash
# chain_train.sh v2 — CPT -> merge -> SFT -> HF upload -> stop_pod
# Runs inside a RunPod pod. Host orchestration is launch_train_pod.py.
#
# Required env (injected by launch_train_pod.py):
#   HF_TOKEN          HuggingFace write token
#   HF_USERNAME       HuggingFace user/org
#   RUNPOD_POD_ID     auto-injected by RunPod
#   GITHUB_TOKEN      (optional) to push run manifests back to repo
#   GITHUB_REPO       (optional)
#   NTFY_TOPIC        (optional) ntfy.sh push channel
#   SENTRY_DSN        (optional)
#   WANDB_API_KEY     (optional)
#   WANDB_PROJECT     (optional)
#
# Graceful degradation: every failure writes DONE.txt with the stage, then
# stop_pod. All stdout/stderr tee'd to /workspace/logs/chain.log which is
# preserved on the network volume after pod EXIT.

set -uo pipefail

WORKSPACE="/workspace"
LOG_DIR="${WORKSPACE}/logs"
LOG_FILE="${LOG_DIR}/chain.log"
OUT_DIR="${WORKSPACE}/out"
DATA_DIR="${WORKSPACE}/data"
HF_CACHE_DIR="${WORKSPACE}/hf_cache"
SCRIPTS_DIR="${WORKSPACE}/scripts"
REPO_CLONE_DIR="${WORKSPACE}/repo"

CPT_OUT="${OUT_DIR}/cpt-lora"
CPT_MERGED="${OUT_DIR}/cpt-merged-fp16"
SFT_OUT="${OUT_DIR}/sft-lora"
DONE_FILE="${OUT_DIR}/DONE.txt"

TRAIN_CPT_JSONL="${INPUT_JSONL:-${DATA_DIR}/cpt_corpus.v2.jsonl}"
TRAIN_SFT_PAIR_JSONL="${SFT_PAIR_JSONL:-${DATA_DIR}/sft_pairs.v2.jsonl}"
TRAIN_VAL_JSONL="${CPT_VAL_JSONL:-${DATA_DIR}/val_set.v2.jsonl}"
CPT_EPOCHS="${CPT_NUM_EPOCHS:-1}"
SFT_EPOCHS="${SFT_NUM_EPOCHS:-2}"
BASE_MODEL_CPT="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
TIMESTAMP="$(date -u '+%Y%m%d-%H%M')"
HF_REPO_SFT="${HF_USERNAME:-unoa}/dalbitalba-qwen3-sft-${TIMESTAMP}"
HF_REPO_CPT="${HF_USERNAME:-unoa}/dalbitalba-qwen3-cpt-${TIMESTAMP}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_REPO_SFT HF_REPO_CPT

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_DIR}" "${HF_CACHE_DIR}"

log() {
    local msg="$1"
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "${ts} ${msg}" | tee -a "${LOG_FILE}"
}

notify() {
    local msg="$1"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -m 10 \
            -H "Content-Type: text/plain; charset=utf-8" \
            --data-binary "${msg}" \
            "https://ntfy.sh/${NTFY_TOPIC}" \
            >> "${LOG_FILE}" 2>&1 || true
    fi
}

stop_pod() {
    local reason="$1"
    log "[stop] reason=${reason}"
    notify "dalbitalba chain ${reason}"
    if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl stop pod "${RUNPOD_POD_ID}" >> "${LOG_FILE}" 2>&1 || true
    fi
}

write_done() {
    local status="$1"
    local extra="${2:-}"
    {
        echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "status=${status}"
        [ -n "${extra}" ] && echo "detail=${extra}"
        echo "base_model=${BASE_MODEL_CPT}"
        echo "cpt_out=${CPT_OUT}"
        echo "cpt_merged=${CPT_MERGED}"
        echo "sft_out=${SFT_OUT}"
        echo "hf_repo_cpt=${HF_REPO_CPT:-}"
        echo "hf_repo_sft=${HF_REPO_SFT:-}"
        echo "pod_id=${RUNPOD_POD_ID:-unknown}"
    } > "${DONE_FILE}"
    log "[DONE] ${DONE_FILE}"
}

persist_run_artifacts() {
    local final_status="$1"
    if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_REPO:-}" ]; then
        log "[runs] GITHUB_TOKEN/GITHUB_REPO 미설정 → push skip"
        return 0
    fi
    if [ ! -d "${REPO_CLONE_DIR}/.git" ]; then
        log "[runs] repo clone 없음 → push skip"
        return 0
    fi
    local stamp branch run_dir latest_tmp source_ref repo_url
    stamp="$(date -u '+%Y%m%d-%H%M%S')"
    branch="train-run-${stamp}"
    run_dir="${REPO_CLONE_DIR}/runs/${branch}"
    latest_tmp="$(mktemp)"
    mkdir -p "${run_dir}"

    cp "${DONE_FILE}" "${run_dir}/DONE.txt" 2>/dev/null || true
    [ -f "${LOG_FILE}" ] && cp "${LOG_FILE}" "${run_dir}/chain.log"
    [ -f "${WORKSPACE}/train_cpt.log" ] && cp "${WORKSPACE}/train_cpt.log" "${run_dir}/train_cpt.log"
    [ -f "${WORKSPACE}/train_sft.log" ] && cp "${WORKSPACE}/train_sft.log" "${run_dir}/train_sft.log"

    cat > "${run_dir}/manifest.json" <<EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "status": "${final_status}",
  "base_model": "${BASE_MODEL_CPT}",
  "pod_id": "${RUNPOD_POD_ID:-unknown}",
  "hf_repo_cpt": "${HF_REPO_CPT:-}",
  "hf_repo_sft": "${HF_REPO_SFT:-}",
  "cpt_epochs": ${CPT_EPOCHS},
  "sft_epochs": ${SFT_EPOCHS},
  "source_repo": "${GITHUB_REPO}"
}
EOF

    source_ref="$(cd "${REPO_CLONE_DIR}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    mkdir -p "${REPO_CLONE_DIR}/runs"
    cat > "${latest_tmp}" <<EOF
{
  "branch": "${branch}",
  "status": "${final_status}",
  "hf_repo_cpt": "${HF_REPO_CPT:-}",
  "hf_repo_sft": "${HF_REPO_SFT:-}",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF
    cp "${latest_tmp}" "${REPO_CLONE_DIR}/runs/latest-train.json"

    repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
    (
        cd "${REPO_CLONE_DIR}"
        git checkout -b "${branch}" >/dev/null 2>&1 || true
        git add "runs/${branch}" "runs/latest-train.json" >/dev/null 2>&1 || true
        git commit -m "train: ${branch}" >/dev/null 2>&1 || exit 0
        git push "${repo_url}" "${branch}" >/dev/null 2>&1
    ) && log "[runs] push branch ${branch}" || log "[runs] push 실패 ${branch}"

    if [ -n "${source_ref}" ] && [ "${source_ref}" != "HEAD" ]; then
        (
            cd "${REPO_CLONE_DIR}"
            git checkout "${source_ref}" >/dev/null 2>&1
            mkdir -p runs
            cp "${latest_tmp}" "runs/latest-train.json"
            git add "runs/latest-train.json"
            git commit -m "train: update latest pointer" >/dev/null 2>&1 || exit 0
            git push "${repo_url}" "${source_ref}" >/dev/null 2>&1
        ) && log "[runs] latest pointer pushed to ${source_ref}" || true
    fi
    rm -f "${latest_tmp}"
}

# ── Trap: on SIGTERM / SIGINT make sure DONE.txt reflects the state ──
on_exit() {
    local rc=$?
    if [ "${rc}" -ne 0 ] && [ ! -f "${DONE_FILE}" ]; then
        write_done "unexpected_exit" "rc=${rc}"
    fi
}
trap on_exit EXIT

# ── STEP 0 env check ────────────────────────────────────────────────
log "=========================================="
log "dalbitalba chain_train.sh v2 시작"
log "  pod          : ${RUNPOD_POD_ID:-unknown}"
log "  base         : ${BASE_MODEL_CPT}"
log "  hf_repo_cpt  : ${HF_REPO_CPT}"
log "  hf_repo_sft  : ${HF_REPO_SFT}"
log "  cpt_epochs   : ${CPT_EPOCHS}"
log "  sft_epochs   : ${SFT_EPOCHS}"
log "=========================================="

if [ -z "${HF_TOKEN:-}" ]; then
    log "[ERROR] HF_TOKEN 미설정"
    write_done "env_error" "HF_TOKEN"
    stop_pod "env_error"
    exit 1
fi
if [ -z "${HF_USERNAME:-}" ]; then
    log "[ERROR] HF_USERNAME 미설정"
    write_done "env_error" "HF_USERNAME"
    stop_pod "env_error"
    exit 1
fi

for f in "${TRAIN_CPT_JSONL}" "${TRAIN_SFT_PAIR_JSONL}"; do
    if [ ! -f "${f}" ]; then
        log "[ERROR] 데이터 파일 없음: ${f}"
        write_done "data_missing" "${f}"
        stop_pod "data_missing"
        exit 1
    fi
done

# ── STEP 1 pip install ──────────────────────────────────────────────
log "[1/6] pip install (pinned)"
pip install -q --no-cache-dir --upgrade pip >> "${LOG_FILE}" 2>&1
pip install -q --no-cache-dir \
    "transformers==4.51.3" \
    "peft==0.13.2" \
    "bitsandbytes==0.49.2" \
    "trl==0.12.1" \
    "accelerate==0.33.0" \
    "datasets==2.21.0" \
    "huggingface_hub>=0.24.0" \
    "safetensors>=0.4.3" \
    "sentencepiece" \
    "tokenizers>=0.21" \
    "protobuf" \
    "wandb" \
    "sentry-sdk" \
    "numpy<2.0" \
    >> "${LOG_FILE}" 2>&1
PIP_RC=$?

# flash-attn은 빌드 오래 걸려서 별도 + 실패해도 계속 진행 (eager fallback)
pip install -q --no-cache-dir --no-build-isolation \
    "flash-attn==2.6.3" >> "${LOG_FILE}" 2>&1 || \
    log "[WARN] flash-attn 설치 실패 — eager attention fallback"

if [ ${PIP_RC} -ne 0 ]; then
    log "[ERROR] pip 필수 패키지 설치 실패"
    write_done "install_failed"
    stop_pod "install_failed"
    exit 1
fi

python - <<'PY' >> "${LOG_FILE}" 2>&1
import torch, transformers, peft, bitsandbytes, datasets, trl, accelerate
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("trl", trl.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
PY

# ── copy train scripts from repo to workspace if needed ─────────────
for script in train_cpt.py train_sft.py; do
    if [ ! -f "${WORKSPACE}/${script}" ]; then
        if [ -f "${REPO_CLONE_DIR}/${script}" ]; then
            cp "${REPO_CLONE_DIR}/${script}" "${WORKSPACE}/${script}"
        fi
    fi
done
if [ ! -f "${WORKSPACE}/scripts/merge_cpt_to_fp16.py" ]; then
    mkdir -p "${WORKSPACE}/scripts"
    cp "${REPO_CLONE_DIR}/scripts/merge_cpt_to_fp16.py" "${WORKSPACE}/scripts/merge_cpt_to_fp16.py" 2>/dev/null || true
fi

# ── STEP 2 CPT ──────────────────────────────────────────────────────
log "[2/6] CPT 학습 시작 (base=${BASE_MODEL_CPT})"
notify "dalbitalba CPT 시작 — ${RUNPOD_POD_ID:-unknown}"

CPT_START=$(date +%s)
BASE_MODEL="${BASE_MODEL_CPT}" \
INPUT_JSONL="${TRAIN_CPT_JSONL}" \
CPT_VAL_JSONL="${TRAIN_VAL_JSONL}" \
CPT_NUM_EPOCHS="${CPT_EPOCHS}" \
CPT_OUTPUT_DIR="${CPT_OUT}" \
CPT_CKPT_DIR="${OUT_DIR}/cpt-ckpt" \
CPT_LOG_FILE="${WORKSPACE}/train_cpt.log" \
CPT_HUB_MODEL_ID="${HF_REPO_CPT}" \
python "${WORKSPACE}/train_cpt.py" >> "${LOG_FILE}" 2>&1
CPT_EXIT=$?
CPT_END=$(date +%s)
CPT_ELAPSED=$(( (CPT_END - CPT_START) / 60 ))

if [ ${CPT_EXIT} -ne 0 ]; then
    log "[ERROR] CPT 실패 exit=${CPT_EXIT} elapsed=${CPT_ELAPSED}m"
    write_done "cpt_failed" "exit=${CPT_EXIT} min=${CPT_ELAPSED}"
    notify "dalbitalba CPT 실패 — exit ${CPT_EXIT}"
    persist_run_artifacts "cpt_failed"
    stop_pod "cpt_failed"
    exit 1
fi
log "[2/6] CPT 완료 (${CPT_ELAPSED}m)"
notify "dalbitalba CPT 완료 (${CPT_ELAPSED}m)"

# ── STEP 3 merge ────────────────────────────────────────────────────
log "[3/6] CPT adapter → fp16 merge"
BASE_MODEL="${BASE_MODEL_CPT}" \
CPT_LORA_DIR="${CPT_OUT}" \
CPT_MERGED_DIR="${CPT_MERGED}" \
python "${WORKSPACE}/scripts/merge_cpt_to_fp16.py" >> "${LOG_FILE}" 2>&1
MERGE_EXIT=$?
if [ ${MERGE_EXIT} -ne 0 ]; then
    log "[ERROR] merge 실패 exit=${MERGE_EXIT}"
    write_done "merge_failed" "exit=${MERGE_EXIT}"
    notify "dalbitalba merge 실패"
    persist_run_artifacts "merge_failed"
    stop_pod "merge_failed"
    exit 1
fi
log "[3/6] merge 완료"

# ── STEP 4 SFT ──────────────────────────────────────────────────────
log "[4/6] SFT 학습 시작 (base=${CPT_MERGED})"
notify "dalbitalba SFT 시작"
SFT_START=$(date +%s)
BASE_MODEL="${CPT_MERGED}" \
SFT_RAW_JSONL="${TRAIN_CPT_JSONL}" \
SFT_PAIR_JSONL="${TRAIN_SFT_PAIR_JSONL}" \
SFT_VAL_JSONL="${TRAIN_VAL_JSONL}" \
SFT_NUM_EPOCHS="${SFT_EPOCHS}" \
SFT_OUTPUT_DIR="${SFT_OUT}" \
SFT_CKPT_DIR="${OUT_DIR}/sft-ckpt" \
SFT_LOG_FILE="${WORKSPACE}/train_sft.log" \
SFT_HUB_MODEL_ID="${HF_REPO_SFT}" \
python "${WORKSPACE}/train_sft.py" >> "${LOG_FILE}" 2>&1
SFT_EXIT=$?
SFT_END=$(date +%s)
SFT_ELAPSED=$(( (SFT_END - SFT_START) / 60 ))

SFT_STATUS="ok"
if [ ${SFT_EXIT} -ne 0 ]; then
    log "[WARN] SFT 실패 exit=${SFT_EXIT} — CPT merged만 배포"
    SFT_STATUS="sft_failed"
fi

# ── STEP 5 HF upload (both CPT lora and SFT lora already pushed via hub_strategy)
log "[5/6] HF 최종 upload (adapter dirs)"
python - <<PYEOF >> "${LOG_FILE}" 2>&1
import os, sys, time
from huggingface_hub import HfApi, create_repo
api = HfApi()
tok = os.environ["HF_TOKEN"]
cpt_repo = os.environ["HF_REPO_CPT"]
sft_repo = os.environ["HF_REPO_SFT"]
cpt_out = os.environ.get("CPT_OUT", "/workspace/out/cpt-lora")
sft_out = os.environ.get("SFT_OUT", "/workspace/out/sft-lora")
sft_status = os.environ.get("SFT_STATUS", "ok")

def push(repo_id, folder, path_in_repo="adapter"):
    try:
        create_repo(repo_id=repo_id, token=tok, private=True, exist_ok=True)
    except Exception as e:
        print("create_repo err", repo_id, e)
        return False
    for attempt in range(1, 4):
        try:
            api.upload_folder(
                folder_path=folder, repo_id=repo_id,
                path_in_repo=path_in_repo, token=tok, repo_type="model",
            )
            print(f"[push] {folder} -> {repo_id}/{path_in_repo}")
            return True
        except Exception as e:
            print(f"[push] attempt {attempt} fail: {e}")
            time.sleep(20 * attempt)
    return False

ok1 = push(cpt_repo, cpt_out, "cpt-lora") if os.path.isdir(cpt_out) else False
ok2 = push(sft_repo, sft_out, "sft-lora") if (sft_status == "ok" and os.path.isdir(sft_out)) else False
print(f"[hf] cpt_push={ok1} sft_push={ok2}")
sys.exit(0 if (ok1 and (ok2 or sft_status != "ok")) else 3)
PYEOF
HF_UPLOAD_EXIT=$?
case "${HF_UPLOAD_EXIT}" in
    0) log "[5/6] HF upload 완료"; HF_STATUS="uploaded" ;;
    3) log "[WARN] HF upload 일부 실패"; HF_STATUS="upload_partial" ;;
    *) log "[ERROR] HF upload 오류 ${HF_UPLOAD_EXIT}"; HF_STATUS="upload_failed" ;;
esac

# ── STEP 6 wrap up ──────────────────────────────────────────────────
CHAIN_END=$(date +%s)
TOTAL_MIN=$(( (CHAIN_END - CPT_START) / 60 ))

FINAL_STATUS="done_ok"
[ "${SFT_STATUS}" = "sft_failed" ] && FINAL_STATUS="done_cpt_only"
[ "${HF_STATUS}" = "upload_failed" ] && FINAL_STATUS="done_no_upload"

{
    echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "status=${FINAL_STATUS}"
    echo "base_model=${BASE_MODEL_CPT}"
    echo "cpt_elapsed_min=${CPT_ELAPSED}"
    echo "sft_elapsed_min=${SFT_ELAPSED:-0}"
    echo "total_elapsed_min=${TOTAL_MIN}"
    echo "cpt_out=${CPT_OUT}"
    echo "cpt_merged=${CPT_MERGED}"
    echo "sft_out=${SFT_OUT}"
    echo "hf_status=${HF_STATUS}"
    echo "hf_repo_cpt=${HF_REPO_CPT}"
    echo "hf_repo_sft=${HF_REPO_SFT}"
    echo "pod_id=${RUNPOD_POD_ID:-unknown}"
} > "${DONE_FILE}"
log "[DONE] ${FINAL_STATUS} (total ${TOTAL_MIN}m)"
cat "${DONE_FILE}" | tee -a "${LOG_FILE}"

notify "dalbitalba chain ${FINAL_STATUS} (${TOTAL_MIN}m) cpt=${HF_REPO_CPT} sft=${HF_REPO_SFT}"
persist_run_artifacts "${FINAL_STATUS}"
stop_pod "${FINAL_STATUS}"
