#!/usr/bin/env bash
# chain_train.sh ??dalbitalba 臾댁씤 CPT?뭆FT?믪뾽濡쒕뱶?믪쥌猷??뚯씠?꾨씪??#
# ?ㅽ뻾 ?꾩튂: RunPod pod ?대? (nohup ?먮뒗 startup script 濡??ㅽ뻾)
# 濡쒓렇: /workspace/logs/chain.log (tee 濡?stdout ?숈떆 異쒕젰)
#
# ?꾩닔 ?섍꼍蹂??(pod env ??二쇱엯):
#   HF_TOKEN       ??HuggingFace write token
#   HF_USERNAME    ??HuggingFace username (?? unoa-labs)
#   RUNPOD_POD_ID  ??RunPod pod ID (RunPod ?먮룞 二쇱엯)
#   NTFY_TOPIC     ??(?좏깮) ntfy.sh topic, 鍮꾩썙?먮㈃ ?뚮┝ skip
#   GITHUB_TOKEN   ??(?좏깮) train repo branch push token
#   GITHUB_REPO    ??(?좏깮) train repo name, 湲곕낯媛?unoa-eng/dalbitalba-train-data
#
# ?μ븷 ?꾨왂: graceful degradation
#   - CPT ?ㅽ뙣 ??DONE.txt ??cpt_failed 湲곕줉, exit 1
#   - SFT ?ㅽ뙣 ??CPT 寃곌낵留?HF ?낅줈?? DONE.txt ??sft_failed 湲곕줉
#   - HF ?낅줈???ㅽ뙣 ??3???ъ떆?? 洹몃옒???ㅽ뙣?섎㈃ volume 蹂댁〈 ??醫낅즺

set -uo pipefail
# set -e ?쒓굅: 媛쒕퀎 ?④퀎蹂꾨줈 ?ㅻ쪟 泥섎━?섏뿬 graceful degradation 援ы쁽

# ?? ?붾젆?좊━ 諛?濡쒓렇 ?ㅼ젙 ?????????????????????????????????????????????????????
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
TRAIN_CPT_JSONL="${INPUT_JSONL:-${DATA_DIR}/cpt_corpus.jsonl}"
TRAIN_SFT_JSONL="${SFT_JSONL:-${DATA_DIR}/sft_pairs_v2.jsonl}"
TRAIN_CAI_JSONL="${CAI_JSONL:-${DATA_DIR}/cai_pairs.jsonl}"
TRAIN_CPT_EPOCHS="${CPT_NUM_EPOCHS:-1}"
TRAIN_SFT_EPOCHS="${SFT_NUM_EPOCHS:-3}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_DIR}"

# ?? 濡쒓렇 ?⑥닔 (stdout + ?뚯씪 ?숈떆 湲곕줉, HF_TOKEN 留덉뒪?? ?????????????????????
log() {
    local msg="$1"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "${ts} ${msg}" | tee -a "${LOG_FILE}"
}

# HF_TOKEN ??濡쒓렇???몄텧?섏? ?딅룄濡?env 異쒕젰 ?듭젣
log "=========================================="
log "dalbitalba chain_train.sh ?쒖옉"
log "  POD_ID: ${RUNPOD_POD_ID:-unknown}"
log "  HF_USERNAME: ${HF_USERNAME:-unset}"
log "  NTFY_TOPIC: ${NTFY_TOPIC:-(?놁쓬, ?뚮┝ skip)}"
log "  CPT_DATA: ${TRAIN_CPT_JSONL}"
log "  SFT_DATA: ${TRAIN_SFT_JSONL}"
log "  CPT_EPOCHS: ${TRAIN_CPT_EPOCHS}"
log "  SFT_EPOCHS: ${TRAIN_SFT_EPOCHS}"
log "=========================================="

# ?? ntfy ?뚮┝ ?⑥닔 ????????????????????????????????????????????????????????????
notify() {
    local msg="$1"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -s -X POST \
            -H "Content-Type: text/plain; charset=utf-8" \
            --data-binary "${msg}" \
            "https://ntfy.sh/${NTFY_TOPIC}" \
            >> "${LOG_FILE}" 2>&1 || true
        log "[ntfy] ?뚮┝ ?꾩넚: ${msg}"
    fi
}

# ?? 醫낅즺 泥섎━: pod 以묒? (蹂쇰ⅷ 蹂댁〈) ?????????????????????????????????????????
stop_pod() {
    local reason="$1"
    log "[醫낅즺] ?댁쑀: ${reason}"
    notify "dalbitalba chain ${reason} ??pod 以묒?"
    if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        log "[醫낅즺] runpodctl stop pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}" >> "${LOG_FILE}" 2>&1 || true
    else
        log "[醫낅즺] runpodctl ?놁쓬 ?먮뒗 RUNPOD_POD_ID 誘몄꽕?????섎룞 醫낅즺 ?꾩슂"
    fi
}

# ?? DONE.txt 湲곕줉 ?⑥닔 ???????????????????????????????????????????????????????
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
    log "[DONE] ${DONE_FILE} 湲곕줉 ?꾨즺: status=${status}"
}

persist_run_artifacts() {
    local final_status="$1"
    local stamp
    local branch
    local run_dir
    local repo_url
    local source_ref
    local latest_tmp

    if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_REPO:-}" ]; then
        log "[runs] GITHUB_TOKEN/GITHUB_REPO 誘몄꽕?????먭꺽 push skip"
        return 0
    fi
    if [ ! -d "${REPO_CLONE_DIR}/.git" ]; then
        log "[runs] repo clone 誘몄〈?????먭꺽 push skip"
        return 0
    fi

    stamp="$(date -u '+%Y%m%d-%H%M%S')"
    branch="train-run-${stamp}"
    run_dir="${REPO_CLONE_DIR}/runs/${branch}"
    latest_tmp="$(mktemp)"
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

    source_ref="$(
        cd "${REPO_CLONE_DIR}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || true
    )"

    mkdir -p "${REPO_CLONE_DIR}/runs"
    cat > "${latest_tmp}" <<EOF
{
  "branch": "${branch}",
  "status": "${final_status}",
  "hf_repo": "${HF_REPO:-}",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF
    cp "${latest_tmp}" "${REPO_CLONE_DIR}/runs/latest-train.json"

    repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
    (
        cd "${REPO_CLONE_DIR}"
        git checkout -b "${branch}" >/dev/null 2>&1
        git add "runs/${branch}" "runs/latest-train.json"
        git commit -m "train: ${branch}" >/dev/null 2>&1 || exit 0
        git push "${repo_url}" "${branch}" >/dev/null 2>&1
    )

    if [ $? -eq 0 ]; then
        log "[runs] ?먭꺽 branch push ?꾨즺: ${branch}"
        notify "dalbitalba train artifact push ?꾨즺: ${branch}"
        echo "${branch}" > "${OUT_DIR}/RUN_BRANCH.txt"
        if [ -n "${source_ref}" ] && [ "${source_ref}" != "HEAD" ]; then
            if (
                cd "${REPO_CLONE_DIR}"
                git checkout "${source_ref}" >/dev/null 2>&1
                mkdir -p runs
                cp "${latest_tmp}" "runs/latest-train.json"
                git add "runs/latest-train.json"
                git commit -m "train: update latest pointer" >/dev/null 2>&1 || exit 0
                git push "${repo_url}" "${source_ref}" >/dev/null 2>&1
            ); then
                log "[runs] latest pointer updated on ${source_ref}"
            else
                log "[runs] latest pointer update failed on ${source_ref}"
            fi
        else
            log "[runs] source ref unavailable; latest pointer push skipped"
        fi
        rm -f "${latest_tmp}"
    else
        log "[runs] ?먭꺽 branch push ?ㅽ뙣: ${branch}"
    fi
}

# ?? STEP 0: ?꾩닔 ?섍꼍蹂???뺤씤 ???????????????????????????????????????????????
log "[0/5] ?섍꼍 蹂???뺤씤..."
if [ -z "${HF_TOKEN:-}" ]; then
    log "[ERROR] HF_TOKEN ???ㅼ젙?섏? ?딆븯?듬땲?? HF ?낅줈??遺덇?."
    write_done "env_error" "HF_TOKEN missing"
    stop_pod "env_error"
    exit 1
fi
if [ -z "${HF_USERNAME:-}" ]; then
    log "[ERROR] HF_USERNAME ???ㅼ젙?섏? ?딆븯?듬땲??"
    write_done "env_error" "HF_USERNAME missing"
    stop_pod "env_error"
    exit 1
fi

# ?곗씠???뚯씪 議댁옱 ?뺤씤
for f in "${TRAIN_CPT_JSONL}" "${TRAIN_SFT_JSONL}"; do
    if [ ! -f "${f}" ]; then
        log "[ERROR] ?꾩닔 ?곗씠???뚯씪 ?놁쓬: ${f}"
        write_done "data_missing" "${f}"
        stop_pod "data_missing"
        exit 1
    fi
done
log "[0/5] ?섍꼍 ?뺤씤 ?꾨즺. ?곗씠???뚯씪 ?뺤씤 ?꾨즺."

# ?? STEP 1: ?⑦궎吏 ?ㅼ튂 ??????????????????????????????????????????????????????
log "[1/5] ?꾩닔 ?⑦궎吏 ?ㅼ튂..."
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
    log "[ERROR] ?⑦궎吏 ?ㅼ튂 ?ㅽ뙣"
    write_done "install_failed"
    stop_pod "install_failed"
    exit 1
fi
log "[1/5] ?⑦궎吏 ?ㅼ튂 ?꾨즺."

# ?? train scripts ?뺤씤 諛?蹂듭궗 ???????????????????????????????????????????????
# launch_chain.py 媛 scripts/ 瑜??④퍡 ?낅줈?쒗뻽?ㅺ퀬 媛??# ?놁쑝硫?SCRIPTS_DIR ?먯꽌 李얠븘 WORKSPACE 猷⑦듃濡?蹂듭궗
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
            log "  蹂듭궗: ${SCRIPTS_DIR}/${script} ??${WORKSPACE}/${script}"
        else
            log "[ERROR] ${script} 瑜?李얠쓣 ???놁뒿?덈떎. (${WORKSPACE}/ ?먮뒗 ${SCRIPTS_DIR}/)"
            write_done "script_missing" "${script}"
            stop_pod "script_missing"
            exit 1
        fi
    fi
done

# CAI ?뚯씪? ?좏깮??(?놁쑝硫?SFT ?꾩슜?쇰줈 吏꾪뻾)
if [ ! -f "${DATA_DIR}/cai_pairs.filtered.jsonl" ]; then
    log "[WARN] cai_pairs.filtered.jsonl ?놁쓬 ??SFT ?꾩슜?쇰줈 吏꾪뻾"
    # train_sft.py ??CAI ?뚯씪 ?놁쑝硫?SFT留??ъ슜 (graceful)
else
    # train_sft.py 媛 cai_pairs.jsonl ??李얠쑝誘濡?symlink ?앹꽦
    ln -sf "${DATA_DIR}/cai_pairs.filtered.jsonl" "${TRAIN_CAI_JSONL}" 2>/dev/null || true
fi

# ?? STEP 2: CPT ?숈뒿 ?????????????????????????????????????????????????????????
log "[2/5] CPT ?숈뒿 ?쒖옉 (?덉긽 ~18?쒓컙)..."
notify "dalbitalba CPT ?숈뒿 ?쒖옉 ??pod ${RUNPOD_POD_ID:-unknown}"

CPT_START=$(date +%s)
INPUT_JSONL="${TRAIN_CPT_JSONL}" \
CPT_NUM_EPOCHS="${TRAIN_CPT_EPOCHS}" \
CPT_OUTPUT_DIR="${CPT_OUT}" \
CPT_CKPT_DIR="${OUT_DIR}/cpt-ckpt" \
CPT_LOG_FILE="${WORKSPACE}/train_cpt.log" \
python "${WORKSPACE}/train_cpt.py" >> "${LOG_FILE}" 2>&1
CPT_EXIT=$?
CPT_END=$(date +%s)
CPT_ELAPSED=$(( (CPT_END - CPT_START) / 60 ))

if [ ${CPT_EXIT} -ne 0 ]; then
    log "[ERROR] CPT ?숈뒿 ?ㅽ뙣 (exit ${CPT_EXIT}, ${CPT_ELAPSED}遺?寃쎄낵)"
    write_done "cpt_failed" "exit_code=${CPT_EXIT} elapsed_min=${CPT_ELAPSED}"
    notify "dalbitalba CPT ?ㅽ뙣 ??exit ${CPT_EXIT}"
    stop_pod "cpt_failed"
    exit 1
fi

# CPT 理쒖쥌 loss 異붿텧 (train_cpt.log ?먯꽌 留덉?留?loss ??
CPT_LOSS=$(grep -E "loss" "${WORKSPACE}/train_cpt.log" 2>/dev/null | tail -1 || echo "loss=N/A")
log "[2/5] CPT ?꾨즺: ${CPT_ELAPSED}遺??뚯슂 | ${CPT_LOSS}"
notify "dalbitalba CPT ?꾨즺 (${CPT_ELAPSED}min) ??SFT ?쒖옉"

# ?? STEP 3: SFT ?숈뒿 ?????????????????????????????????????????????????????????
log "[3/5] SFT ?숈뒿 ?쒖옉 (?덉긽 ~5?쒓컙)..."

SFT_START=$(date +%s)
SFT_JSONL="${TRAIN_SFT_JSONL}" \
CAI_JSONL="${TRAIN_CAI_JSONL}" \
CPT_LORA_DIR="${CPT_OUT}" \
SFT_NUM_EPOCHS="${TRAIN_SFT_EPOCHS}" \
SFT_OUTPUT_DIR="${SFT_OUT}" \
SFT_CKPT_DIR="${OUT_DIR}/sft-ckpt" \
SFT_LOG_FILE="${WORKSPACE}/train_sft.log" \
python "${WORKSPACE}/train_sft.py" >> "${LOG_FILE}" 2>&1
SFT_EXIT=$?
SFT_END=$(date +%s)
SFT_ELAPSED=$(( (SFT_END - SFT_START) / 60 ))

SFT_LOSS=$(grep -E "loss" "${WORKSPACE}/train_sft.log" 2>/dev/null | tail -1 || echo "loss=N/A")

if [ ${SFT_EXIT} -ne 0 ]; then
    log "[WARN] SFT ?숈뒿 ?ㅽ뙣 (exit ${SFT_EXIT}, ${SFT_ELAPSED}遺?寃쎄낵). CPT 寃곌낵留??낅줈??吏꾪뻾."
    SFT_STATUS="sft_failed"
    notify "dalbitalba SFT failed, upload CPT adapter only"
else
    log "[3/5] SFT ?꾨즺: ${SFT_ELAPSED}遺??뚯슂 | ${SFT_LOSS}"
    SFT_STATUS="ok"
    notify "dalbitalba SFT ?꾨즺 (${SFT_ELAPSED}min) ??HF ?낅줈???쒖옉"
fi

# ?? STEP 4: HuggingFace Hub ?낅줈????????????????????????????????????????????
log "[4/5] HuggingFace Hub ?낅줈??.."

TIMESTAMP=$(date -u '+%Y%m%d-%H%M')
HF_REPO="${HF_USERNAME}/dalbitalba-solar-cpt-sft-${TIMESTAMP}"
export HF_REPO

# ?낅줈??Python ?ㅽ겕由쏀듃 (?몃씪??
# HF_TOKEN ? ?섍꼍蹂?섎줈 ?꾨떖 ??濡쒓렇??吏곸젒 異쒕젰?섏? ?딆쓬
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

# repo ?앹꽦 (private)
try:
    create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
    print(f"[hf_upload] repo ?앹꽦/?뺤씤 ?꾨즺")
except Exception as e:
    print(f"[hf_upload] repo ?앹꽦 ?ㅻ쪟: {e}")
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
            print(f"[hf_upload] ?낅줈???꾨즺: {local_path} ??{path_in_repo}")
            return True
        except Exception as e:
            print(f"[hf_upload] attempt {attempt}/{retries} ?ㅽ뙣: {e}")
            if attempt < retries:
                time.sleep(30 * attempt)
    return False

# CPT adapter ?낅줈??if os.path.isdir(cpt_out):
    ok = upload_folder_with_retry(cpt_out, "cpt-lora")
    if not ok:
        print("[hf_upload] CPT ?낅줈??理쒖쥌 ?ㅽ뙣")
        sys.exit(2)
else:
    print(f"[hf_upload] CPT ?붾젆?좊━ ?놁쓬: {cpt_out}")
    sys.exit(2)

# SFT adapter ?낅줈??(sft ?깃났 ?쒕쭔)
if sft_status == "ok" and os.path.isdir(sft_out):
    ok = upload_folder_with_retry(sft_out, "sft-lora")
    if not ok:
        print("[hf_upload] SFT ?낅줈??理쒖쥌 ?ㅽ뙣 (CPT ???낅줈?쒕맖)")
        sys.exit(3)
else:
    print(f"[hf_upload] SFT skip (status={sft_status})")

print(f"[hf_upload] ?꾩껜 ?꾨즺: https://huggingface.co/{repo_id}")
PYEOF
)

HF_UPLOAD_EXIT=0
CPT_OUT="${CPT_OUT}" SFT_OUT="${SFT_OUT}" SFT_STATUS="${SFT_STATUS}" \
    python - << EOF >> "${LOG_FILE}" 2>&1
${HF_UPLOAD_SCRIPT}
EOF
HF_UPLOAD_EXIT=$?

# ?낅줈??寃곌낵 泥섎━
case ${HF_UPLOAD_EXIT} in
    0)
        log "[4/5] HF ?낅줈???꾨즺: https://huggingface.co/${HF_REPO}"
        notify "dalbitalba HF ?낅줈???꾨즺 ??https://huggingface.co/${HF_REPO}"
        HF_STATUS="uploaded"
        ;;
    2)
        log "[ERROR] HF CPT ?낅줈???ㅽ뙣 (3???쒕룄). adapter ??volume ??蹂댁〈??"
        notify "dalbitalba HF ?낅줈???ㅽ뙣 ??volume 蹂댁〈"
        HF_STATUS="upload_failed"
        ;;
    3)
        log "[WARN] HF SFT ?낅줈???ㅽ뙣. CPT ???낅줈?쒕맖."
        HF_STATUS="sft_upload_failed"
        ;;
    *)
        log "[ERROR] HF ?낅줈???????녿뒗 ?ㅻ쪟 (exit ${HF_UPLOAD_EXIT})"
        HF_STATUS="upload_error"
        ;;
esac

# ?? STEP 5: DONE.txt 湲곕줉 ????????????????????????????????????????????????????
log "[5/5] ?꾨즺 湲곕줉..."

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

# ?? 理쒖쥌 醫낅즺 ?????????????????????????????????????????????????????????????????
FINAL_STATUS="done_ok"
if [ "${SFT_STATUS}" = "sft_failed" ]; then
    FINAL_STATUS="done_cpt_only"
fi
if [ "${HF_STATUS}" = "upload_failed" ]; then
    FINAL_STATUS="done_no_upload"
fi

log "=========================================="
log "chain_train.sh ?꾨즺: ${FINAL_STATUS} (珥?${CHAIN_TOTAL}遺?"
log "=========================================="

notify "dalbitalba chain ?꾨즺: ${FINAL_STATUS} (${CHAIN_TOTAL}min) | HF: ${HF_REPO}"
persist_run_artifacts "${FINAL_STATUS}"

stop_pod "${FINAL_STATUS}"

