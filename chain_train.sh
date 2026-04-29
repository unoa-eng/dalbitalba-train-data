#!/usr/bin/env bash
# chain_train.sh v3 — CPT -> merge -> SFT -> HF upload -> stop_pod
#
# Design philosophy after 3 failed pilot launches (2026-04-24):
#   1. NEVER rely on $PATH resolution of pip/python binaries (WSL and some
#      runpod images differ). Always use `python3 -m pip` and `python3 X`.
#   2. Pre-flight smoke test BEFORE training starts — verify imports, CUDA,
#      bnb runtime, tokenizer/config load — and ntfy full telemetry on fail.
#      A single pod run then reveals all env mismatches in one shot.
#   3. Every failure path attaches head+tail of the relevant component log to
#      ntfy so diagnosis is possible without SSH into ephemeral pods.
#   4. Git identity is configured explicitly so persist_run_artifacts pushes
#      run logs to a `runs/train-run-<stamp>` branch on origin.
#   5. Use hf_transfer + HF_HUB_ENABLE_HF_TRANSFER for faster model download.
#
# Required env (injected by launch_train_pod.py):
#   HF_TOKEN, HF_USERNAME, GITHUB_TOKEN, GITHUB_REPO, RUNPOD_POD_ID
# Optional:
#   NTFY_TOPIC, SENTRY_DSN, WANDB_API_KEY, WANDB_PROJECT, TRAIN_REPORT_TO

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
CPT_PY_LOG="${WORKSPACE}/train_cpt.log"
SFT_PY_LOG="${WORKSPACE}/train_sft.log"
MERGE_PY_LOG="${WORKSPACE}/merge_cpt.log"
PREFLIGHT_LOG="${WORKSPACE}/preflight.log"

DEFAULT_TRAIN_CPT_JSONL="${DATA_DIR}/cpt_corpus.v2.jsonl"
if [ -f "${DATA_DIR}/cpt_corpus.v3.jsonl" ]; then
    DEFAULT_TRAIN_CPT_JSONL="${DATA_DIR}/cpt_corpus.v3.jsonl"
fi
if [ -f "${DATA_DIR}/cpt_context_stream.jsonl" ]; then
    DEFAULT_TRAIN_CPT_JSONL="${DATA_DIR}/cpt_context_stream.jsonl"
fi
DEFAULT_TRAIN_SFT_PAIR_JSONL="${DATA_DIR}/sft_pairs.v2.jsonl"
DEFAULT_TRAIN_SFT_INPUT_JSONL="${DEFAULT_TRAIN_SFT_PAIR_JSONL}"
DEFAULT_TRAIN_VAL_JSONL="${DATA_DIR}/val_set.v3.jsonl"

TRAIN_CPT_JSONL="${TRAIN_CPT_JSONL:-${INPUT_JSONL:-${DEFAULT_TRAIN_CPT_JSONL}}}"
TRAIN_SFT_INPUT_JSONL="${TRAIN_SFT_INPUT_JSONL:-${SFT_INPUT_JSONL:-${TRAIN_SFT_PAIR_JSONL:-${SFT_PAIR_JSONL:-${DEFAULT_TRAIN_SFT_INPUT_JSONL}}}}}"
TRAIN_SFT_PAIR_JSONL="${TRAIN_SFT_PAIR_JSONL:-${TRAIN_SFT_INPUT_JSONL}}"
TRAIN_VAL_JSONL="${TRAIN_VAL_JSONL:-${CPT_VAL_JSONL:-${DEFAULT_TRAIN_VAL_JSONL}}}"

# Keep both the legacy and explicit TRAIN_* aliases in sync so local/manual
# launches and launch_train_pod.py resolve the same dataset paths.
export INPUT_JSONL="${TRAIN_CPT_JSONL}"
export TRAIN_CPT_JSONL
export SFT_INPUT_JSONL="${TRAIN_SFT_INPUT_JSONL}"
export TRAIN_SFT_INPUT_JSONL
export SFT_PAIR_JSONL="${TRAIN_SFT_PAIR_JSONL}"
export TRAIN_SFT_PAIR_JSONL
export CPT_VAL_JSONL="${TRAIN_VAL_JSONL}"
export TRAIN_VAL_JSONL
CPT_LR_VALUE="${CPT_LR:-1e-4}"
CPT_EPOCHS="${CPT_NUM_EPOCHS:-2}"
SFT_EPOCHS="${SFT_NUM_EPOCHS:-2}"
BASE_MODEL_CPT="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
# Hard refusal: only Qwen3 family is supported on this branch.
# 0618 burned $60 on SOLAR-10.7B (Llama-2 BPE, ~48% Korean UNK on extended studies).
# Do not allow that regression. Override the guard with FORCE_BASE_MODEL=1 only
# for explicit experiments — never on the budget30 path.
case "${BASE_MODEL_CPT}" in
    Qwen/Qwen3-8B-Base|Qwen/Qwen3-*-Base)
        ;;
    *)
        if [ "${FORCE_BASE_MODEL:-0}" = "1" ]; then
            log "[WARN] BASE_MODEL_CPT=${BASE_MODEL_CPT} not Qwen3 — FORCE_BASE_MODEL=1 set, proceeding"
        else
            log "[FATAL] BASE_MODEL_CPT=${BASE_MODEL_CPT} is not a Qwen3 base model."
            log "[FATAL] This branch refuses to launch with non-Qwen3 to prevent the 0618 SOLAR regression."
            log "[FATAL] Set FORCE_BASE_MODEL=1 only for explicit experiments outside the budget30 path."
            exit 2
        fi
        ;;
esac
SKIP_SFT="${SKIP_SFT:-0}"
CPT_TIMEOUT_HOURS="${CPT_TIMEOUT_HOURS:-36}"
MERGE_TIMEOUT_HOURS="${MERGE_TIMEOUT_HOURS:-8}"
SFT_TIMEOUT_HOURS="${SFT_TIMEOUT_HOURS:-96}"
HF_UPLOAD_TIMEOUT_HOURS="${HF_UPLOAD_TIMEOUT_HOURS:-4}"
TIMESTAMP="$(date -u '+%Y%m%d-%H%M')"
HF_REPO_SFT="${HF_USERNAME:-unoa}/dalbitalba-qwen3-sft-${TIMESTAMP}"
HF_REPO_CPT="${HF_USERNAME:-unoa}/dalbitalba-qwen3-cpt-${TIMESTAMP}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export HF_REPO_SFT HF_REPO_CPT

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_DIR}" "${HF_CACHE_DIR}"
if [ -d "${REPO_CLONE_DIR}/v3-data" ] && [ ! -d "${DATA_DIR}/v3-data" ]; then
    cp -r "${REPO_CLONE_DIR}/v3-data" "${DATA_DIR}/v3-data" 2>/dev/null || true
fi

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

# Compress a file into a single-line ntfy-friendly blob with head+tail.
blob_head_tail() {
    local file="$1"
    local nhead="${2:-15}"
    local ntail="${3:-50}"
    if [ ! -f "${file}" ]; then
        echo "(no ${file})"
        return
    fi
    # Favor the tail (where the actual error usually lives). Cap at 3500
    # chars total — ntfy.sh supports up to 4KB per message body.
    {
        echo "====HEAD===="
        head -n "${nhead}" "${file}" 2>/dev/null || true
        echo "====TAIL===="
        tail -n "${ntail}" "${file}" 2>/dev/null || true
    } | tr '\n' '|' | cut -c1-3500
}

run_timeout() {
    local hours="$1"
    shift
    if command -v timeout >/dev/null 2>&1; then
        timeout --foreground "${hours}h" "$@"
    else
        log "[WARN] timeout command not found; running without timeout: $*"
        "$@"
    fi
}

resolve_pod_id() {
    if [ -n "${RUNPOD_POD_ID:-}" ] && [ "${RUNPOD_POD_ID}" != "__SELF__" ]; then
        printf '%s' "${RUNPOD_POD_ID}"
        return 0
    fi
    curl -fsS -m 5 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || \
        curl -fsS -m 5 http://metadata.runpod.io/pod-id 2>/dev/null || true
}

stop_pod() {
    local reason="$1"
    local pod_id
    pod_id="$(resolve_pod_id)"
    log "[stop] reason=${reason} pod_id=${pod_id:-unknown}"
    if [ -n "${pod_id}" ]; then
        export RUNPOD_POD_ID="${pod_id}"
    fi
    if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl pod stop "${RUNPOD_POD_ID}" >> "${LOG_FILE}" 2>&1 && return 0
        runpodctl stop pod "${RUNPOD_POD_ID}" >> "${LOG_FILE}" 2>&1 && return 0
    fi
    if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
        curl -fsS -X POST \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
            >> "${LOG_FILE}" 2>&1 || true
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

fail_with_logs() {
    # fail_with_logs <status_tag> <component_log> <exit_code>
    # Sends HEAD+TAIL of component_log to ntfy in a single message, then stops.
    local status="$1"
    local cfile="$2"
    local rc="${3:-1}"
    local blob
    blob="$(blob_head_tail "${cfile}" 25 25)"
    write_done "${status}" "rc=${rc} log=${cfile}"
    notify "dalbit ${status} rc=${rc} ${blob}"
    persist_run_artifacts "${status}"
    stop_pod "${status}"
    exit 1
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
    # Configure git identity (required for commits in containers w/o default)
    git -C "${REPO_CLONE_DIR}" config user.email "runpod-bot@dalbitalba.local" 2>/dev/null
    git -C "${REPO_CLONE_DIR}" config user.name  "dalbitalba-runpod" 2>/dev/null

    local stamp branch run_dir latest_tmp source_ref repo_url
    stamp="$(date -u '+%Y%m%d-%H%M%S')"
    branch="train-run-${stamp}"
    run_dir="${REPO_CLONE_DIR}/runs/${branch}"
    latest_tmp="$(mktemp)"
    mkdir -p "${run_dir}"

    cp "${DONE_FILE}" "${run_dir}/DONE.txt" 2>/dev/null || true
    [ -f "${LOG_FILE}" ]     && cp "${LOG_FILE}" "${run_dir}/chain.log"
    [ -f "${CPT_PY_LOG}" ]   && cp "${CPT_PY_LOG}"   "${run_dir}/train_cpt.log"
    [ -f "${SFT_PY_LOG}" ]   && cp "${SFT_PY_LOG}"   "${run_dir}/train_sft.log"
    [ -f "${MERGE_PY_LOG}" ] && cp "${MERGE_PY_LOG}" "${run_dir}/merge_cpt.log"
    [ -f "${PREFLIGHT_LOG}" ] && cp "${PREFLIGHT_LOG}" "${run_dir}/preflight.log"

    cat > "${run_dir}/manifest.json" <<EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "status": "${final_status}",
  "base_model": "${BASE_MODEL_CPT}",
  "pod_id": "${RUNPOD_POD_ID:-unknown}",
  "hf_repo_cpt": "${HF_REPO_CPT:-}",
  "hf_repo_sft": "${HF_REPO_SFT:-}",
  "cpt_lr": "${CPT_LR_VALUE}",
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
        git commit -m "train: ${branch} (${final_status})" >/dev/null 2>&1 || exit 0
        git push "${repo_url}" "${branch}" >/dev/null 2>&1
    ) && log "[runs] push branch ${branch}" || log "[runs] push 실패 ${branch}"
    rm -f "${latest_tmp}"
}

on_exit() {
    local rc=$?
    if [ "${rc}" -ne 0 ] && [ ! -f "${DONE_FILE}" ]; then
        write_done "unexpected_exit" "rc=${rc}"
    fi
}

# Signal-safe abort: persist whatever we have, then stop the paid pod so the
# wallet does not bleed when the wrapper shell is killed (TERM/INT/HUP/QUIT).
graceful_abort() {
    local sig="$1"
    log "[abort] signal=${sig} — persisting artifacts then stopping pod"
    if [ ! -f "${DONE_FILE}" ]; then
        write_done "aborted_${sig}" "signal=${sig}"
    fi
    persist_run_artifacts "aborted_${sig}" 2>/dev/null || log "[abort] persist failed"
    stop_pod "aborted_${sig}" 2>/dev/null || log "[abort] stop_pod failed"
    exit 130
}
trap 'graceful_abort TERM' TERM
trap 'graceful_abort INT' INT
trap 'graceful_abort HUP' HUP
trap 'graceful_abort QUIT' QUIT
trap on_exit EXIT

# ═══ STEP 0 env + data check ═══════════════════════════════════════════
log "=========================================="
log "dalbitalba chain_train.sh v3 시작"
log "  pod          : ${RUNPOD_POD_ID:-unknown}"
log "  base         : ${BASE_MODEL_CPT}"
log "  hf_repo_cpt  : ${HF_REPO_CPT}"
log "  hf_repo_sft  : ${HF_REPO_SFT}"
log "  cpt_lr       : ${CPT_LR_VALUE}"
log "  cpt_epochs   : ${CPT_EPOCHS}"
log "  sft_epochs   : ${SFT_EPOCHS}"
log "  skip_sft     : ${SKIP_SFT}"
log "  cpt_jsonl    : ${TRAIN_CPT_JSONL}"
log "  sft_input    : ${TRAIN_SFT_INPUT_JSONL}"
log "  val_jsonl    : ${TRAIN_VAL_JSONL}"
log "  timeouts     : cpt=${CPT_TIMEOUT_HOURS}h merge=${MERGE_TIMEOUT_HOURS}h sft=${SFT_TIMEOUT_HOURS}h upload=${HF_UPLOAD_TIMEOUT_HOURS}h"
log "=========================================="

# System snapshot
{
    echo "--- system snapshot ---"
    date -u
    uname -a
    echo "--- nvidia-smi ---"
    nvidia-smi 2>&1 || echo "(nvidia-smi N/A)"
    echo "--- disk ---"
    df -h /workspace / 2>&1 || true
    echo "--- mem ---"
    free -h 2>&1 || true
    echo "--- python ---"
    command -v python3 || echo "python3 NOT found"
    python3 --version 2>&1 || true
    python3 -m pip --version 2>&1 || true
    echo "--- /workspace ---"
    ls -la /workspace 2>&1 | head -30
    echo "------"
} >> "${LOG_FILE}" 2>&1

[ -z "${HF_TOKEN:-}" ] && { log "[ERROR] HF_TOKEN 미설정"; fail_with_logs "env_error" "${LOG_FILE}" 1; }
[ -z "${HF_USERNAME:-}" ] && { log "[ERROR] HF_USERNAME 미설정"; fail_with_logs "env_error" "${LOG_FILE}" 1; }

required_data_files=("${TRAIN_CPT_JSONL}" "${TRAIN_VAL_JSONL}")
if [ "${SKIP_SFT}" != "1" ] && [ "${SKIP_SFT}" != "true" ]; then
    required_data_files+=("${TRAIN_SFT_INPUT_JSONL}")
fi

for f in "${required_data_files[@]}"; do
    if [ ! -f "${f}" ]; then
        log "[ERROR] 데이터 파일 없음: ${f}"
        fail_with_logs "data_missing" "${LOG_FILE}" 1
    fi
done

# ═══ STEP 1 pip install (pinned, resolver-clean) ═══════════════════════
log "[1/6] pip install"

python3 -m pip install -q --no-cache-dir --upgrade pip >> "${LOG_FILE}" 2>&1 || \
    log "[WARN] pip self-upgrade 실패 — 기존 버전 유지"

python3 -m pip install -q --no-cache-dir \
    "transformers==4.51.3" \
    "peft==0.13.2" \
    "bitsandbytes==0.49.2" \
    "trl==0.12.1" \
    "accelerate==0.34.2" \
    "datasets==2.21.0" \
    "huggingface_hub>=0.30.0,<1.0" \
    "hf_transfer>=0.1.6" \
    "safetensors>=0.4.3" \
    "sentencepiece>=0.2.0" \
    "tokenizers>=0.21,<0.22" \
    "tiktoken>=0.7.0" \
    "protobuf" \
    "wandb>=0.16" \
    "sentry-sdk" \
    "numpy<2.0" \
    "pyyaml>=5.1" \
    >> "${LOG_FILE}" 2>&1
PIP_RC=$?

if [ ${PIP_RC} -ne 0 ]; then
    log "[ERROR] pip 필수 패키지 설치 실패 (rc=${PIP_RC})"
    fail_with_logs "install_failed" "${LOG_FILE}" "${PIP_RC}"
fi

# flash-attn — heavy build, tolerant
python3 -m pip install -q --no-cache-dir --no-build-isolation \
    "flash-attn==2.6.3" >> "${LOG_FILE}" 2>&1 || \
    log "[WARN] flash-attn 설치 실패 — eager attention fallback"

{
    echo "--- pip list (training deps) ---"
    python3 -m pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|trl|accelerate|bitsandbytes|datasets|tokenizers|huggingface|safetensors|sentencepiece|tiktoken|wandb|flash|numpy|protobuf|pyyaml) ' || true
} >> "${LOG_FILE}" 2>&1

# ═══ STEP 1.5 pre-flight smoke test ════════════════════════════════════
# Fails fast with full telemetry BEFORE CPT starts. Anything wrong with the
# env — CUDA mismatch, bnb, flash-attn, transformers-qwen3, HF auth, model
# access — is surfaced here in one shot instead of 5-second CPT crashes.
log "[1.5/6] pre-flight smoke test"
python3 - >> "${PREFLIGHT_LOG}" 2>&1 <<'PREFLIGHT'
import os, sys, traceback, json
steps = []
def step(name, fn):
    try:
        rv = fn()
        steps.append((name, "OK", str(rv)[:200]))
        print(f"[OK ] {name}: {str(rv)[:200]}")
    except Exception as e:
        steps.append((name, "FAIL", repr(e)[:400]))
        print(f"[ERR] {name}: {repr(e)}")
        traceback.print_exc()

step("import torch", lambda: __import__("torch").__version__)
import torch
step("torch.cuda.is_available", lambda: torch.cuda.is_available())
step("torch.version.cuda", lambda: torch.version.cuda)
if torch.cuda.is_available():
    step("device_name", lambda: torch.cuda.get_device_name(0))
    step("device_capability", lambda: torch.cuda.get_device_capability(0))
step("import transformers", lambda: __import__("transformers").__version__)
step("import peft", lambda: __import__("peft").__version__)
step("import bitsandbytes", lambda: __import__("bitsandbytes").__version__)
step("import trl", lambda: __import__("trl").__version__)
step("import accelerate", lambda: __import__("accelerate").__version__)
step("import datasets", lambda: __import__("datasets").__version__)
step("import huggingface_hub", lambda: __import__("huggingface_hub").__version__)
step("import safetensors", lambda: __import__("safetensors").__version__)
try:
    import flash_attn
    step("import flash_attn", lambda: flash_attn.__version__)
except Exception as e:
    steps.append(("import flash_attn", "MISSING", repr(e)[:200]))
    print(f"[MIS] flash_attn: will use eager fallback")

# bnb 4bit quantize runtime check (requires GPU) — this is where many env
# mismatches show up (CUDA vs bnb wheel)
def bnb_op():
    import torch, bitsandbytes as bnb
    import bitsandbytes.functional as F
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    q, state = F.quantize_4bit(x, blocksize=64, quant_type="nf4")
    y = F.dequantize_4bit(q, state)
    return f"q.shape={tuple(q.shape)} y.shape={tuple(y.shape)} mae={float((x-y).abs().mean()):.4f}"
step("bnb quantize 4bit", bnb_op)

# Transformers Qwen3 registration
def qwen3_reg():
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    return f"qwen3 in MAPPING: {'qwen3' in CONFIG_MAPPING}"
step("qwen3 in CONFIG_MAPPING", qwen3_reg)

# HF auth
def hf_whoami():
    from huggingface_hub import HfApi
    api = HfApi()
    return api.whoami(token=os.environ["HF_TOKEN"])["name"]
step("HF whoami", hf_whoami)

# Pull Qwen3-8B-Base config only (tiny, proves access)
def qwen3_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B-Base", trust_remote_code=True)
    return f"arch={cfg.architectures} vocab={cfg.vocab_size} hidden={cfg.hidden_size}"
step("load Qwen3-8B-Base config", qwen3_config)

# Pull Qwen3-8B-Base tokenizer (~7MB, proves BBPE tokenizer loads)
def qwen3_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base", trust_remote_code=True, use_fast=True)
    ids = tok.encode("안녕하세요, 어제 강남 이태원에서 놀았어요.")
    return f"vocab={tok.vocab_size} sample_ids={ids[:8]}..."
step("load Qwen3-8B-Base tokenizer", qwen3_tok)

# Summary
fail = [s for s in steps if s[1] == "FAIL"]
print("="*60)
print(f"preflight result: {len(steps)} steps, {len(fail)} FAIL")
for name, status, detail in steps:
    print(f"  [{status:4}] {name}: {detail}")
sys.exit(2 if fail else 0)
PREFLIGHT
PREFLIGHT_RC=$?

cat "${PREFLIGHT_LOG}" >> "${LOG_FILE}"

if [ ${PREFLIGHT_RC} -ne 0 ]; then
    log "[ERROR] pre-flight smoke test FAIL rc=${PREFLIGHT_RC}"
    fail_with_logs "preflight_failed" "${PREFLIGHT_LOG}" "${PREFLIGHT_RC}"
fi
log "[1.5/6] pre-flight OK"
notify "dalbit preflight OK — starting CPT on ${RUNPOD_POD_ID:-unknown}"

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

# ═══ STEP 2 CPT ════════════════════════════════════════════════════════
log "[2/6] CPT 학습 시작 (base=${BASE_MODEL_CPT})"
notify "dalbit CPT start — ${RUNPOD_POD_ID:-unknown}"

CPT_START=$(date +%s)
run_timeout "${CPT_TIMEOUT_HOURS}" env \
    BASE_MODEL="${BASE_MODEL_CPT}" \
    INPUT_JSONL="${TRAIN_CPT_JSONL}" \
    CPT_VAL_JSONL="${TRAIN_VAL_JSONL}" \
    CPT_LR="${CPT_LR_VALUE}" \
    CPT_NUM_EPOCHS="${CPT_EPOCHS}" \
    CPT_OUTPUT_DIR="${CPT_OUT}" \
    CPT_CKPT_DIR="${OUT_DIR}/cpt-ckpt" \
    CPT_LOG_FILE="${CPT_PY_LOG}" \
    CPT_HUB_MODEL_ID="${HF_REPO_CPT}" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    python3 "${WORKSPACE}/train_cpt.py" >> "${LOG_FILE}" 2>&1
CPT_EXIT=$?
CPT_END=$(date +%s)
CPT_ELAPSED=$(( (CPT_END - CPT_START) / 60 ))

if [ ${CPT_EXIT} -ne 0 ]; then
    log "[ERROR] CPT 실패 exit=${CPT_EXIT} elapsed=${CPT_ELAPSED}m"
    fail_with_logs "cpt_failed" "${CPT_PY_LOG}" "${CPT_EXIT}"
fi
log "[2/6] CPT 완료 (${CPT_ELAPSED}m)"
notify "dalbit CPT done (${CPT_ELAPSED}m)"

# ═══ STEP 3 merge ══════════════════════════════════════════════════════
log "[3/6] CPT adapter → fp16 merge"
BASE_MODEL="${BASE_MODEL_CPT}" \
CPT_LORA_DIR="${CPT_OUT}" \
CPT_MERGED_DIR="${CPT_MERGED}" \
run_timeout "${MERGE_TIMEOUT_HOURS}" env \
    BASE_MODEL="${BASE_MODEL_CPT}" \
    CPT_LORA_DIR="${CPT_OUT}" \
    CPT_MERGED_DIR="${CPT_MERGED}" \
    python3 "${WORKSPACE}/scripts/merge_cpt_to_fp16.py" 2>&1 | tee "${MERGE_PY_LOG}" >> "${LOG_FILE}"
MERGE_EXIT=${PIPESTATUS[0]}
if [ ${MERGE_EXIT} -ne 0 ]; then
    log "[ERROR] merge 실패 exit=${MERGE_EXIT}"
    fail_with_logs "merge_failed" "${MERGE_PY_LOG}" "${MERGE_EXIT}"
fi
log "[3/6] merge 완료"

# ═══ STEP 4 SFT ════════════════════════════════════════════════════════
log "[4/6] SFT stage (base=${CPT_MERGED})"
SFT_STATUS="ok"
SFT_ELAPSED=0
if [ "${SKIP_SFT}" = "1" ] || [ "${SKIP_SFT}" = "true" ]; then
    log "[4/6] SFT skipped by SKIP_SFT=${SKIP_SFT}; CPT-only budget run"
    notify "dalbit SFT skipped (CPT-only budget run)"
    SFT_STATUS="skipped"
else
    notify "dalbit SFT start"
    SFT_START=$(date +%s)
    # A-path resume: paged_adamw_32bit -> paged_adamw_8bit optimizer formats
    # are incompatible, so move ONLY optimizer.pt+scheduler.pt to backup.
    # trainer_state.json (step counter, LR pos) and rng_state.pth stay so
    # HF Trainer does a real resume: step/LR/data-order preserved,
    # only optimizer momentum is reset.
    SFT_CKPT_PARENT_A="${OUT_DIR}/sft-ckpt"
    if [ -d "${SFT_CKPT_PARENT_A}" ]; then
        for d in "${SFT_CKPT_PARENT_A}"/checkpoint-*/; do
            [ -d "${d}" ] || continue
            backup_a="${d%/}/__backup-pre-warm-restart"
            mkdir -p "${backup_a}"
            # If a previous run wrongly moved trainer_state/rng to backup, restore them
            for f in trainer_state.json rng_state.pth; do
                if [ ! -f "${d%/}/${f}" ] && [ -f "${backup_a}/${f}" ]; then
                    mv "${backup_a}/${f}" "${d%/}/${f}"
                    log "[4/6] restored ${f} from backup -> ${d}"
                fi
            done
            # Move only the format-incompatible optimizer state to backup
            for f in optimizer.pt scheduler.pt; do
                if [ -f "${d%/}/${f}" ]; then
                    mv "${d%/}/${f}" "${backup_a}/"
                fi
            done
            log "[4/6] warm-restart prep: optimizer-only backup for ${d}"
        done
    fi
    run_timeout "${SFT_TIMEOUT_HOURS}" env \
    BASE_MODEL="${CPT_MERGED}" \
    SFT_RAW_JSONL="${TRAIN_CPT_JSONL}" \
    SFT_INPUT_JSONL="${TRAIN_SFT_INPUT_JSONL}" \
    SFT_PAIR_JSONL="${TRAIN_SFT_INPUT_JSONL}" \
    SFT_VAL_JSONL="${TRAIN_VAL_JSONL}" \
    SFT_NUM_EPOCHS="${SFT_EPOCHS}" \
    SFT_OUTPUT_DIR="${SFT_OUT}" \
    SFT_CKPT_DIR="${OUT_DIR}/sft-ckpt" \
    SFT_LOG_FILE="${SFT_PY_LOG}" \
    SFT_HUB_MODEL_ID="${HF_REPO_SFT}" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
        python3 "${WORKSPACE}/train_sft.py" >> "${LOG_FILE}" 2>&1
    SFT_EXIT=$?
    SFT_END=$(date +%s)
    SFT_ELAPSED=$(( (SFT_END - SFT_START) / 60 ))

    SFT_STATUS="ok"
    if [ ${SFT_EXIT} -ne 0 ]; then
        log "[WARN] SFT 실패 exit=${SFT_EXIT} — CPT merged만 배포"
        SFT_STATUS="sft_failed"
        # Non-fatal: include SFT log tail in ntfy but continue to upload CPT
        TAIL_BLOB="$(blob_head_tail "${SFT_PY_LOG}" 15 25)"
        notify "dalbit SFT failed rc=${SFT_EXIT} ${TAIL_BLOB}"
    fi
fi

# ═══ STEP 5 HF upload ══════════════════════════════════════════════════
log "[5/6] HF 최종 upload (adapter dirs)"
run_timeout "${HF_UPLOAD_TIMEOUT_HOURS}" env SFT_STATUS="${SFT_STATUS}" python3 - >> "${LOG_FILE}" 2>&1 <<'PYEOF'
import os, sys, time
from huggingface_hub import HfApi, create_repo
api = HfApi()
tok = os.environ["HF_TOKEN"]
cpt_repo = os.environ["HF_REPO_CPT"]
sft_repo = os.environ["HF_REPO_SFT"]
cpt_out = "/workspace/out/cpt-lora"
sft_out = "/workspace/out/sft-lora"
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

# ═══ STEP 6 wrap up ════════════════════════════════════════════════════
CHAIN_END=$(date +%s)
TOTAL_MIN=$(( (CHAIN_END - CPT_START) / 60 ))

FINAL_STATUS="done_ok"
[ "${SFT_STATUS}" = "sft_failed" ] && FINAL_STATUS="done_cpt_only"
[ "${SFT_STATUS}" = "skipped" ] && FINAL_STATUS="done_cpt_only"
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

notify "dalbit chain ${FINAL_STATUS} (${TOTAL_MIN}m) cpt=${HF_REPO_CPT} sft=${HF_REPO_SFT}"
persist_run_artifacts "${FINAL_STATUS}"
stop_pod "${FINAL_STATUS}"
