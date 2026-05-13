#!/usr/bin/env bash
# chain_train_round2.sh - Round-2 cycle-1 RunPod orchestrator.
#
# Phase 1: CPT broad
# Phase 2: CPT clean + DoRA
# Phase 2.5: merge CPT LoRA
# Phase 3: thread-conditioned SFT
# Phase 3.5: merge SFT LoRA
# Phase 4: ORPO preference pass
# Phase 5: eval gate + mutator
#
# This file intentionally mirrors the operational hardening in chain_train.sh:
# explicit dependency install, preflight telemetry, durable logs/DONE.txt,
# failure persistence, and best-effort pod stop.

# Strict mode: -e (exit on error) added per 3-model audit HIGH #4 — paper-grade
# pipeline cannot silently swallow non-essential failures. Each remaining
# `|| true` below carries a comment explaining why best-effort is correct.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
DATA_DIR="${DATA_DIR:-${WORKSPACE}/data}"
OUT_DIR="${OUT_DIR:-${WORKSPACE}/out}"
LOG_DIR="${LOG_DIR:-${WORKSPACE}/logs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${WORKSPACE}/hf_cache}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${WORKSPACE}/scripts}"
REPO_CLONE_DIR="${REPO_CLONE_DIR:-${WORKSPACE}/repo}"
RECIPES_DIR="${RECIPES_DIR:-${WORKSPACE}/recipes}"

ROUND2_LOG="${LOG_DIR}/chain_round2.log"
LOG_FILE="${ROUND2_LOG}"
DONE_FILE="${OUT_DIR}/ROUND2_DONE.txt"
INSTALL_LOG="${WORKSPACE}/round2_install.log"
PREFLIGHT_LOG="${WORKSPACE}/round2_preflight.log"
PHASE1_LOG="${WORKSPACE}/round2_phase1_cpt.log"
PHASE2_LOG="${WORKSPACE}/round2_phase2_cpt.log"
MERGE_CPT_LOG="${WORKSPACE}/round2_merge_cpt.log"
PHASE3_LOG="${WORKSPACE}/round2_phase3_sft.log"
MERGE_SFT_LOG="${WORKSPACE}/round2_merge_sft.log"
PHASE4_LOG="${WORKSPACE}/round2_phase4_orpo.log"
PHASE5_LOG="${WORKSPACE}/round2_phase5_eval.log"
HF_UPLOAD_LOG="${WORKSPACE}/round2_hf_upload.log"

TIMESTAMP="$(date -u '+%Y%m%d-%H%M')"
HF_REPO_ROUND2="${HF_REPO_ROUND2:-${HF_USERNAME:-unoa}/dalbitalba-qwen3-round2-${TIMESTAMP}}"

export WORKSPACE DATA_DIR OUT_DIR LOG_DIR HF_CACHE_DIR SCRIPTS_DIR REPO_CLONE_DIR RECIPES_DIR
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export HF_REPO_ROUND2

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${DATA_DIR}" "${HF_CACHE_DIR}"

log() {
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "${ts} $1" | tee -a "${ROUND2_LOG}"
}

notify() {
    local msg="$1"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        # KEEP || true: outbound notification is best-effort; ntfy outage
        # must not break training.
        curl -fsS -m 10 \
            -H "Content-Type: text/plain; charset=utf-8" \
            --data-binary "${msg}" \
            "https://ntfy.sh/${NTFY_TOPIC}" \
            >> "${ROUND2_LOG}" 2>&1 || true
    fi
}

blob_head_tail() {
    local file="$1"
    local nhead="${2:-15}"
    local ntail="${3:-50}"
    if [ ! -f "${file}" ]; then
        echo "(no ${file})"
        return
    fi
    # File existence guarded above; head/tail must not silently swallow.
    {
        echo "====HEAD===="
        head -n "${nhead}" "${file}" 2>/dev/null
        echo "====TAIL===="
        tail -n "${ntail}" "${file}" 2>/dev/null
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

require() {
    if [ -z "${!1:-}" ]; then
        log "[FATAL] required env var $1 is unset"
        return 2
    fi
}

resolve_pod_id() {
    if [ -n "${RUNPOD_POD_ID:-}" ] && [ "${RUNPOD_POD_ID}" != "__SELF__" ]; then
        printf '%s' "${RUNPOD_POD_ID}"
        return 0
    fi
    # KEEP || true: pod-id resolution is best-effort; absence is handled by
    # callers (printf empty string -> caller logs "unknown").
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
    elif [ "${RUNPOD_POD_ID:-}" = "__SELF__" ]; then
        unset RUNPOD_POD_ID
    fi
    if [ "${ROUND2_STOP_POD:-1}" = "0" ]; then
        log "[stop] ROUND2_STOP_POD=0 -> skip"
        return 0
    fi
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl stop pod "${RUNPOD_POD_ID}" >> "${ROUND2_LOG}" 2>&1 && return 0
    fi
    if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
        # KEEP || true: pod-stop is best-effort cleanup; failure must not
        # mask the underlying training rc that triggered stop_pod.
        curl -fsS -X POST \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
            >> "${ROUND2_LOG}" 2>&1 || true
    fi
}

write_done() {
    local status="$1"
    local extra="${2:-}"
    {
        echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "status=${status}"
        [ -n "${extra}" ] && echo "detail=${extra}"
        echo "base_model=${BASE_MODEL:-}"
        echo "hf_repo_round2=${HF_REPO_ROUND2:-}"
        echo "out_dir=${OUT_DIR}"
        echo "phase1_out=${OUT_DIR}/round2-phase1-cpt-lora"
        echo "phase2_out=${OUT_DIR}/round2-phase2-cpt-lora"
        echo "phase2_merged=${OUT_DIR}/round2-phase2-cpt-merged-fp16"
        echo "phase3_out=${OUT_DIR}/round2-phase3-sft-lora"
        echo "phase3_merged=${OUT_DIR}/round2-phase3-sft-merged-fp16"
        echo "phase4_out=${OUT_DIR}/round2-phase4-orpo"
        echo "phase5_eval=${OUT_DIR}/phase5-eval-v2.json"
        echo "pod_id=${RUNPOD_POD_ID:-unknown}"
    } > "${DONE_FILE}"
    log "[DONE] ${DONE_FILE}"
}

copy_if_file() {
    local src="$1"
    local dst="$2"
    # KEEP || true: artifact-persistence helper; copy failure must not
    # crash a run that's already in fail_with_logs cleanup.
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}" 2>/dev/null || true
    fi
}

persist_run_artifacts() {
    local final_status="$1"
    if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_REPO:-}" ]; then
        log "[runs] GITHUB_TOKEN/GITHUB_REPO unset -> push skip"
        return 0
    fi
    if [ ! -d "${REPO_CLONE_DIR}/.git" ]; then
        log "[runs] repo clone missing -> push skip"
        return 0
    fi

    # KEEP || true: idempotent identity config inside an artifact-push
    # helper; pre-existing identical config is a normal no-op.
    git -C "${REPO_CLONE_DIR}" config user.email "runpod-bot@dalbitalba.local" 2>/dev/null || true
    git -C "${REPO_CLONE_DIR}" config user.name "dalbitalba-runpod" 2>/dev/null || true

    local stamp branch run_dir latest_tmp repo_url
    stamp="$(date -u '+%Y%m%d-%H%M%S')"
    branch="round2-train-run-${stamp}"
    run_dir="${REPO_CLONE_DIR}/runs/${branch}"
    latest_tmp="$(mktemp)"
    mkdir -p "${run_dir}"

    copy_if_file "${DONE_FILE}" "${run_dir}/DONE.txt"
    copy_if_file "${ROUND2_LOG}" "${run_dir}/chain_round2.log"
    copy_if_file "${INSTALL_LOG}" "${run_dir}/install.log"
    copy_if_file "${PREFLIGHT_LOG}" "${run_dir}/preflight.log"
    copy_if_file "${PHASE1_LOG}" "${run_dir}/phase1_cpt.log"
    copy_if_file "${PHASE2_LOG}" "${run_dir}/phase2_cpt.log"
    copy_if_file "${MERGE_CPT_LOG}" "${run_dir}/merge_cpt.log"
    copy_if_file "${PHASE3_LOG}" "${run_dir}/phase3_sft.log"
    copy_if_file "${MERGE_SFT_LOG}" "${run_dir}/merge_sft.log"
    copy_if_file "${PHASE4_LOG}" "${run_dir}/phase4_orpo.log"
    copy_if_file "${PHASE5_LOG}" "${run_dir}/phase5_eval.log"
    copy_if_file "${HF_UPLOAD_LOG}" "${run_dir}/hf_upload.log"
    copy_if_file "${OUT_DIR}/phase5-eval-v2.json" "${run_dir}/phase5-eval-v2.json"
    copy_if_file "${OUT_DIR}/phase5-ai-generated.jsonl" "${run_dir}/phase5-ai-generated.jsonl"

    # A05: collect extended metadata for manifest
    local _git_commit _git_branch _py_ver _torch_ver _transformers_ver _peft_ver _bnb_ver
    local _cpt1_sha _cpt2_sha _sft_sha _val_sha _orpo_sha
    _git_commit="$(git -C "${REPO_CLONE_DIR}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
    _git_branch="$(git -C "${REPO_CLONE_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    _py_ver="$(python3 -c "import sys; print(sys.version.split()[0])" 2>/dev/null || echo 'unknown')"
    _torch_ver="$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 'missing')"
    _transformers_ver="$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo 'missing')"
    _peft_ver="$(python3 -c "import peft; print(peft.__version__)" 2>/dev/null || echo 'missing')"
    _bnb_ver="$(python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null || echo 'missing')"
    # sha256sum on Linux; shasum -a 256 on macOS — try both, fall back to "missing"
    _sha256() { local f="$1"; if [ -f "${f}" ]; then (sha256sum "${f}" 2>/dev/null || shasum -a 256 "${f}" 2>/dev/null) | head -1 | cut -d' ' -f1; else echo "missing"; fi; }
    _cpt1_sha="$(_sha256 "${DATA_DIR}/${CPT_PHASE_1_DATA:-}")"
    _cpt2_sha="$(_sha256 "${DATA_DIR}/${CPT_PHASE_2_DATA:-}")"
    _sft_sha="$(_sha256 "${DATA_DIR}/${SFT_DATA:-}")"
    _val_sha="$(_sha256 "${DATA_DIR}/${SFT_EVAL_DATA:-sft_thread_conditioned.eval.jsonl}")"
    _orpo_sha="$(_sha256 "${DATA_DIR}/${ORPO_DATA:-}")"
    local _docker_image
    _docker_image="${CONTAINER_IMAGE:-$(cat /etc/runpod-release 2>/dev/null | head -1 || echo 'unknown')}"

    cat > "${run_dir}/manifest.json" <<EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "status": "${final_status}",
  "base_model": "${BASE_MODEL:-}",
  "pod_id": "${RUNPOD_POD_ID:-unknown}",
  "hf_repo_round2": "${HF_REPO_ROUND2:-}",
  "source_repo": "${GITHUB_REPO}",
  "budget_profile": "${BUDGET_PROFILE:-budget30}",
  "budget_cap_usd": "${BUDGET_CAP_USD:-60}",
  "wandb_project": "${WANDB_PROJECT:-dalbitalba-round2}",
  "wandb_run_group": "${WANDB_RUN_GROUP:-}",
  "wandb_tags": "${WANDB_TAGS:-}",
  "wandb_username": "${WANDB_USERNAME:-}",
  "git_commit": "${_git_commit}",
  "git_branch": "${_git_branch}",
  "python_version": "${_py_ver}",
  "torch_version": "${_torch_ver}",
  "transformers_version": "${_transformers_ver}",
  "peft_version": "${_peft_ver}",
  "bitsandbytes_version": "${_bnb_ver}",
  "docker_image": "${_docker_image}",
  "base_model_revision": "${BASE_MODEL_REVISION:-main}",
  "data_sha256": {
    "cpt_phase1": "${_cpt1_sha}",
    "cpt_phase2": "${_cpt2_sha}",
    "sft": "${_sft_sha}",
    "val": "${_val_sha}",
    "orpo": "${_orpo_sha}"
  }
}
EOF

    mkdir -p "${REPO_CLONE_DIR}/runs"
    cat > "${latest_tmp}" <<EOF
{
  "branch": "${branch}",
  "status": "${final_status}",
  "hf_repo_round2": "${HF_REPO_ROUND2:-}",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "git_commit": "${_git_commit}",
  "git_branch": "${_git_branch}",
  "python_version": "${_py_ver}",
  "torch_version": "${_torch_ver}",
  "transformers_version": "${_transformers_ver}",
  "peft_version": "${_peft_ver}",
  "bitsandbytes_version": "${_bnb_ver}",
  "docker_image": "${_docker_image}",
  "base_model_revision": "${BASE_MODEL_REVISION:-main}",
  "data_sha256": {
    "cpt_phase1": "${_cpt1_sha}",
    "cpt_phase2": "${_cpt2_sha}",
    "sft": "${_sft_sha}",
    "val": "${_val_sha}",
    "orpo": "${_orpo_sha}"
  }
}
EOF
    cp "${latest_tmp}" "${REPO_CLONE_DIR}/runs/latest-round2-train.json"

    repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
    # KEEP || true on checkout/add: branch may already exist (re-runs);
    # `git add` may be a no-op if files unchanged. Outer (...) result
    # gates the push log message.
    (
        cd "${REPO_CLONE_DIR}" || exit 0
        git checkout -b "${branch}" >/dev/null 2>&1 || true
        git add "runs/${branch}" "runs/latest-round2-train.json" >/dev/null 2>&1 || true
        git commit -m "train: ${branch} (${final_status})" >/dev/null 2>&1 || exit 0
        git push "${repo_url}" "${branch}" >/dev/null 2>&1
    ) && log "[runs] push branch ${branch}" || log "[runs] push failed ${branch}"
    rm -f "${latest_tmp}"
}

fail_with_logs() {
    local status="$1"
    local cfile="$2"
    local rc="${3:-1}"
    local blob
    blob="$(blob_head_tail "${cfile}" 25 35)"
    write_done "${status}" "rc=${rc} log=${cfile}"
    notify "dalbit round2 ${status} rc=${rc} ${blob}"
    persist_run_artifacts "${status}"
    stop_pod "${status}"
    exit "${rc}"
}

on_exit() {
    local rc=$?
    if [ "${rc}" -ne 0 ] && [ ! -f "${DONE_FILE}" ]; then
        write_done "unexpected_exit" "rc=${rc}"
        # KEEP || true: trap handler — cleanup helpers must not re-throw.
        persist_run_artifacts "unexpected_exit" 2>/dev/null || true
        stop_pod "unexpected_exit" 2>/dev/null || true
    fi
}

graceful_abort() {
    local sig="$1"
    log "[abort] signal=${sig}; persisting artifacts then stopping pod"
    if [ ! -f "${DONE_FILE}" ]; then
        write_done "aborted_${sig}" "signal=${sig}"
    fi
    persist_run_artifacts "aborted_${sig}" 2>/dev/null || log "[abort] persist failed"
    stop_pod "aborted_${sig}" 2>/dev/null || log "[abort] stop_pod failed"
    exit 130
}

install_traps() {
    trap 'graceful_abort TERM' TERM
    trap 'graceful_abort INT' INT
    trap 'graceful_abort HUP' HUP
    trap 'graceful_abort QUIT' QUIT
    trap on_exit EXIT
}

resolve_existing_path() {
    local raw="$1"
    if [ -z "${raw}" ]; then
        return 1
    fi
    if [ -e "${raw}" ]; then
        printf '%s' "${raw}"
        return 0
    fi
    if [ -e "${WORKSPACE}/${raw}" ]; then
        printf '%s' "${WORKSPACE}/${raw}"
        return 0
    fi
    if [ -e "${REPO_CLONE_DIR}/${raw}" ]; then
        printf '%s' "${REPO_CLONE_DIR}/${raw}"
        return 0
    fi
    if [ -e "${DATA_DIR}/${raw}" ]; then
        printf '%s' "${DATA_DIR}/${raw}"
        return 0
    fi
    printf '%s' "${raw}"
}

source_env_file() {
    local env_file="$1"
    # shellcheck disable=SC1090
    set -a
    source "${env_file}"
    set +a
}

source_round2_recipe() {
    local launch_profile="${BUDGET_PROFILE:-}"
    local recipe="${ROUND2_RECIPE:-}"
    if [ -z "${recipe}" ]; then
        if [ -f "${RECIPES_DIR}/round2-cycle1.env" ]; then
            recipe="${RECIPES_DIR}/round2-cycle1.env"
        elif [ -f "${REPO_CLONE_DIR}/recipes/round2-cycle1.env" ]; then
            recipe="${REPO_CLONE_DIR}/recipes/round2-cycle1.env"
        elif [ -f "recipes/round2-cycle1.env" ]; then
            recipe="recipes/round2-cycle1.env"
        fi
    fi
    if [ -z "${recipe}" ] || [ ! -f "${recipe}" ]; then
        log "[FATAL] recipes/round2-cycle1.env not found; copy from repo or set ROUND2_RECIPE"
        return 2
    fi
    # R3 BLOCKER fix: recipe vars (SFT_LOSS_WEIGHT_*, BASE_MODEL, ORPO_*,
    # CPT_*, etc.) must reach Python subprocess builders such as
    # scripts/round2_build_tc_sft.py. Plain `source` only sets shell vars,
    # not env vars, so subprocesses spawned by this script wouldn't see
    # them. `set -a` toggles auto-export so every assignment in the recipe
    # becomes an exported env var; `set +a` restores normal behaviour.
    source_env_file "${recipe}"
    log "[recipe] ${recipe} (auto-exported via set -a)"

    # RunPod launch profiles (smoke/budget30) are partial overlays. The base
    # round2 recipe is still sourced first for required defaults, then the
    # launch-requested profile is re-applied so paid-run controls such as
    # SKIP_SFT, CPT limits, timeouts, and CPT_PHASE_* data choices cannot be
    # silently overwritten by round2-cycle1.env inside the container.
    if [ -n "${launch_profile}" ] && [ "${launch_profile}" != "paper8b" ]; then
        local profile_recipe=""
        local candidate
        for candidate in \
            "${RECIPES_DIR}/${launch_profile}.env" \
            "${REPO_CLONE_DIR}/recipes/${launch_profile}.env" \
            "recipes/${launch_profile}.env"; do
            if [ -f "${candidate}" ]; then
                profile_recipe="${candidate}"
                break
            fi
        done
        if [ -z "${profile_recipe}" ]; then
            log "[FATAL] BUDGET_PROFILE=${launch_profile} but ${launch_profile}.env not found"
            return 2
        fi
        if [ "${profile_recipe}" != "${recipe}" ]; then
            source_env_file "${profile_recipe}"
            log "[recipe-profile] ${profile_recipe} (launch overlay re-applied)"
        fi
    fi
    log "[recipe-effective] budget=${BUDGET_PROFILE:-unset} cpt_phase1=${CPT_PHASE_1_DATA:-unset} cpt_phase2=${CPT_PHASE_2_DATA:-unset} skip_sft=${SKIP_SFT:-0} sft_epochs=${SFT_NUM_EPOCHS:-unset}"
}

system_snapshot() {
    # KEEP || true on diagnostics below: snapshot must tolerate missing
    # tooling on heterogeneous runtimes (e.g. `free` absent on macOS dev
    # workstations); diagnostics are informational, not load-bearing.
    {
        echo "--- system snapshot ---"
        date -u
        uname -a
        echo "--- nvidia-smi ---"
        nvidia-smi 2>&1 || echo "(nvidia-smi N/A)"
        echo "--- disk ---"
        df -h "${WORKSPACE}" / 2>&1 || true
        echo "--- mem ---"
        free -h 2>&1 || true
        echo "--- python ---"
        command -v python3 || echo "python3 NOT found"
        python3 --version 2>&1 || true
        python3 -m pip --version 2>&1 || true
        echo "--- workspace ---"
        ls -la "${WORKSPACE}" 2>&1 | head -60
        echo "--- data ---"
        ls -lh "${DATA_DIR}" 2>&1 | head -80
        echo "------"
    } >> "${ROUND2_LOG}" 2>&1
}

install_deps() {
    log "[install] pip install training dependencies"
    python3 -m pip install -q --no-cache-dir --upgrade pip >> "${INSTALL_LOG}" 2>&1 || \
        log "[WARN] pip self-upgrade failed; continuing with existing pip"

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
        >> "${INSTALL_LOG}" 2>&1
    local rc=$?
    cat "${INSTALL_LOG}" >> "${ROUND2_LOG}"
    if [ ${rc} -ne 0 ]; then
        fail_with_logs "install_failed" "${INSTALL_LOG}" "${rc}"
    fi

    python3 -m pip install -q --no-cache-dir --no-build-isolation \
        "flash-attn==2.6.3" >> "${INSTALL_LOG}" 2>&1 || \
        log "[WARN] flash-attn install failed; training scripts should fall back if configured"

    # KEEP || true on grep below: empty match (no rows after filter) is
    # exit 1 and not a real failure; diagnostic only.
    {
        echo "--- pip list (training deps) ---"
        python3 -m pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|trl|accelerate|bitsandbytes|datasets|tokenizers|huggingface|safetensors|sentencepiece|tiktoken|wandb|flash|numpy|protobuf|pyyaml) ' || true
    } >> "${ROUND2_LOG}" 2>&1
}

preflight() {
    log "[preflight] dependency/CUDA/HF/model smoke"
    BASE_MODEL="${BASE_MODEL}" python3 - >> "${PREFLIGHT_LOG}" 2>&1 <<'PY'
import os
import sys
import traceback

steps = []

def step(name, fn, fatal=True):
    try:
        value = fn()
        steps.append((name, "OK", str(value)[:240], fatal))
        print(f"[OK ] {name}: {str(value)[:240]}")
    except Exception as exc:
        status = "FAIL" if fatal else "WARN"
        steps.append((name, status, repr(exc)[:400], fatal))
        print(f"[{status}] {name}: {repr(exc)}")
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
step("import flash_attn", lambda: __import__("flash_attn").__version__, fatal=False)

def bnb_op():
    import bitsandbytes.functional as F
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    q, state = F.quantize_4bit(x, blocksize=64, quant_type="nf4")
    y = F.dequantize_4bit(q, state)
    return f"q.shape={tuple(q.shape)} y.shape={tuple(y.shape)} mae={float((x-y).abs().mean()):.4f}"

step("bnb quantize 4bit", bnb_op)

def hf_whoami():
    from huggingface_hub import HfApi
    return HfApi().whoami(token=os.environ["HF_TOKEN"])["name"]

step("HF whoami", hf_whoami)

base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base")

def model_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    return f"model={base_model} arch={cfg.architectures} vocab={cfg.vocab_size}"

step("load base config", model_config)

def model_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True, token=os.environ.get("HF_TOKEN"))
    ids = tok.encode("안녕하세요, 어제 강남 이태원에서 놀았어요.")
    return f"vocab={tok.vocab_size} sample_ids={ids[:8]}..."

step("load base tokenizer", model_tokenizer)

fatal_fail = [s for s in steps if s[1] == "FAIL" and s[3]]
print("=" * 60)
print(f"preflight result: {len(steps)} steps, {len(fatal_fail)} fatal FAIL")
for name, status, detail, _fatal in steps:
    print(f"  [{status:4}] {name}: {detail}")
sys.exit(2 if fatal_fail else 0)
PY
    local rc=$?
    cat "${PREFLIGHT_LOG}" >> "${ROUND2_LOG}"
    if [ ${rc} -ne 0 ]; then
        fail_with_logs "preflight_failed" "${PREFLIGHT_LOG}" "${rc}"
    fi
    log "[preflight] OK"
    notify "dalbit round2 preflight OK - starting training on ${RUNPOD_POD_ID:-unknown} | wandb: project=${WANDB_PROJECT:-dalbitalba-round2} group=${WANDB_RUN_GROUP:-} url=https://wandb.ai/${WANDB_USERNAME:-anonymous}/${WANDB_PROJECT:-dalbitalba-round2}"
}

validate_inputs() {
    require BASE_MODEL || return 2
    require CPT_PHASE_1_DATA || return 2
    require CPT_PHASE_2_DATA || return 2
    require SFT_DATA || return 2
    require SFT_EVAL_DATA || return 2
    require ORPO_DATA || return 2
    [ -z "${HF_TOKEN:-}" ] && { log "[ERROR] HF_TOKEN unset"; return 2; }
    [ -z "${HF_USERNAME:-}" ] && { log "[ERROR] HF_USERNAME unset"; return 2; }

    local missing=0
    for f in "${DATA_DIR}/${CPT_PHASE_1_DATA}" "${DATA_DIR}/${CPT_PHASE_2_DATA}" "${DATA_DIR}/${SFT_DATA}" "${DATA_DIR}/${SFT_EVAL_DATA:-sft_thread_conditioned.eval.jsonl}"; do
        if [ ! -f "${f}" ]; then
            log "[ERROR] required data file missing: ${f}"
            missing=1
        fi
    done
    if [ ${missing} -ne 0 ]; then
        return 2
    fi
}

phase1_cpt_broad() {
    log "=== Phase 1: CPT broad ==="
    local out_dir="${OUT_DIR}/round2-phase1-cpt-lora"
    local input="${DATA_DIR}/${CPT_PHASE_1_DATA}"
    mkdir -p "${out_dir}"
    if [ ! -f "${input}" ]; then
        log "[FATAL] phase-1 input missing: ${input}"
        return 2
    fi
    run_timeout "${CPT_TIMEOUT_HOURS:-36}" env \
        INPUT_JSONL="${input}" \
        BASE_MODEL="${BASE_MODEL}" \
        CPT_NUM_EPOCHS="${CPT_NUM_EPOCHS:-1}" \
        CPT_LR="${CPT_LR:-2e-4}" \
        CPT_WARMUP_RATIO="${CPT_WARMUP_RATIO:-0.03}" \
        CPT_LORA_R="${LORA_R:-64}" \
        CPT_LORA_ALPHA="${LORA_ALPHA:-128}" \
        CPT_USE_DORA="0" \
        CPT_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
        CPT_OUTPUT_DIR="${out_dir}" \
        CPT_CKPT_DIR="${OUT_DIR}/round2-phase1-cpt-ckpt" \
        CPT_LOG_FILE="${PHASE1_LOG}" \
        WANDB_NAME="${WANDB_NAME:-phase1-cpt-broad}" \
        WANDB_ENTITY="${WANDB_ENTITY:-}" \
        TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-none}" \
        WANDB_PROJECT="${WANDB_PROJECT:-dalbitalba-round2}" \
        WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}" \
        WANDB_RESUME="${WANDB_RESUME:-allow}" \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        python3 "${SCRIPTS_DIR}/../train_cpt.py" 2>&1 | tee -a "${PHASE1_LOG}" >> "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase1 exit=${rc}"
    [ "${rc}" = "0" ] && upload_phase_adapter "phase1-cpt-broad" "${OUT_DIR}/round2-phase1-cpt-lora" "${HF_USERNAME:-unoa}/dalbitalba-qwen3-round2-${TIMESTAMP}-phase1"
    return ${rc}
}

phase2_cpt_clean() {
    log "=== Phase 2: CPT clean + DoRA ==="
    local out_dir="${OUT_DIR}/round2-phase2-cpt-lora"
    local input="${DATA_DIR}/${CPT_PHASE_2_DATA}"
    mkdir -p "${out_dir}"
    if [ ! -f "${input}" ]; then
        log "[FATAL] phase-2 input missing: ${input}"
        return 2
    fi
    run_timeout "${CPT_TIMEOUT_HOURS:-36}" env \
        INPUT_JSONL="${input}" \
        BASE_MODEL="${BASE_MODEL}" \
        CPT_NUM_EPOCHS="${CPT_NUM_EPOCHS:-1}" \
        CPT_LR="${CPT_LR:-2e-4}" \
        CPT_WARMUP_RATIO="${CPT_WARMUP_RATIO:-0.03}" \
        CPT_LORA_R="128" \
        CPT_LORA_ALPHA="128" \
        CPT_USE_DORA="1" \
        CPT_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
        CPT_OUTPUT_DIR="${out_dir}" \
        CPT_CKPT_DIR="${OUT_DIR}/round2-phase2-cpt-ckpt" \
        CPT_LOG_FILE="${PHASE2_LOG}" \
        WANDB_NAME="${WANDB_NAME:-phase2-cpt-dora}" \
        WANDB_ENTITY="${WANDB_ENTITY:-}" \
        TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-none}" \
        WANDB_PROJECT="${WANDB_PROJECT:-dalbitalba-round2}" \
        WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}" \
        WANDB_RESUME="${WANDB_RESUME:-allow}" \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        python3 "${SCRIPTS_DIR}/../train_cpt.py" 2>&1 | tee -a "${PHASE2_LOG}" >> "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase2 exit=${rc}"
    [ "${rc}" = "0" ] && upload_phase_adapter "phase2-cpt-clean" "${OUT_DIR}/round2-phase2-cpt-lora" "${HF_USERNAME:-unoa}/dalbitalba-qwen3-round2-${TIMESTAMP}-phase2"
    return ${rc}
}

phase2_5_merge_cpt() {
    log "=== Phase 2.5: CPT LoRA merge ==="
    local cpt_lora="${OUT_DIR}/round2-phase2-cpt-lora"
    local cpt_merged="${OUT_DIR}/round2-phase2-cpt-merged-fp16"
    if [ ! -d "${cpt_lora}" ]; then
        log "[FATAL] phase-2 CPT lora dir missing: ${cpt_lora}"
        return 2
    fi
    run_timeout "${MERGE_TIMEOUT_HOURS:-8}" env \
        BASE_MODEL="${BASE_MODEL}" \
        CPT_LORA_DIR="${cpt_lora}" \
        CPT_MERGED_DIR="${cpt_merged}" \
        python3 "${SCRIPTS_DIR}/merge_cpt_to_fp16.py" 2>&1 | tee -a "${MERGE_CPT_LOG}" >> "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    if [ ${rc} -eq 0 ] && [ -d "${cpt_merged}" ]; then
        export SFT_BASE_MODEL="${cpt_merged}"
        log "phase2.5 merge OK: Phase 3 SFT base=${SFT_BASE_MODEL}"
        return 0
    fi
    log "[FATAL] phase2.5 merge failed rc=${rc}"
    return ${rc:-2}
}

phase3_sft_threaded() {
    log "=== Phase 3: TC-SFT + Persona + Loss-Weight ==="
    local sft_data="${DATA_DIR}/${SFT_DATA}"
    local persona_list
    persona_list="$(resolve_existing_path "${SFT_PERSONA_LIST:-runs/round2-obsidian-synthesis/persona-30-extracted.json}")"

    # R3 BLOCKER: manifest-hash check. SFT_LOSS_WEIGHT_* env vars and dedup
    # toggle change which rows get loss_weight escalation, so a stale
    # ${sft_data} from a previous run with different env must be rebuilt.
    # We persist the env footprint to ${sft_data}.manifest and rebuild when
    # the SFT data is missing or an existing manifest differs. Fresh clones may
    # have committed SFT data without the sidecar; that path is reused and a
    # run-local sidecar is written instead of forcing a source_db_cache rebuild.
    local _SFT_MANIFEST="weight=${SFT_LOSS_WEIGHT_ARGOT:-1.5}_thresh=${SFT_LOSS_WEIGHT_THRESHOLD:-2}_terms=${SFT_LOSSWEIGHT_TERMS_FOOTPRINT:-${SFT_LOSS_WEIGHT_TERMS:-default}}_dedup=${SFT_APPLY_DEDUP:-1}"
    local _SFT_MANIFEST_FILE="${sft_data}.manifest"
    local _STORED_MANIFEST=""
    [ -f "${_SFT_MANIFEST_FILE}" ] && _STORED_MANIFEST="$(cat "${_SFT_MANIFEST_FILE}" 2>/dev/null || true)"
    if [ -f "${sft_data}" ] && [ -z "${_STORED_MANIFEST}" ]; then
        if [ "${SFT_STRICT_MANIFEST:-0}" = "1" ]; then
            log "[FATAL] ${_SFT_MANIFEST_FILE} missing and SFT_STRICT_MANIFEST=1"
            return 2
        fi
        log "[WARN] ${_SFT_MANIFEST_FILE} missing; reusing committed ${SFT_DATA} and writing run-local manifest"
        echo "${_SFT_MANIFEST}" > "${_SFT_MANIFEST_FILE}"
        _STORED_MANIFEST="${_SFT_MANIFEST}"
    fi
    if [ ! -f "${sft_data}" ] || [ "${_SFT_MANIFEST}" != "${_STORED_MANIFEST}" ]; then
        if [ ! -f "${sft_data}" ]; then
            log "[INFO] ${SFT_DATA} missing; building from cpt_context_stream"
        else
            log "[INFO] Rebuilding SFT data (manifest changed: stored='${_STORED_MANIFEST}' vs current='${_SFT_MANIFEST}')"
        fi
        local _DEDUP_FLAG="--apply-dedup"
        local _RAW_SOURCE_DIR="${DATA_DIR}/source_db_cache"
        if [ "${SFT_APPLY_DEDUP:-1}" = "0" ]; then
            _DEDUP_FLAG="--no-apply-dedup"
        fi
        if [ ! -d "${_RAW_SOURCE_DIR}" ]; then
            log "[FATAL] tc-sft rebuild requested but raw source dir missing: ${_RAW_SOURCE_DIR}"
            return 2
        fi
        if [ ! -f "${DATA_DIR}/cpt_context_stream.jsonl" ]; then
            log "[FATAL] tc-sft rebuild requested but context stream missing: ${DATA_DIR}/cpt_context_stream.jsonl"
            return 2
        fi
        python3 "${SCRIPTS_DIR}/round2_build_tc_sft.py" \
            --context-stream "${DATA_DIR}/cpt_context_stream.jsonl" \
            --raw-source-dir "${_RAW_SOURCE_DIR}" \
            --persona-list "${persona_list}" \
            ${_DEDUP_FLAG} \
            --out "${sft_data}" 2>&1 | tee -a "${PHASE3_LOG}" >> "${ROUND2_LOG}"
        local build_rc=${PIPESTATUS[0]}
        if [ ${build_rc} -ne 0 ] || [ ! -f "${sft_data}" ]; then
            log "[FATAL] tc-sft build failed rc=${build_rc}"
            return ${build_rc:-2}
        fi
        echo "${_SFT_MANIFEST}" > "${_SFT_MANIFEST_FILE}"
        log "[INFO] Wrote SFT manifest: ${_SFT_MANIFEST_FILE}"
    else
        log "[INFO] SFT data manifest matches; reusing ${sft_data}"
    fi
    local out_dir="${OUT_DIR}/round2-phase3-sft-lora"
    local sft_base="${SFT_BASE_MODEL:-${BASE_MODEL}}"
    mkdir -p "${out_dir}"
    run_timeout "${SFT_TIMEOUT_HOURS:-96}" env \
        SFT_PAIR_JSONL="${sft_data}" \
        SFT_RAW_JSONL="${DATA_DIR}/${CPT_PHASE_2_DATA}" \
        SFT_VAL_JSONL="${DATA_DIR}/${SFT_EVAL_DATA:-sft_thread_conditioned.eval.jsonl}" \
        SFT_RAW_RATIO="${SFT_RAW_RATIO:-0.0}" \
        BASE_MODEL="${sft_base}" \
        SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-2}" \
        SFT_LORA_R="${LORA_R:-128}" \
        SFT_LORA_ALPHA="${LORA_ALPHA:-256}" \
        SFT_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
        SFT_LOSS_WEIGHT_ARGOT="${SFT_LOSS_WEIGHT_ARGOT:-1.5}" \
        SFT_LOSS_WEIGHT_THRESHOLD="${SFT_LOSS_WEIGHT_THRESHOLD:-2}" \
        SFT_OUTPUT_DIR="${out_dir}" \
        SFT_CKPT_DIR="${OUT_DIR}/round2-phase3-sft-ckpt" \
        SFT_LOG_FILE="${PHASE3_LOG}" \
        WANDB_NAME="${WANDB_NAME:-phase3-tc-sft}" \
        WANDB_ENTITY="${WANDB_ENTITY:-}" \
        TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-none}" \
        WANDB_PROJECT="${WANDB_PROJECT:-dalbitalba-round2}" \
        WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}" \
        WANDB_RESUME="${WANDB_RESUME:-allow}" \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        python3 "${SCRIPTS_DIR}/../train_sft.py" 2>&1 | tee -a "${PHASE3_LOG}" >> "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase3 exit=${rc} (base=${sft_base})"
    [ "${rc}" = "0" ] && upload_phase_adapter "phase3-tc-sft" "${OUT_DIR}/round2-phase3-sft-lora" "${HF_USERNAME:-unoa}/dalbitalba-qwen3-round2-${TIMESTAMP}-phase3"
    return ${rc}
}

phase3_5_merge_sft() {
    log "=== Phase 3.5: SFT LoRA merge ==="
    local sft_lora="${OUT_DIR}/round2-phase3-sft-lora"
    local sft_merged="${OUT_DIR}/round2-phase3-sft-merged-fp16"
    if [ ! -d "${sft_lora}" ]; then
        log "[INFO] phase-3 SFT lora dir missing; skip merge, ORPO will use BASE_MODEL"
        return 0
    fi
    run_timeout "${MERGE_TIMEOUT_HOURS:-8}" env \
        BASE_MODEL="${SFT_BASE_MODEL:-${BASE_MODEL}}" \
        SFT_LORA_DIR="${sft_lora}" \
        SFT_MERGED_DIR="${sft_merged}" \
        python3 "${SCRIPTS_DIR}/merge_sft_to_fp16.py" 2>&1 | tee -a "${MERGE_SFT_LOG}" >> "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    if [ ${rc} -eq 0 ] && [ -d "${sft_merged}" ]; then
        export ORPO_BASE_MODEL="${sft_merged}"
        log "phase3.5 merge OK: ORPO will use ${sft_merged}"
    else
        log "[WARN] phase3.5 merge failed rc=${rc}; ORPO falls back to BASE_MODEL"
    fi
    return 0
}

phase4_orpo() {
    log "=== Phase 4: ORPO ==="
    if [ "${ORPO_NUM_EPOCHS:-0}" = "0" ]; then
        log "[INFO] ORPO_NUM_EPOCHS=0; Phase 4 deferred until real judged preference pairs are available"
        return 0
    fi
    local orpo_data="${DATA_DIR}/${ORPO_DATA}"
    if [ ! -f "${orpo_data}" ]; then
        log "[INFO] ${ORPO_DATA} missing; building from runs/refinement-*"
        python3 "${SCRIPTS_DIR}/round2_build_orpo_pairs.py" \
            --runs-glob "${REPO_CLONE_DIR}/runs/refinement-2026042*" \
            --val-set "${DATA_DIR}/${SFT_EVAL_DATA:-sft_thread_conditioned.eval.jsonl}" \
            --cpt-corpus "${DATA_DIR}/${CPT_PHASE_2_DATA:-cpt_corpus.v3.jsonl}" \
            --out "${orpo_data}" 2>&1 | tee -a "${PHASE4_LOG}" >> "${ROUND2_LOG}"
        local build_rc=${PIPESTATUS[0]}
        if [ ${build_rc} -ne 0 ] || [ ! -f "${orpo_data}" ]; then
            # If ORPO is enabled (epochs > 0), build failure is FATAL —
            # otherwise the "5-phase paper-grade" run silently completes
            # without Phase 4 supervision. (3-model audit BLOCKER)
            if [ "${ORPO_NUM_EPOCHS:-0}" -gt 0 ]; then
                log "[FATAL] ORPO pair build failed rc=${build_rc} and ORPO_NUM_EPOCHS=${ORPO_NUM_EPOCHS}>0"
                fail_with_logs "phase4_orpo_build_failed" "${PHASE4_LOG}" "${build_rc:-2}"
            fi
            log "[INFO] ORPO disabled (ORPO_NUM_EPOCHS=${ORPO_NUM_EPOCHS:-0}); build failure ignored rc=${build_rc}"
            return 0
        fi
    fi
    if [ ! -f "${SCRIPTS_DIR}/../train_orpo.py" ]; then
        log "[INFO] train_orpo.py missing; Phase 4 deferred"
        return 0
    fi
    local out_dir="${OUT_DIR}/round2-phase4-orpo"
    local orpo_base="${ORPO_BASE_MODEL:-${BASE_MODEL}}"
    mkdir -p "${out_dir}"
    run_timeout "${ORPO_TIMEOUT_HOURS:-48}" env \
        ORPO_DATA="${orpo_data}" \
        ORPO_NUM_EPOCHS="${ORPO_NUM_EPOCHS:-0}" \
        ORPO_BETA="${ORPO_BETA:-0.1}" \
        ORPO_OUTPUT_DIR="${out_dir}" \
        ORPO_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
        WANDB_NAME="${WANDB_NAME:-phase4-orpo}" \
        WANDB_ENTITY="${WANDB_ENTITY:-}" \
        TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-none}" \
        WANDB_PROJECT="${WANDB_PROJECT:-dalbitalba-round2}" \
        WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}" \
        WANDB_RESUME="${WANDB_RESUME:-allow}" \
        BASE_MODEL="${orpo_base}" \
        python3 "${SCRIPTS_DIR}/../train_orpo.py" 2>&1 | tee -a "${PHASE4_LOG}" >> "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase4 exit=${rc} (base=${orpo_base})"
    return ${rc}
}

phase5_eval_gate() {
    log "=== Phase 5: phase6_eval_v2 ==="
    local ai="${OUT_DIR}/phase5-ai-generated.jsonl"
    local eval_base="${SFT_BASE_MODEL:-${BASE_MODEL}}"
    local eval_adapter="${OUT_DIR}/round2-phase3-sft-lora"
    local eval_cpt_merged=""
    local persona_list
    persona_list="$(resolve_existing_path "${EVAL_PERSONA_LIST:-runs/round2-obsidian-synthesis/persona-30-extracted.json}")"
    if [ -d "${OUT_DIR}/round2-phase4-orpo" ] && [ -d "${OUT_DIR}/round2-phase3-sft-merged-fp16" ]; then
        eval_base="${OUT_DIR}/round2-phase3-sft-merged-fp16"
        eval_adapter="${OUT_DIR}/round2-phase4-orpo"
    fi
    if [ "${SKIP_SFT:-0}" = "1" ] || [ "${SFT_NUM_EPOCHS:-2}" = "0" ]; then
        eval_cpt_merged="${OUT_DIR}/round2-phase2-cpt-merged-fp16"
        eval_base="${BASE_MODEL}"
        eval_adapter=""
        if [ ! -d "${eval_cpt_merged}" ]; then
            log "[FATAL] SKIP_SFT=1 but CPT merged model missing: ${eval_cpt_merged}"
            return 2
        fi
        log "[eval] CPT-only profile detected; using CPT_MERGED_PATH=${eval_cpt_merged}"
    fi
    if [ ! -f "${ai}" ]; then
        if [ -n "${eval_adapter}" ] && [ ! -d "${eval_adapter}" ]; then
            log "[FATAL] no eval adapter found: ${eval_adapter}"
            return 2
        fi
        log "[eval] generating samples base=${eval_base} adapter=${eval_adapter:-none} cpt_merged=${eval_cpt_merged:-none}"
        run_timeout "${EVAL_TIMEOUT_HOURS:-12}" env \
            BASE_MODEL="${eval_base}" \
            SFT_ADAPTER_REPO="${eval_adapter}" \
            CPT_MERGED_PATH="${eval_cpt_merged}" \
            EVAL_INPUT_JSONL="${DATA_DIR}/${EVAL_INPUT_DATA:-${SFT_EVAL_DATA:-sft_thread_conditioned.eval.jsonl}}" \
            EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-500}" \
            MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-400}" \
            TEMPERATURE="${GENERATION_TEMP:-0.6}" \
            TOP_P="${GENERATION_TOP_P:-0.9}" \
            python3 "${SCRIPTS_DIR}/phase6_generate.py" 2>&1 | tee -a "${PHASE5_LOG}" >> "${ROUND2_LOG}"
        local gen_rc=${PIPESTATUS[0]}
        if [ ${gen_rc} -ne 0 ] || [ ! -f "${WORKSPACE}/ai_generated.jsonl" ]; then
            log "[FATAL] phase5 generation failed rc=${gen_rc}"
            return ${gen_rc:-2}
        fi
        cp "${WORKSPACE}/ai_generated.jsonl" "${ai}"
    fi
    # Phase 5.9.1: MAUVE eval gate enabled by default in production.
    # --skip-mauve is now ENV-GATED — only passed when MAUVE_DISABLED=1 is set
    # explicitly (e.g. for smoke tests / dev envs without mauve-text installed).
    # Default behavior runs full MAUVE evaluation as required for paper-grade gates.
    local mauve_flag=()
    if [ "${MAUVE_DISABLED:-0}" = "1" ]; then
        log "[eval] MAUVE_DISABLED=1 -> passing --skip-mauve (env-gated, non-default)"
        mauve_flag+=("--skip-mauve")
    fi
    python3 "${SCRIPTS_DIR}/phase6_eval_v2.py" \
        --ai "${ai}" \
        --raw "${DATA_DIR}/${EVAL_INPUT_DATA:-${SFT_EVAL_DATA:-sft_thread_conditioned.eval.jsonl}}" \
        --persona-list "${persona_list}" \
        --out "${OUT_DIR}/phase5-eval-v2.json" \
        "${mauve_flag[@]}" 2>&1 | tee -a "${PHASE5_LOG}" >> "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase5 exit=${rc}"
    if [ -f "${OUT_DIR}/phase5-eval-v2.json" ]; then
        # KEEP || true: mutator suggests the next recipe; failure must
        # not mask the underlying eval rc that we're about to return.
        python3 "${SCRIPTS_DIR}/round2_mutator.py" \
            --metrics "${OUT_DIR}/phase5-eval-v2.json" \
            >> "${PHASE5_LOG}" 2>&1 || true
        cat "${PHASE5_LOG}" >> "${ROUND2_LOG}"
    fi
    return ${rc}
}

upload_phase_adapter() {
    local phase_label="$1"      # e.g. phase1-cpt-broad
    local adapter_dir="$2"      # e.g. ${OUT_DIR}/round2-phase1-cpt-lora
    local hf_repo="$3"          # e.g. ${HF_USERNAME}/dalbitalba-qwen3-round2-${TIMESTAMP}-${phase_label}
    if [ ! -d "${adapter_dir}" ] || [ -z "${HF_TOKEN:-}" ]; then
        log "[hf-incremental] skip ${phase_label}: dir or token missing"
        return 0
    fi
    log "[hf-incremental] uploading ${phase_label} → ${hf_repo}"
    # KEEP || true: incremental upload is best-effort. Final phase5 push remains authoritative.
    run_timeout "${HF_UPLOAD_TIMEOUT_HOURS:-1}" python3 -c "
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
repo = '${hf_repo}'
try:
    api.create_repo(repo, repo_type='model', private=True, exist_ok=True)
    api.upload_folder(folder_path='${adapter_dir}', repo_id=repo, repo_type='model', commit_message='incremental: ${phase_label}')
    print('[hf-incremental] uploaded ${phase_label}')
except Exception as e:
    print(f'[hf-incremental] WARN {e}', file=sys.stderr)
    sys.exit(1)
" >> "${ROUND2_LOG}" 2>&1 || notify "WARN: HF incremental ${phase_label} upload failed (chain continues)"
}

upload_hf_artifacts() {
    if [ "${ROUND2_SKIP_HF_UPLOAD:-0}" = "1" ]; then
        log "[hf] ROUND2_SKIP_HF_UPLOAD=1 -> skip"
        return 0
    fi
    if [ -z "${HF_TOKEN:-}" ] || [ -z "${HF_REPO_ROUND2:-}" ]; then
        log "[hf] HF_TOKEN/HF_REPO_ROUND2 unset -> skip"
        return 0
    fi
    log "[hf] upload round2 adapter/eval artifacts to ${HF_REPO_ROUND2}"
    run_timeout "${HF_UPLOAD_TIMEOUT_HOURS:-4}" python3 - >> "${HF_UPLOAD_LOG}" 2>&1 <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
repo_id = os.environ["HF_REPO_ROUND2"]
out_dir = Path(os.environ.get("OUT_DIR", "/workspace/out"))
api = HfApi()
create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)

folders = {
    "round2-phase1-cpt-lora": out_dir / "round2-phase1-cpt-lora",
    "round2-phase2-cpt-lora": out_dir / "round2-phase2-cpt-lora",
    "round2-phase3-sft-lora": out_dir / "round2-phase3-sft-lora",
    "round2-phase4-orpo": out_dir / "round2-phase4-orpo",
}
for path_in_repo, folder in folders.items():
    if folder.is_dir() and any(folder.iterdir()):
        print(f"[upload_folder] {folder} -> {repo_id}/{path_in_repo}")
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            token=token,
            repo_type="model",
        )

files = {
    "eval/phase5-eval-v2.json": out_dir / "phase5-eval-v2.json",
    "eval/phase5-ai-generated.jsonl": out_dir / "phase5-ai-generated.jsonl",
    "DONE.txt": out_dir / "ROUND2_DONE.txt",
}
for path_in_repo, file_path in files.items():
    if file_path.is_file():
        print(f"[upload_file] {file_path} -> {repo_id}/{path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(file_path),
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            token=token,
            repo_type="model",
        )
print("[hf] upload complete")
PY
    local rc=$?
    # KEEP || true: log-append is best-effort; cat may race with rotation.
    cat "${HF_UPLOAD_LOG}" >> "${ROUND2_LOG}" 2>/dev/null || true
    if [ ${rc} -ne 0 ]; then
        log "[WARN] HF artifact upload failed rc=${rc}"
        return ${rc}
    fi
    log "[hf] upload complete"
    return 0
}

run_main() {
    source_round2_recipe || fail_with_logs "recipe_missing" "${ROUND2_LOG}" 2

    log "=========================================="
    log "chain_train_round2 cycle-1 START"
    log "pod=${RUNPOD_POD_ID:-unknown}"
    log "base=${BASE_MODEL:-unset}"
    log "hf_repo_round2=${HF_REPO_ROUND2}"
    log "budget=${BUDGET_PROFILE:-budget30} cap_usd=${BUDGET_CAP_USD:-60}"
    log "timeouts cpt=${CPT_TIMEOUT_HOURS:-36}h merge=${MERGE_TIMEOUT_HOURS:-8}h sft=${SFT_TIMEOUT_HOURS:-96}h orpo=${ORPO_TIMEOUT_HOURS:-48}h eval=${EVAL_TIMEOUT_HOURS:-12}h upload=${HF_UPLOAD_TIMEOUT_HOURS:-4}h"
    log "=========================================="

    system_snapshot
    validate_inputs || fail_with_logs "env_or_data_error" "${ROUND2_LOG}" 2
    install_deps
    preflight

    # cycle-7 US-C704: spawn cost watchdog in background. Polls RunPod API
    # every 60s, warns at 95% of BUDGET_CAP_USD, calls stop_pod + writes
    # .state/round2/COST_CAP_HIT on 100% so phase-loop callers can graceful_abort.
    if [ -n "${RUNPOD_POD_ID:-}" ] && [ -x "${SCRIPTS_DIR}/runpod_cost_watchdog.py" ] || [ -f "${SCRIPTS_DIR}/runpod_cost_watchdog.py" ]; then
        python3 "${SCRIPTS_DIR}/runpod_cost_watchdog.py" \
            --pod-id "${RUNPOD_POD_ID}" \
            --cap "${BUDGET_CAP_USD:-60}" \
            --state-dir ".state/round2" \
            >> "${WORKSPACE}/logs/cost_watchdog.log" 2>&1 &
        WATCHDOG_PID=$!
        log "cost watchdog pid=${WATCHDOG_PID} cap=\$${BUDGET_CAP_USD:-60}"
        trap 'kill ${WATCHDOG_PID} 2>/dev/null; true' EXIT
    fi

    phase1_cpt_broad || fail_with_logs "phase1_failed" "${PHASE1_LOG}" "$?"
    phase2_cpt_clean || fail_with_logs "phase2_failed" "${PHASE2_LOG}" "$?"
    phase2_5_merge_cpt || fail_with_logs "phase2_merge_failed" "${MERGE_CPT_LOG}" "$?"
    # Phase 5.9.2: explicit if-then-else wrap for Phase 3 SFT (matches Phase 4
    # ORPO pattern). Captures rc for ntfy notification + log persistence on
    # SFT failure (previously relied on `|| fail_with_logs` short-circuit which
    # is functionally equivalent but less explicit). fail_with_logs calls
    # notify() → ntfy alert + persist_run_artifacts → stop_pod.
    if [ "${SKIP_SFT:-0}" = "1" ] || [ "${SFT_NUM_EPOCHS:-2}" = "0" ]; then
        log "[INFO] SKIP_SFT=${SKIP_SFT:-0} SFT_NUM_EPOCHS=${SFT_NUM_EPOCHS:-unset}; Phase 3/3.5 SFT skipped"
    else
        if phase3_sft_threaded; then
            :
        else
            local phase3_rc=$?
            log "[FATAL] phase 3 SFT failed rc=${phase3_rc}"
            fail_with_logs "phase3_failed" "${PHASE3_LOG}" "${phase3_rc:-2}"
        fi
        phase3_5_merge_sft || log "[WARN] phase 3.5 merge non-fatal failure"
    fi
    if phase4_orpo; then
        :
    else
        local phase4_rc=$?
        # When ORPO is enabled, Phase 4 failure is FATAL — the paper-grade
        # 5-phase pipeline cannot silently skip preference alignment.
        # (3-model audit BLOCKER)
        if [ "${ORPO_NUM_EPOCHS:-0}" -gt 0 ]; then
            log "[FATAL] phase 4 ORPO failed rc=${phase4_rc} and ORPO_NUM_EPOCHS=${ORPO_NUM_EPOCHS}>0"
            fail_with_logs "phase4_failed" "${PHASE4_LOG}" "${phase4_rc:-2}"
        fi
        log "[INFO] ORPO disabled (ORPO_NUM_EPOCHS=${ORPO_NUM_EPOCHS:-0}); phase 4 skipped rc=${phase4_rc}"
    fi

    if phase5_eval_gate; then
        log "[eval] gate completed"
    else
        local eval_rc=$?
        log "[FATAL] phase 5 gate failed rc=${eval_rc}; preserving artifacts for mutation/next cycle"
        notify "dalbit round2 eval gate failed rc=${eval_rc}; artifacts will be persisted"
        fail_with_logs "eval_gate_failed" "${PHASE5_LOG}" "${eval_rc}"
    fi

    local final_status="done_ok"
    write_done "${final_status}"
    upload_hf_artifacts || final_status="${final_status}_hf_upload_warn"
    write_done "${final_status}"
    notify "dalbit round2 ${final_status} repo=${HF_REPO_ROUND2} | wandb: project=${WANDB_PROJECT:-dalbitalba-round2} group=${WANDB_RUN_GROUP:-} url=https://wandb.ai/${WANDB_USERNAME:-anonymous}/${WANDB_PROJECT:-dalbitalba-round2}"
    persist_run_artifacts "${final_status}"
    stop_pod "${final_status}"
    log "chain_train_round2 cycle-1 END status=${final_status}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    install_traps
    run_main
fi
