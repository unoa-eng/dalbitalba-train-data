#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="/workspace"
LOG_DIR="${WORKSPACE}/logs"
LOG_FILE="${LOG_DIR}/eval.log"
REPO_DIR="${WORKSPACE}/repo"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"

# Stage timeouts (hours). Defaults sized for budget30 eval ~$1.5 ceiling.
EVAL_INSTALL_TIMEOUT_HOURS="${EVAL_INSTALL_TIMEOUT_HOURS:-1}"
EVAL_GENERATE_TIMEOUT_HOURS="${EVAL_GENERATE_TIMEOUT_HOURS:-2}"
EVAL_METRIC_TIMEOUT_HOURS="${EVAL_METRIC_TIMEOUT_HOURS:-1}"

# Pin python interpreter — never trust PATH (chain_train.sh established this rule).
PY="${PY:-python3}"
PIP="${PIP:-${PY} -m pip}"

EVAL_FAILED=0
ARTIFACTS_PERSISTED=0

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "${LOG_FILE}"
}

notify() {
  local msg="$1"
  if [[ -n "${NTFY_TOPIC:-}" ]]; then
    curl -fsS -m 10 -H "Content-Type: text/plain" -d "${msg}" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
  fi
}

require_env() {
  local key="$1"
  if [[ -z "${!key:-}" ]]; then
    log "[ERROR] missing env: ${key}"
    exit 1
  fi
}

# run_timeout HOURS LABEL CMD...
run_timeout() {
  local hours="$1"
  local label="$2"
  shift 2
  if command -v timeout >/dev/null 2>&1; then
    log "[stage:start] ${label} timeout=${hours}h"
    timeout --foreground "${hours}h" "$@"
  else
    log "[WARN] timeout binary missing — running ${label} unbounded"
    "$@"
  fi
}

stop_pod() {
  local reason="$1"
  log "[stop] reason=${reason}"
  if [[ "${RUNPOD_POD_ID:-}" == "__SELF__" || -z "${RUNPOD_POD_ID:-}" ]]; then
    RUNPOD_POD_ID="$(curl -fsS -m 5 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || curl -fsS -m 5 http://metadata.runpod.io/pod-id 2>/dev/null || true)"
  fi
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    runpodctl stop pod "${RUNPOD_POD_ID}" >/dev/null 2>&1 || \
      curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" >/dev/null 2>&1 || true
  fi
}

graceful_abort() {
  local sig="$1"
  log "[abort] signal=${sig} — persisting partial artifacts then stopping pod"
  EVAL_FAILED=1
  if [[ "${ARTIFACTS_PERSISTED}" -eq 0 ]] && [[ -d "${REPO_DIR}/.git" ]]; then
    persist_eval_artifacts "aborted_${sig}" || log "[abort] persist failed"
    ARTIFACTS_PERSISTED=1
  fi
  notify "dalbitalba eval aborted (${sig})"
  stop_pod "aborted_${sig}"
  exit 130
}

on_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 ]] && [[ "${ARTIFACTS_PERSISTED}" -eq 0 ]] && [[ -d "${REPO_DIR}/.git" ]]; then
    log "[exit] non-zero rc=${rc} — persisting failure artifacts"
    persist_eval_artifacts "failed_rc_${rc}" || log "[exit] persist failed"
    ARTIFACTS_PERSISTED=1
    notify "dalbitalba eval failed rc=${rc}"
    stop_pod "failed_rc_${rc}"
  fi
  exit ${rc}
}

trap 'graceful_abort TERM' TERM
trap 'graceful_abort INT' INT
trap 'graceful_abort HUP' HUP
trap 'graceful_abort QUIT' QUIT
trap on_exit EXIT

persist_eval_artifacts() {
  local status="$1"
  local stamp branch run_dir repo_url source_ref latest_tmp

  stamp="$(date -u '+%Y%m%d-%H%M%S')"
  branch="eval-run-${stamp}"
  run_dir="${REPO_DIR}/runs/${branch}"
  latest_tmp="$(mktemp)"
  mkdir -p "${run_dir}"

  cp "${LOG_FILE}" "${run_dir}/eval.log"
  [[ -f "${REPO_DIR}/ai_generated.jsonl" ]] && cp "${REPO_DIR}/ai_generated.jsonl" "${run_dir}/ai_generated.jsonl"
  [[ -f "${REPO_DIR}/eval_samples.jsonl" ]] && cp "${REPO_DIR}/eval_samples.jsonl" "${run_dir}/eval_samples.jsonl"
  [[ -f "${REPO_DIR}/eval_key.json" ]] && cp "${REPO_DIR}/eval_key.json" "${run_dir}/eval_key.json"
  [[ -d "${REPO_DIR}/eval/results" ]] && cp -R "${REPO_DIR}/eval/results" "${run_dir}/results"
  [[ -f "${REPO_DIR}/eval/native_kit.html" ]] && cp "${REPO_DIR}/eval/native_kit.html" "${run_dir}/native_kit.html"
  [[ -f "${REPO_DIR}/eval/native_kit.md" ]] && cp "${REPO_DIR}/eval/native_kit.md" "${run_dir}/native_kit.md"
  [[ -f "${REPO_DIR}/eval/metrics.json" ]] && cp "${REPO_DIR}/eval/metrics.json" "${run_dir}/metrics.json"

  cat > "${run_dir}/manifest.json" <<EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "status": "${status}",
  "github_repo": "${GITHUB_REPO}",
  "hf_adapter_repo": "${HF_ADAPTER_REPO}",
  "sft_adapter_repo": "${SFT_ADAPTER_REPO:-}",
  "eval_mode": "${EVAL_MODE:-phase6}",
  "base_model": "${BASE_MODEL:-Qwen/Qwen3-8B-Base}",
  "pod_id": "${RUNPOD_POD_ID:-unknown}"
}
EOF

  source_ref="$(
    cd "${REPO_DIR}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || true
  )"

  cat > "${latest_tmp}" <<EOF
{
  "branch": "${branch}",
  "status": "${status}",
  "hf_adapter_repo": "${HF_ADAPTER_REPO}",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF
  cp "${latest_tmp}" "${REPO_DIR}/runs/latest-eval.json"

  repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
  if (
    cd "${REPO_DIR}"
    git checkout -b "${branch}" >/dev/null 2>&1
    git add "runs/${branch}" "runs/latest-eval.json"
    git commit -m "eval: ${branch}" >/dev/null 2>&1 || exit 0
    git push "${repo_url}" "${branch}" >/dev/null 2>&1
  ); then
    log "[push] remote branch created: ${branch}"
    notify "dalbitalba eval artifacts pushed: ${branch}"
  else
    log "[push] failed to push ${branch}"
  fi

  if [[ -n "${source_ref}" && "${source_ref}" != "HEAD" ]]; then
    if (
      cd "${REPO_DIR}"
      git checkout "${source_ref}" >/dev/null 2>&1
      mkdir -p runs
      cp "${latest_tmp}" "runs/latest-eval.json"
      git add "runs/latest-eval.json"
      git commit -m "eval: update latest pointer" >/dev/null 2>&1 || exit 0
      git push "${repo_url}" "${source_ref}" >/dev/null 2>&1
    ); then
      log "[push] latest pointer updated on ${source_ref}"
    else
      log "[push] failed to update latest pointer on ${source_ref}"
    fi
  else
    log "[push] source ref unavailable; latest pointer update skipped"
  fi

  rm -f "${latest_tmp}"
}

log "=== dalbitalba eval chain start ==="

require_env "GITHUB_TOKEN"
require_env "GITHUB_REPO"
if [[ "${EVAL_MODE:-phase6}" = "legacy" ]]; then
  require_env "HF_ADAPTER_REPO"
  require_env "ANTHROPIC_API_KEY"
else
  if [[ -z "${SFT_ADAPTER_REPO:-}" && -n "${HF_ADAPTER_REPO:-}" ]]; then
    export SFT_ADAPTER_REPO="${HF_ADAPTER_REPO}"
  fi
  if [[ -z "${SFT_ADAPTER_REPO:-}" && -z "${CPT_MERGED_REPO:-}" && -z "${CPT_MERGED_PATH:-}" ]]; then
    log "[FATAL] phase6 requires SFT_ADAPTER_REPO or CPT_MERGED_REPO/CPT_MERGED_PATH"
    exit 2
  fi
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
  git clone "${repo_url}" "${REPO_DIR}" >> "${LOG_FILE}" 2>&1
fi

cd "${REPO_DIR}"
git config user.email "${GITHUB_EMAIL:-eval@dalbitalba.local}"
git config user.name "${GITHUB_NAME:-dalbitalba-eval}"

log "[1/5] install python dependencies"
run_timeout "${EVAL_INSTALL_TIMEOUT_HOURS}" "pip install" \
  ${PIP} install -q --no-cache-dir --upgrade \
  "transformers==4.51.3" \
  "peft==0.13.2" \
  "bitsandbytes==0.49.2" \
  "accelerate==0.34.2" \
  "datasets==2.21.0" \
  "huggingface_hub>=0.30.0,<1.0" \
  "safetensors>=0.4.3" \
  "sentencepiece" \
  "tokenizers>=0.21,<0.22" \
  "tiktoken>=0.7.0" \
  "protobuf" \
  anthropic \
  openai \
  jinja2 \
  >> "${LOG_FILE}" 2>&1

log "[1/5] verify python imports"
${PY} - <<'EOF' >> "${LOG_FILE}" 2>&1
import torch
import transformers
import peft
import bitsandbytes
import accelerate

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("accelerate", accelerate.__version__)
EOF

if [[ "${EVAL_MODE:-phase6}" = "legacy" ]]; then
  log "[2/5] generate ai samples (legacy blind judge mode)"
  HF_ADAPTER_REPO="${HF_ADAPTER_REPO}" HF_TOKEN="${HF_TOKEN:-}" BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}" \
    run_timeout "${EVAL_GENERATE_TIMEOUT_HOURS}" "legacy:generate_samples" \
    ${PY} scripts/generate_samples.py >> "${LOG_FILE}" 2>&1

  log "[3/5] build blind eval set"
  run_timeout "${EVAL_METRIC_TIMEOUT_HOURS}" "legacy:make_eval_samples" \
    ${PY} eval/make_eval_samples.py \
    --ai-output ai_generated.jsonl \
    --crawl cpt_corpus.jsonl \
    --n "${EVAL_PER_CLASS:-30}" \
    --output eval_samples.jsonl \
    >> "${LOG_FILE}" 2>&1

  log "[4/5] run judges"
  ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    run_timeout "${EVAL_METRIC_TIMEOUT_HOURS}" "legacy:judge_3way" \
    ${PY} eval/judge_3way.py \
    --samples eval_samples.jsonl \
    --output-dir eval/results \
    >> "${LOG_FILE}" 2>&1

  log "[5/5] render reports"
  ${PY} eval/native_eval_kit.py \
    --samples eval_samples.jsonl \
    --results-dir eval/results \
    --format html \
    --output eval/native_kit.html \
    >> "${LOG_FILE}" 2>&1

  ${PY} eval/native_eval_kit.py \
    --samples eval_samples.jsonl \
    --results-dir eval/results \
    --format md \
    --output eval/native_kit.md \
    >> "${LOG_FILE}" 2>&1
else
  log "[2/5] generate ai samples (phase6 deterministic gate)"
  phase6_input="${EVAL_INPUT_DATA:-sft_thread_conditioned.eval.jsonl}"
  if [[ "${phase6_input}" = /* ]]; then
    phase6_input_path="${phase6_input}"
  else
    phase6_input_path="${REPO_DIR}/${phase6_input}"
  fi
  if [[ ! -f "${phase6_input_path}" ]]; then
    log "[ERROR] phase6 input missing: ${phase6_input_path}"
    exit 1
  fi

  phase6_persona_path=""
  if [[ -n "${EVAL_PERSONA_LIST:-}" ]]; then
    if [[ "${EVAL_PERSONA_LIST}" = /* ]]; then
      phase6_persona_path="${EVAL_PERSONA_LIST}"
    else
      phase6_persona_path="${REPO_DIR}/${EVAL_PERSONA_LIST}"
    fi
  fi

  phase6_secondary_path=""
  if [[ -n "${EVAL_SECONDARY_AI:-}" ]]; then
    if [[ "${EVAL_SECONDARY_AI}" = /* ]]; then
      phase6_secondary_path="${EVAL_SECONDARY_AI}"
    else
      phase6_secondary_path="${REPO_DIR}/${EVAL_SECONDARY_AI}"
    fi
  fi

  mkdir -p eval
  BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}" \
  SFT_ADAPTER_REPO="${SFT_ADAPTER_REPO}" \
  EVAL_INPUT_JSONL="${phase6_input_path}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  TEMPERATURE="${TEMPERATURE:-${GENERATION_TEMP:-1.1}}" \
  TOP_P="${TOP_P:-${GENERATION_TOP_P:-0.9}}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${GENERATION_MAX_NEW_TOKENS:-200}}" \
    run_timeout "${EVAL_GENERATE_TIMEOUT_HOURS}" "phase6:generate" \
    ${PY} scripts/phase6_generate.py >> "${LOG_FILE}" 2>&1
  cp /workspace/ai_generated.jsonl ai_generated.jsonl

  log "[3/5] run phase6 metric gate"
  phase6_args=(--ai ai_generated.jsonl --raw "${phase6_input_path}" --out eval/metrics.json)
  if [[ -n "${phase6_persona_path}" ]]; then
    phase6_args+=(--persona-list "${phase6_persona_path}")
  fi
  if [[ -n "${phase6_secondary_path}" ]]; then
    phase6_args+=(--secondary-ai "${phase6_secondary_path}")
  fi
  phase6_mauve_args=()
  if [[ "${MAUVE_DISABLED:-0}" = "1" || "${RUN_MAUVE:-1}" = "0" ]]; then
    phase6_mauve_args+=(--skip-mauve)
  fi
  run_timeout "${EVAL_METRIC_TIMEOUT_HOURS}" "phase6:eval-v2" \
    ${PY} scripts/phase6_eval_v2.py \
    "${phase6_args[@]}" \
    "${phase6_mauve_args[@]}" \
    >> "${LOG_FILE}" 2>&1 || PHASE6_RC=$?
  PHASE6_RC="${PHASE6_RC:-0}"
  log "[4/5] phase6 gate exit=${PHASE6_RC}"
  if [ "${PHASE6_RC}" != "0" ]; then
    persist_eval_artifacts "phase6_gate_failed"
    ARTIFACTS_PERSISTED=1
    notify "dalbitalba eval phase6 gate failed rc=${PHASE6_RC}"
    stop_pod "phase6_gate_failed"
    exit "${PHASE6_RC}"
  fi
  log "[5/5] phase6 report ready"
fi

persist_eval_artifacts "done_ok"
ARTIFACTS_PERSISTED=1
notify "dalbitalba eval done"
stop_pod "done_ok"
log "=== dalbitalba eval chain complete ==="
