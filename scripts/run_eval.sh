#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="/workspace"
LOG_DIR="${WORKSPACE}/logs"
LOG_FILE="${LOG_DIR}/eval.log"
REPO_DIR="${WORKSPACE}/repo"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"

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

  cat > "${run_dir}/manifest.json" <<EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "status": "${status}",
  "github_repo": "${GITHUB_REPO}",
  "hf_adapter_repo": "${HF_ADAPTER_REPO}",
  "base_model": "${BASE_MODEL:-upstage/SOLAR-10.7B-v1.0}",
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
require_env "HF_ADAPTER_REPO"
require_env "ANTHROPIC_API_KEY"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
  git clone "${repo_url}" "${REPO_DIR}" >> "${LOG_FILE}" 2>&1
fi

cd "${REPO_DIR}"
git config user.email "${GITHUB_EMAIL:-eval@dalbitalba.local}"
git config user.name "${GITHUB_NAME:-dalbitalba-eval}"

log "[1/5] install python dependencies"
pip install -q --no-cache-dir --upgrade \
  "transformers==4.44.2" \
  "peft==0.12.0" \
  "bitsandbytes==0.43.3" \
  "accelerate==0.33.0" \
  "datasets==2.21.0" \
  "huggingface_hub>=0.24.0" \
  "safetensors>=0.4.3" \
  "sentencepiece" \
  "tokenizers>=0.19" \
  "protobuf" \
  anthropic \
  openai \
  jinja2 \
  >> "${LOG_FILE}" 2>&1

log "[1/5] verify python imports"
python - <<'EOF' >> "${LOG_FILE}" 2>&1
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

log "[2/5] generate ai samples"
HF_ADAPTER_REPO="${HF_ADAPTER_REPO}" HF_TOKEN="${HF_TOKEN:-}" BASE_MODEL="${BASE_MODEL:-upstage/SOLAR-10.7B-v1.0}" \
  python scripts/generate_samples.py >> "${LOG_FILE}" 2>&1

log "[3/5] build blind eval set"
python eval/make_eval_samples.py \
  --ai-output ai_generated.jsonl \
  --crawl cpt_corpus.jsonl \
  --n "${EVAL_PER_CLASS:-30}" \
  --output eval_samples.jsonl \
  >> "${LOG_FILE}" 2>&1

log "[4/5] run judges"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  python eval/judge_3way.py \
  --samples eval_samples.jsonl \
  --output-dir eval/results \
  >> "${LOG_FILE}" 2>&1

log "[5/5] render reports"
python eval/native_eval_kit.py \
  --samples eval_samples.jsonl \
  --results-dir eval/results \
  --format html \
  --output eval/native_kit.html \
  >> "${LOG_FILE}" 2>&1

python eval/native_eval_kit.py \
  --samples eval_samples.jsonl \
  --results-dir eval/results \
  --format md \
  --output eval/native_kit.md \
  >> "${LOG_FILE}" 2>&1

persist_eval_artifacts "done_ok"
notify "dalbitalba eval done"
stop_pod "done_ok"
log "=== dalbitalba eval chain complete ==="
