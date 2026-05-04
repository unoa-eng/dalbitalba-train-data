#!/usr/bin/env bash
# chain_train_round2.sh — Round-2 cycle-1 orchestrator (5-phase)
#
# Phase 1: CPT (broad)        cpt_enriched.jsonl       1 epoch  LORA_R=64
# Phase 2: CPT (clean) + DoRA cpt_corpus.v3.jsonl      1 epoch  LORA_R=128 DoRA=1
# Phase 3: TC-SFT + Persona   sft_thread_conditioned   2 epoch  loss-weight argot
# Phase 4: ORPO               orpo_pairs.jsonl         1 epoch  beta=0.1
# Phase 5: Eval gate          phase6_eval_v2 (base+v2)
#
# Wraps the existing CPT/SFT pipelines from chain_train.sh by repeatedly
# invoking train_cpt.py / train_sft.py with phase-specific env vars.
#
# Usage on RunPod (after chain_train.sh checkout phase):
#   source recipes/round2-cycle1.env
#   bash chain_train_round2.sh
#
# Required env (carried from chain_train.sh):
#   WORKSPACE, DATA_DIR, OUT_DIR, LOG_DIR, HF_TOKEN, HF_USERNAME,
#   GITHUB_TOKEN, GITHUB_REPO, RUNPOD_POD_ID
#
# Required env (from recipes/round2-cycle1.env):
#   BASE_MODEL, CPT_PHASE_1_DATA, CPT_PHASE_2_DATA, CPT_NUM_EPOCHS,
#   LORA_R, CPT_USE_DORA, SEQ_LEN, OVERSAMPLE_LG_XXL,
#   SFT_DATA, SFT_NUM_EPOCHS, SFT_PERSONA_LIST,
#   SFT_LOSS_WEIGHT_ARGOT, SFT_LOSS_WEIGHT_THRESHOLD,
#   ORPO_DATA, ORPO_NUM_EPOCHS, ORPO_BETA,
#   GENERATION_TEMP, GENERATION_TOP_P, GENERATION_MAX_NEW_TOKENS,
#   EVAL_GATE, EVAL_PERSONA_LIST,
#   MUTATOR_RULESET, BUDGET_PROFILE, BUDGET_CAP_USD

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
DATA_DIR="${DATA_DIR:-${WORKSPACE}/data}"
OUT_DIR="${OUT_DIR:-${WORKSPACE}/out}"
LOG_DIR="${LOG_DIR:-${WORKSPACE}/logs}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${WORKSPACE}/scripts}"
ROUND2_LOG="${LOG_DIR}/chain_round2.log"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

log() {
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "${ts} $1" | tee -a "${ROUND2_LOG}"
}

require() {
    if [ -z "${!1:-}" ]; then
        log "[FATAL] required env var $1 is unset"
        exit 2
    fi
}

resolve_path() {
    local raw="$1"
    local rel="${raw#./}"
    if [ -z "${raw}" ]; then
        return 1
    fi
    for candidate in \
        "${raw}" \
        "${DATA_DIR}/${rel}" \
        "${WORKSPACE}/${rel}" \
        "${WORKSPACE}/repo/${rel}"
    do
        if [ -e "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    printf '%s\n' "${raw}"
    return 0
}

adapter_dir_ready() {
    local path="$1"
    [ -d "${path}" ] && [ -f "${path}/adapter_config.json" ]
}

merge_cpt_phase() {
    local base_model="$1"
    local lora_dir="$2"
    local merged_dir="$3"
    local label="$4"
    if ! adapter_dir_ready "${lora_dir}"; then
        log "[FATAL] ${label} adapter missing: ${lora_dir}"
        return 2
    fi
    BASE_MODEL="${base_model}" \
    CPT_LORA_DIR="${lora_dir}" \
    CPT_MERGED_DIR="${merged_dir}" \
    python3 "${SCRIPTS_DIR}/merge_cpt_to_fp16.py" \
        2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "${label} merge exit=${rc}"
    return $rc
}

generate_phase5_samples() {
    local eval_input="$1"
    local ai_output="$2"
    local base_model="$3"
    local sft_adapter_repo="$4"
    local cpt_merged_path="$5"
    local cpt_adapter_repo="$6"

    EVAL_INPUT_PATH="${eval_input}" \
    EVAL_OUTPUT_PATH="${ai_output}" \
    BASE_MODEL="${base_model}" \
    SFT_ADAPTER_REPO="${sft_adapter_repo}" \
    CPT_MERGED_PATH="${cpt_merged_path}" \
    CPT_ADAPTER_REPO="${cpt_adapter_repo}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    python3 "${SCRIPTS_DIR}/phase6_generate.py" \
        2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase5 generate exit=${rc}"
    return $rc
}

# ---------------- Phase 1: CPT broad ----------------

phase1_cpt_broad() {
    require BASE_MODEL
    require CPT_PHASE_1_DATA
    log "=== Phase 1: CPT broad ==="
    local out_dir="${OUT_DIR}/round2-phase1-cpt-lora"
    local merged_dir="${OUT_DIR}/round2-phase1-cpt-merged-fp16"
    mkdir -p "${out_dir}"
    local input="${DATA_DIR}/${CPT_PHASE_1_DATA}"
    if [ ! -f "${input}" ]; then
        log "[FATAL] phase-1 input missing: ${input}"
        return 2
    fi
    local phase_base="${BASE_MODEL}"
    INPUT_JSONL="${input}" \
    BASE_MODEL="${phase_base}" \
    CPT_NUM_EPOCHS="${CPT_NUM_EPOCHS:-1}" \
    CPT_LORA_R="${LORA_R:-64}" \
    CPT_LORA_ALPHA="${LORA_R:-64}" \
    CPT_USE_DORA="0" \
    CPT_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
    CPT_OUTPUT_DIR="${out_dir}" \
    python3 "${SCRIPTS_DIR}/../train_cpt.py" \
        2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase1 exit=${rc}"
    if [ $rc -ne 0 ]; then
        return $rc
    fi
    merge_cpt_phase "${phase_base}" "${out_dir}" "${merged_dir}" "phase1" || return $?
    export ROUND2_PHASE1_MERGED="${merged_dir}"
    return 0
}

# ---------------- Phase 2: CPT clean + DoRA ----------------

phase2_cpt_clean() {
    require CPT_PHASE_2_DATA
    log "=== Phase 2: CPT clean + DoRA ==="
    local out_dir="${OUT_DIR}/round2-phase2-cpt-lora"
    local merged_dir="${OUT_DIR}/round2-phase2-cpt-merged-fp16"
    mkdir -p "${out_dir}"
    local input="${DATA_DIR}/${CPT_PHASE_2_DATA}"
    if [ ! -f "${input}" ]; then
        log "[FATAL] phase-2 input missing: ${input}"
        return 2
    fi
    local phase_base="${ROUND2_PHASE1_MERGED:-${BASE_MODEL}}"
    INPUT_JSONL="${input}" \
    BASE_MODEL="${phase_base}" \
    CPT_NUM_EPOCHS="${CPT_NUM_EPOCHS:-1}" \
    CPT_LORA_R="128" \
    CPT_LORA_ALPHA="128" \
    CPT_USE_DORA="1" \
    CPT_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
    CPT_OUTPUT_DIR="${out_dir}" \
    python3 "${SCRIPTS_DIR}/../train_cpt.py" \
        2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase2 exit=${rc}"
    if [ $rc -ne 0 ]; then
        return $rc
    fi
    merge_cpt_phase "${phase_base}" "${out_dir}" "${merged_dir}" "phase2" || return $?
    export ROUND2_PHASE2_MERGED="${merged_dir}"
    return 0
}

# ---------------- Phase 3: TC-SFT + Persona + Loss-Weight ----------------

phase3_sft_threaded() {
    require SFT_DATA
    log "=== Phase 3: TC-SFT + Persona + Loss-Weight ==="
    local sft_data="${DATA_DIR}/${SFT_DATA}"
    local active_sft_data="${sft_data}"
    if [ ! -f "${sft_data}" ]; then
        log "[INFO] ${SFT_DATA} missing — building from cpt_context_stream"
        python3 "${SCRIPTS_DIR}/round2_build_tc_sft.py" \
            --context-stream "${DATA_DIR}/cpt_context_stream.jsonl" \
            --raw-source-dir "${DATA_DIR}/source_db_cache" \
            --persona-list "${SFT_PERSONA_LIST:-runs/round2-obsidian-synthesis/persona-30-extracted.json}" \
            --argot-threshold "${SFT_LOSS_WEIGHT_THRESHOLD:-2}" \
            --argot-weight "${SFT_LOSS_WEIGHT_ARGOT:-1.5}" \
            --out "${sft_data}" 2>&1 | tee -a "${ROUND2_LOG}"
        if [ ! -f "${sft_data}" ]; then
            log "[FATAL] tc-sft build failed"
            return 2
        fi
    fi
    if [ "${SFT_OBSIDIAN_ENABLE:-0}" = "1" ] || [ "${SFT_OBSIDIAN_ENABLE:-false}" = "true" ]; then
        local vault_root
        local ablation_dir
        local style_map
        local variant_jsonl
        local matched_jsonl
        local unseen_jsonl
        local summary_json
        local style_map_script
        local variant_script
        vault_root="$(resolve_path "${SFT_OBSIDIAN_VAULT_ROOT:-research/obsidian-export}")"
        ablation_dir="${OUT_DIR}/round2-obsidian-ablation"
        style_map="${ablation_dir}/style_map.json"
        variant_jsonl="${ablation_dir}/sft_thread_conditioned_variant.jsonl"
        matched_jsonl="${ablation_dir}/matched.jsonl"
        unseen_jsonl="${ablation_dir}/unseen.jsonl"
        summary_json="${ablation_dir}/summary.json"
        style_map_script="$(resolve_path "scripts/build_obsidian_style_map.py")"
        variant_script="$(resolve_path "scripts/build_obsidian_sft_variant.py")"
        mkdir -p "${ablation_dir}"
        if [ ! -d "${vault_root}" ]; then
            log "[FATAL] Obsidian vault root not found: ${vault_root}"
            return 2
        fi
        python3 "${style_map_script}" \
            --vault-root "${vault_root}" \
            --out "${style_map}" 2>&1 | tee -a "${ROUND2_LOG}"
        local style_rc=${PIPESTATUS[0]}
        if [ ${style_rc} -ne 0 ]; then
            log "[FATAL] Obsidian style-map build failed rc=${style_rc}"
            return ${style_rc}
        fi
        python3 "${variant_script}" \
            --sft-jsonl "${sft_data}" \
            --obsidian-map "${style_map}" \
            --variant-out "${variant_jsonl}" \
            --matched-out "${matched_jsonl}" \
            --unseen-out "${unseen_jsonl}" \
            --summary-out "${summary_json}" \
            --target-ratio "${SFT_OBSIDIAN_TARGET_RATIO:-0.08}" 2>&1 | tee -a "${ROUND2_LOG}"
        local variant_rc=${PIPESTATUS[0]}
        if [ ${variant_rc} -ne 0 ]; then
            log "[FATAL] Obsidian round2 variant build failed rc=${variant_rc}"
            return ${variant_rc}
        fi
        active_sft_data="${variant_jsonl}"
        log "phase3 obsidian variant OK: ${active_sft_data}"
    fi
    local out_dir="${OUT_DIR}/round2-phase3-sft-lora"
    mkdir -p "${out_dir}"
    local phase_base="${ROUND2_PHASE2_MERGED:-${BASE_MODEL}}"
    SFT_PAIR_JSONL="${active_sft_data}" \
    BASE_MODEL="${phase_base}" \
    SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-2}" \
    SFT_LORA_R="${LORA_R:-128}" \
    SFT_LORA_ALPHA="${LORA_R:-128}" \
    SFT_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
    SFT_LOSS_WEIGHT_ARGOT="${SFT_LOSS_WEIGHT_ARGOT:-1.5}" \
    SFT_LOSS_WEIGHT_THRESHOLD="${SFT_LOSS_WEIGHT_THRESHOLD:-2}" \
    SFT_OUTPUT_DIR="${out_dir}" \
    python3 "${SCRIPTS_DIR}/../train_sft.py" \
        2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase3 exit=${rc}"
    return $rc
}

# ---------------- Phase 3.5: SFT LoRA merge (gate ORPO uses cumulative weights) ----------------

phase3_5_merge_sft() {
    log "=== Phase 3.5: SFT LoRA merge ==="
    local sft_lora="${OUT_DIR}/round2-phase3-sft-lora"
    local sft_merged="${OUT_DIR}/round2-phase3-sft-merged-fp16"
    if [ ! -d "${sft_lora}" ]; then
        log "[INFO] phase-3 SFT lora dir missing — skip merge; ORPO will use BASE_MODEL"
        return 0
    fi
    local phase_base="${ROUND2_PHASE2_MERGED:-${BASE_MODEL}}"
    BASE_MODEL="${phase_base}" \
    SFT_LORA_DIR="${sft_lora}" \
    SFT_MERGED_DIR="${sft_merged}" \
    python3 "${SCRIPTS_DIR}/merge_sft_to_fp16.py" \
        2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ] && [ -d "${sft_merged}" ]; then
        export ORPO_BASE_MODEL="${sft_merged}"
        export ROUND2_PHASE3_SFT_MERGED="${sft_merged}"
        log "phase3.5 merge OK: ORPO will use ${sft_merged}"
    else
        log "[WARN] phase3.5 merge failed (rc=${rc}); ORPO falls back to BASE_MODEL"
    fi
    return 0
}

# ---------------- Phase 4: ORPO ----------------

phase4_orpo() {
    require ORPO_DATA
    log "=== Phase 4: ORPO ==="
    local orpo_data="${DATA_DIR}/${ORPO_DATA}"
    if [ ! -f "${orpo_data}" ]; then
        log "[INFO] ${ORPO_DATA} missing — building from runs/refinement-*"
        python3 "${SCRIPTS_DIR}/round2_build_orpo_pairs.py" \
            --runs-glob "${WORKSPACE}/repo/runs/refinement-2026042*" \
            --val-set "${DATA_DIR}/val_set.v3.jsonl" \
            --out "${orpo_data}" 2>&1 | tee -a "${ROUND2_LOG}"
        if [ ! -f "${orpo_data}" ]; then
            log "[WARN] orpo build skipped (no refinement runs); proceed without ORPO"
            return 0
        fi
    fi
    if [ ! -x "${SCRIPTS_DIR}/../train_orpo.py" ] && [ ! -f "${SCRIPTS_DIR}/../train_orpo.py" ]; then
        log "[INFO] train_orpo.py missing — Phase-4 deferred to cycle-2 (TBD)."
        return 0
    fi
    local out_dir="${OUT_DIR}/round2-phase4-orpo"
    mkdir -p "${out_dir}"
    # Use SFT-merged checkpoint if Phase 3.5 produced one; else fall back to base
    local orpo_base="${ORPO_BASE_MODEL:-${BASE_MODEL}}"
    ORPO_DATA="${orpo_data}" \
    ORPO_NUM_EPOCHS="${ORPO_NUM_EPOCHS:-1}" \
    ORPO_BETA="${ORPO_BETA:-0.1}" \
    ORPO_OUTPUT_DIR="${out_dir}" \
    ORPO_MAX_SEQ_LEN="${SEQ_LEN:-2048}" \
    BASE_MODEL="${orpo_base}" \
    python3 "${SCRIPTS_DIR}/../train_orpo.py" \
        2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase4 exit=${rc} (base=${orpo_base})"
    return $rc
}

# ---------------- Phase 5: Eval gate ----------------

phase5_eval_gate() {
    log "=== Phase 5: phase6_eval_v2 ==="
    local eval_input="${DATA_DIR}/val_set.v3.jsonl"
    local ai="${OUT_DIR}/phase5-ai_generated.jsonl"
    rm -f "${ai}"
    if [ ! -f "${eval_input}" ]; then
        log "[FATAL] eval input missing: ${eval_input}"
        return 2
    fi
    if adapter_dir_ready "${OUT_DIR}/round2-phase4-orpo"; then
        local eval_base="${ROUND2_PHASE3_SFT_MERGED:-${BASE_MODEL}}"
        generate_phase5_samples \
            "${eval_input}" \
            "${ai}" \
            "${eval_base}" \
            "${OUT_DIR}/round2-phase4-orpo" \
            "" \
            "" || return $?
    elif adapter_dir_ready "${OUT_DIR}/round2-phase3-sft-lora"; then
        if [ -z "${ROUND2_PHASE2_MERGED:-}" ]; then
            log "[FATAL] phase-2 merged checkpoint missing for SFT eval"
            return 2
        fi
        generate_phase5_samples \
            "${eval_input}" \
            "${ai}" \
            "${BASE_MODEL}" \
            "${OUT_DIR}/round2-phase3-sft-lora" \
            "${ROUND2_PHASE2_MERGED}" \
            "" || return $?
    elif [ -n "${ROUND2_PHASE2_MERGED:-}" ]; then
        generate_phase5_samples \
            "${eval_input}" \
            "${ai}" \
            "${BASE_MODEL}" \
            "" \
            "${ROUND2_PHASE2_MERGED}" \
            "" || return $?
    else
        log "[WARN] no trained model artifact from phases 2-4 — skipping eval gate"
        return 0
    fi
    python3 "${SCRIPTS_DIR}/phase6_eval_v2.py" \
        --ai "${ai}" \
        --raw "${eval_input}" \
        --persona-list "${EVAL_PERSONA_LIST:-runs/round2-obsidian-synthesis/persona-30-extracted.json}" \
        --out "${OUT_DIR}/phase5-eval-v2.json" \
        --skip-mauve 2>&1 | tee -a "${ROUND2_LOG}"
    local rc=${PIPESTATUS[0]}
    log "phase5 exit=${rc}"
    if [ -f "${OUT_DIR}/phase5-eval-v2.json" ]; then
        python3 "${SCRIPTS_DIR}/round2_mutator.py" \
            --metrics "${OUT_DIR}/phase5-eval-v2.json" \
            >> "${ROUND2_LOG}" 2>&1 || true
    fi
    return $rc
}

# ---------------- Main ----------------

log "=========================================="
log "chain_train_round2 cycle-1 START"
log "BASE_MODEL=${BASE_MODEL:-(unset)}"
log "BUDGET_PROFILE=${BUDGET_PROFILE:-budget30} CAP=\$${BUDGET_CAP_USD:-25}"
log "=========================================="

if [ ! -f "${WORKSPACE}/recipes/round2-cycle1.env" ] && [ ! -f "recipes/round2-cycle1.env" ]; then
    log "[FATAL] recipes/round2-cycle1.env not found — copy from repo or pass via env"
    exit 2
fi
source "${WORKSPACE}/recipes/round2-cycle1.env" 2>/dev/null || \
    source "recipes/round2-cycle1.env"

phase1_cpt_broad || { log "[FATAL] phase 1 failed"; exit $?; }
phase2_cpt_clean || { log "[FATAL] phase 2 failed"; exit $?; }
phase3_sft_threaded || { log "[FATAL] phase 3 failed"; exit $?; }
phase3_5_merge_sft || log "[WARN] phase 3.5 merge non-fatal failure"
phase4_orpo || log "[WARN] phase 4 skipped or non-fatal failure"
phase5_eval_gate || log "[INFO] phase 5 gate FAIL — mutator output above for next cycle"

log "chain_train_round2 cycle-1 END"
