#!/usr/bin/env bash
# autonomous_loop.sh — 72-hour supervisor for the dalbitalba optimization loop.
#
# Orchestration:
#   wait for active train pod to finish
#   → launch eval pod with current SFT adapter
#   → wait for eval to finish
#   → read metrics + run cycle_report.py
#   → if PASS: create_final_pr.sh + exit
#   → if FAIL: recipe_mutator.py → apply env exports → relaunch train pod
#   → record budget + cycle
#   → repeat until deadline (72h) or stop condition
#
# Runs either on the user's WSL (tmux detached) or a RunPod CPU supervisor pod.

set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
STATE_DIR=".state"
LOG="${STATE_DIR}/autonomous_loop.log"
mkdir -p "${STATE_DIR}"

# ---- env load ----
set -a
# shellcheck disable=SC2046
source <(grep -v '^#' .env.local 2>/dev/null | grep '=' || true)
set +a

: "${NTFY_TOPIC:?NTFY_TOPIC required}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required}"

# ---- constants ----
DEADLINE_H=${LOOP_DEADLINE_HOURS:-72}
START_TS=$(date +%s)
DEADLINE=$((START_TS + DEADLINE_H * 3600))
POLL_POD_INTERVAL=${POLL_POD_INTERVAL:-300}   # 5min
MIN_CYCLE_SLEEP=60

log() {
    local ts; ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "${ts} $*" | tee -a "${LOG}"
}

notify() {
    local msg="$1"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -m 10 -H "Content-Type: text/plain; charset=utf-8" \
            --data-binary "${msg}" "https://ntfy.sh/${NTFY_TOPIC}" \
            >> "${LOG}" 2>&1 || true
    fi
}

record_cost() {
    local usd="$1"
    python3 -c "
import json, pathlib
p = pathlib.Path('${STATE_DIR}/budget_spent.json')
d = json.loads(p.read_text()) if p.exists() else {'usd': 0.0, 'cycles': 0}
d['usd'] = round(d.get('usd', 0.0) + float('${usd}'), 3)
d['cycles'] = d.get('cycles', 0) + 1
p.write_text(json.dumps(d))
"
}

get_budget() {
    python3 -c "
import json, pathlib
p = pathlib.Path('${STATE_DIR}/budget_spent.json')
print(json.loads(p.read_text()).get('usd', 0.0) if p.exists() else 0.0)
"
}

pods_running() {
    python3 - <<'PY'
import json, os, urllib.request
key = os.environ['RUNPOD_API_KEY']
req = urllib.request.Request('https://rest.runpod.io/v1/pods',
    headers={'Authorization': f'Bearer {key}'})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    running = [p for p in data if p.get('desiredStatus') == 'RUNNING']
    print(json.dumps([{'id': p.get('id'), 'name': p.get('name'), 'cost': p.get('costPerHr')} for p in running]))
except Exception as e:
    print('[]')
PY
}

wait_pod_exit() {
    local pod_id="$1"
    while :; do
        status="$(python3 scripts/poll_pod.py "${pod_id}" 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status",""))' 2>/dev/null || echo UNKNOWN)"
        log "wait pod=${pod_id} status=${status}"
        [ "${status}" != "RUNNING" ] && break
        [ -f "${STATE_DIR}/STOP" ] && { log "STOP file detected"; return 2; }
        [ "$(date +%s)" -ge "${DEADLINE}" ] && { log "deadline reached while waiting"; return 3; }
        sleep "${POLL_POD_INTERVAL}"
    done
    return 0
}

fetch_latest_eval_branch() {
    git fetch origin >/dev/null 2>&1 || true
    git branch -r | grep -E 'origin/eval-run-[0-9]' | sort | tail -1 | sed 's|^\s*||; s|origin/||'
}

# ═══════════ MAIN LOOP ═══════════
notify "AUTONOMOUS LOOP started deadline=${DEADLINE_H}h"
log "loop start DEADLINE=${DEADLINE}"

while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    # kill switch
    if [ -f "${STATE_DIR}/STOP" ]; then
        log "STOP file present, exiting"
        notify "loop STOP: file present"
        break
    fi

    # budget cap
    BUDGET=$(get_budget)
    if python3 -c "import sys; sys.exit(0 if float('${BUDGET}') >= 25.0 else 1)"; then
        log "budget cap reached \$${BUDGET}"
        notify "loop STOP: budget \$${BUDGET} >= \$25"
        break
    fi

    # Phase 1: wait for any currently running train pod (first cycle picks up pod7)
    TRAIN_POD=$(python3 -c "
import json, pathlib
p = pathlib.Path('.state/train_pod_state.json')
print(json.loads(p.read_text())['pod_id'] if p.exists() else '')
")
    if [ -n "${TRAIN_POD}" ]; then
        RC=0
        wait_pod_exit "${TRAIN_POD}" || RC=$?
        if [ "${RC}" != 0 ]; then break; fi
        # Estimate train pod cost ≈ 3h @ $0.79
        record_cost "2.5"
        notify "train pod ${TRAIN_POD} EXITED"
    fi

    # Phase 2: launch eval pod
    log "launching eval pod"
    python3 scripts/launch_eval_pod.py >> "${LOG}" 2>&1 || {
        log "eval pod launch failed"
        notify "ESCALATE: eval launch failed"
        break
    }
    EVAL_POD=$(python3 -c "
import json, pathlib
p = pathlib.Path('.state/eval_pod_state.json')
print(json.loads(p.read_text())['pod_id'] if p.exists() else '')
")
    if [ -z "${EVAL_POD}" ]; then
        log "eval pod id missing"
        notify "ESCALATE: no eval pod id"
        break
    fi
    notify "eval pod ${EVAL_POD} launched"

    wait_pod_exit "${EVAL_POD}" || break
    record_cost "0.8"

    # Phase 3: fetch eval artifacts from pushed runs/eval-run-* branch
    EVAL_BRANCH=$(fetch_latest_eval_branch)
    if [ -z "${EVAL_BRANCH}" ]; then
        log "no eval-run branch found after pod exit"
        notify "ESCALATE: eval branch missing"
        break
    fi
    STAMP="${EVAL_BRANCH#eval-run-}"
    EVAL_DIR="runs/eval-run-${STAMP}"
    mkdir -p "${EVAL_DIR}"
    git show "origin/${EVAL_BRANCH}:${EVAL_DIR}/metrics.json" > "${EVAL_DIR}/metrics.json" 2>/dev/null || {
        log "could not pull metrics.json from origin/${EVAL_BRANCH}"
        notify "ESCALATE: eval metrics missing"
        break
    }

    # Phase 4: cycle report + gate decision
    python3 scripts/cycle_report.py --eval-run "eval-run-${STAMP}" >> "${LOG}" 2>&1 || true

    GATE=$(python3 -c "
import json
d = json.load(open('${EVAL_DIR}/metrics.json'))
print(d.get('gate', {}).get('verdict', 'UNKNOWN'))
")
    log "gate=${GATE}"
    notify "cycle eval gate=${GATE}"

    if [ "${GATE}" = "PASS" ]; then
        bash scripts/create_final_pr.sh >> "${LOG}" 2>&1 || {
            log "PR creation failed"
            notify "ESCALATE: PR creation failed"
            break
        }
        notify "LOOP COMPLETE: PR created"
        log "loop complete"
        break
    fi

    # Phase 5: mutate recipe + relaunch
    MUTATION_ENV="$(python3 scripts/recipe_mutator.py --metrics "${EVAL_DIR}/metrics.json" 2>> "${LOG}")"
    RC=$?
    if [ "${RC}" = 2 ]; then
        log "mutator signaled stop"
        notify "loop STOP from mutator"
        break
    fi

    # Source the exports into env for next train launch
    set -a
    # shellcheck disable=SC1090
    eval "${MUTATION_ENV}"
    set +a

    # Sanity: if LOOP_STOP or LOOP_PASS set, exit
    [ "${LOOP_STOP:-0}" = "1" ] && { notify "loop STOP signal"; break; }
    [ "${LOOP_PASS:-0}" = "1" ] && { bash scripts/create_final_pr.sh; notify "LOOP COMPLETE PASS"; break; }

    # Phase 6: launch next train pod with mutated recipe
    log "launching next train pod with mutations"
    python3 scripts/launch_train_pod.py >> "${LOG}" 2>&1 || {
        log "train pod relaunch failed"
        notify "ESCALATE: train relaunch failed"
        break
    }
    NEW_POD=$(python3 -c "
import json, pathlib
p = pathlib.Path('.state/train_pod_state.json')
print(json.loads(p.read_text())['pod_id'] if p.exists() else '')
")
    notify "cycle $(cat .state/loop_state.json | python3 -c 'import json,sys;print(json.load(sys.stdin).get(\"cycle\",\"?\"))') relaunched pod=${NEW_POD}"

    sleep "${MIN_CYCLE_SLEEP}"
done

notify "LOOP TERMINATED $(date -u +%FT%TZ) budget=\$$(get_budget)"
log "loop terminated"
