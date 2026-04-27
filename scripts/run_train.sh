#!/usr/bin/env bash
# Compatibility wrapper.
# Some docs/tasks reference scripts/run_train.sh; the implementation lives at
# repo root as chain_train.sh. This wrapper avoids drift.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec bash "${REPO_ROOT}/chain_train.sh" "$@"
