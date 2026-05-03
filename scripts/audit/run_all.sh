#!/usr/bin/env bash
# Run all V4 Stage 0 + Stage 1 audit scripts in sequence.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv-local/bin/activate
mkdir -p runs/audit
python scripts/audit/build_persona_index.py
python scripts/audit/build_dimension_baseline.py
python scripts/audit/build_topic_index.py
python scripts/audit/build_tone_index.py
python scripts/audit/build_vocab_candidates.py
echo "[done] V4 Stage 0+1 audit complete. Outputs in runs/audit/"
