#!/usr/bin/env bash
# OMNI-AUDIT + 블라인드 페어드 사이클 — 로컬 자동 실행 엔트리.
# WSL cron, Windows Task Scheduler, systemd timer 어디서든 호출 가능.

set -euo pipefail

# --- 경로 고정 (심볼릭 링크 호환) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- 환경 ---
export CRAWL_SRC="${CRAWL_SRC:-/mnt/c/Users/mapdr/Desktop/queenalba-crawler/crawled-data-v2}"
export NODE_PATH="${NODE_PATH:-}"
LOG_DIR="$REPO_ROOT/.omc/audit/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y-%m-%d_%H%M%S)
LOG="$LOG_DIR/cycle-$TS.log"

exec > >(tee -a "$LOG") 2>&1

echo "==================================================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] OMNI + BLIND 페어드 사이클 시작"
echo "REPO: $REPO_ROOT"
echo "CRAWL: $CRAWL_SRC"
echo "LOG: $LOG"
echo "==================================================================="

if [ ! -d "$CRAWL_SRC" ]; then
  echo "❌ CRAWL_SRC 미존재 — 스킵"
  exit 0
fi

# --- 1) OMNI-AUDIT 12관점 ---
echo ""
echo "[1/3] OMNI-AUDIT 실행..."
if ! node apps/web/scripts/run-omni-audit.mjs \
     --stats apps/web/lib/ai/corpus-stats.json \
     --report ".omc/audit/omni-$TS.json"; then
  rc=$?
  echo "⚠️  OMNI-AUDIT exit=$rc (PASS=0 WARN=2 BLOCK=3)"
fi

# --- 2) 블라인드 bench (n=50) ---
echo ""
echo "[2/3] 블라인드 bench 빌드..."
BENCH_DIR=".omc/audit/bench-$TS"
node apps/web/scripts/build-scaleup-bench.mjs \
  --version "cycle-$TS" \
  --n 50 \
  --out "$BENCH_DIR" || echo "⚠️  bench 빌드 부분 실패 (AI 샘플 없을 수 있음)"

# --- 3) Multi-judge (API 키 없으면 dry) ---
echo ""
echo "[3/3] Multi-judge 실행..."
DRY_FLAG=""
[ -z "${ANTHROPIC_API_KEY:-}" ] && DRY_FLAG="--dry"

if [ -f "$BENCH_DIR/samples.jsonl" ]; then
  node apps/web/scripts/llm-judge.mjs \
    --judge claude-opus-4-7 \
    --samples "$BENCH_DIR/samples.jsonl" \
    --out "$BENCH_DIR/claude.json" \
    $DRY_FLAG || echo "⚠️  judge 실패"
fi

echo ""
echo "==================================================================="
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 사이클 완료"
echo "보고서: .omc/audit/omni-$TS.json"
echo "벤치:   $BENCH_DIR/"
echo "로그:   $LOG"
echo "==================================================================="
