#!/usr/bin/env bash
# WSL cron 에 OMNI + 블라인드 주기 등록.
#
# Usage:
#   bash scripts/scheduler/install-wsl-cron.sh               # 매일 09:00 KST
#   CRON="0 9 * * 1" bash scripts/scheduler/install-wsl-cron.sh  # 커스텀 (월 09:00)
#   UNINSTALL=1 bash scripts/scheduler/install-wsl-cron.sh   # 제거

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENTRY="$REPO_ROOT/scripts/scheduler/omni-cycle.sh"
TAG="# OMNI-AUDIT-CYCLE"
SCHEDULE="${CRON:-0 9 * * *}"

if [ ! -x "$ENTRY" ]; then
  chmod +x "$ENTRY"
fi

if ! command -v crontab >/dev/null; then
  echo "❌ crontab 미설치. 설치: sudo apt-get install -y cron"
  exit 1
fi
# cron 데몬 체크
if ! pgrep -x cron >/dev/null && ! pgrep -x crond >/dev/null; then
  echo "⚠️  cron 데몬 미실행. 수동 시작:"
  echo "    sudo service cron start"
  echo "    # WSL 자동 시작: /etc/wsl.conf 에 [boot] command = service cron start 추가"
fi

# UNINSTALL
if [ "${UNINSTALL:-}" = "1" ]; then
  crontab -l 2>/dev/null | grep -v "$TAG" | crontab -
  echo "✓ cron 항목 제거"
  exit 0
fi

# 기존 태그 삭제 후 재삽입
CURRENT=$(crontab -l 2>/dev/null | grep -v "$TAG" || true)
LINE="$SCHEDULE bash $ENTRY $TAG"

printf '%s\n%s\n' "$CURRENT" "$LINE" | crontab -

echo "✓ 등록 완료:"
echo "  $LINE"
echo ""
echo "확인:    crontab -l | grep OMNI"
echo "제거:    UNINSTALL=1 bash $0"
echo "로그:    ls -lt .omc/audit/logs/ | head"
