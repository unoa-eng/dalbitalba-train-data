#!/usr/bin/env bash
# Session-independent watchdog for the dalbitalba live ops loop.
# Loaded by com.dalbit.monitor-watchdog (launchd, RunAtLoad + StartInterval).
# Survives Claude session end, logout/login, and reboot (LaunchAgents).
# Belt-and-suspenders over the daemon's own KeepAlive; adds daily-report catch-up.
HERE="/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen/scripts/live"
cd "$HERE" || exit 1
LOG="$HERE/watchdog.log"
PY="/opt/homebrew/bin/python3"
ts() { TZ=Asia/Seoul date '+%F %T'; }

# heartbeat: single-line "last alive" marker (overwritten each tick, no log bloat)
echo "watchdog alive $(ts) KST" > "$HERE/.watchdog_last"

# 1) ensure live daemon is alive (KeepAlive should handle it; this is a second net)
n=$(ps -Ao command | grep 'live/daemon.py' | grep -i Python | grep -v grep | wc -l | tr -d ' ')
if [ "$n" -lt 1 ]; then
  echo "[$(ts)] daemon DOWN (count=$n) -> kickstart com.dalbit.live-daemon" >> "$LOG"
  launchctl kickstart -k "gui/$(id -u)/com.dalbit.live-daemon" >> "$LOG" 2>&1
  "$PY" -c "import sys;sys.path.insert(0,'$HERE');import report;report.send_telegram('🔴 달빛알바 데몬 DOWN(count=$n) → 자동 재시작 실행')" >> "$LOG" 2>&1
fi

# 1a) HANG 감지: 프로세스는 살아있으나(count>=1) 발행 루프가 멈춘 경우.
#     daemon.log 가 150분+ 정체 & comment_queue 에 overdue(release<=now 미방출) 존재하면 hang 확정.
#     (2026-07-06 count=1인데 26h 로그정체 실제 hang 발생 — count 체크만으론 못 잡던 사각지대.)
if [ "$n" -ge 1 ]; then
  stale=$(find "$HERE/daemon.log" -mmin +150 -print 2>/dev/null)
  if [ -n "$stale" ]; then
    hung=$("$PY" - <<PYEOF
import json, time, os
q = json.load(open("$HERE/comment_queue.json")) if os.path.exists("$HERE/comment_queue.json") else []
now = time.time()
overdue = sum(1 for x in q if x.get("release", 9e18) <= now)
# 로그 150분+ 정체 + overdue 방출 대기 = hang 확정. overdue 없으면(정상 유휴) 0.
print(1 if overdue > 0 else 0)
PYEOF
)
    if [ "$hung" = "1" ]; then
      echo "[$(ts)] daemon HANG 감지 (log 150m+ stale, queue overdue) -> 강제 재시작" >> "$LOG"
      launchctl kickstart -k "gui/$(id -u)/com.dalbit.live-daemon" >> "$LOG" 2>&1
      "$PY" -c "import sys;sys.path.insert(0,'$HERE');import report;report.send_telegram('🔴 달빛알바 데몬 HANG 감지(로그 150분+ 정체, 발행대기 밀림) → 강제 재시작. 발행 자동 재개.')" >> "$LOG" 2>&1
    fi
  fi
fi

# 1b) buffer 저수위 조기 경보 (텔레그램, 고갈→회복 사이클당 1회)
"$PY" "$HERE/buffer_alert.py" >> "$LOG" 2>&1

# 1c) 생성 실패 즉시 경보: produce 가 codex+claude 폴백 모두 실패하면 .produce_fail.json 기록.
#     새 실패(직전 경보 이후)면 텔레그램 즉시 발송. 성공 시 produce 가 마커를 지워 자동 리셋.
if [ -f "$HERE/.produce_fail.json" ]; then
  "$PY" - <<PYEOF >> "$LOG" 2>&1
import json, os, sys
sys.path.insert(0, "$HERE")
import report
try:
    m = json.load(open("$HERE/.produce_fail.json"))
except Exception:
    m = {}
ts = m.get("ts", 0)
seen_path = "$HERE/.produce_fail_alerted"
seen = 0
try: seen = int(open(seen_path).read().strip())
except Exception: pass
if ts and ts != seen:  # 새 실패만 경보(중복 방지)
    engs = ", ".join(f"{e.get('engine')}({e.get('why')})" for e in m.get("engines", []))
    if report.send_telegram(f"🔴 달빛알바 생성 실패 — 모든 엔진 폴백 실패\\n\\n{engs}\\n\\ncodex/claude 둘 다 막힘. buffer 소진 시 발행 중단 위험. 수동 점검 필요."):
        open(seen_path, "w").write(str(ts))
        print(f"[produce fail alert sent: {engs}]")
PYEOF
fi

# 2) daily-report catch-up: if today's report .md is missing and it's past 09:00 KST,
#    run report.py once (writes .md, prints, and delivers via .notify.json if present).
hour=$((10#$(TZ=Asia/Seoul date '+%H')))
today=$(TZ=Asia/Seoul date '+%F')
if [ "$hour" -ge 9 ] && [ ! -f "$HERE/reports/daily-$today.md" ]; then
  echo "[$(ts)] today's report ($today) missing -> running report.py" >> "$LOG"
  "$PY" "$HERE/report.py" >> "$LOG" 2>&1
  echo "[$(ts)] report.py exit=$?" >> "$LOG"
fi

exit 0
