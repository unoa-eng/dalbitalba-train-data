# OMNI-AUDIT 로컬 스케줄러

세션 독립·머신 켜두면 알아서 돌아가는 주기 실행 도구 모음.

## 파일

| 파일 | 역할 |
|---|---|
| `omni-cycle.sh` | 실제 실행 엔트리 — OMNI-AUDIT → bench 빌드 → multi-judge (페어드) |
| `install-wsl-cron.sh` | WSL 환경에서 cron 등록 (매일 09:00 기본) |
| `install-windows-task.ps1` | Windows Task Scheduler 등록 (PowerShell) |

## 트리거 3종 (전부 세션 독립)

| 트리거 | 등록 방법 | 조건 |
|---|---|---|
| WSL cron | `bash scripts/scheduler/install-wsl-cron.sh` | WSL 실행 중 + cron 데몬 활성 |
| Windows Task Scheduler | `powershell -File scripts/scheduler/install-windows-task.ps1` | 머신 켜있음 (Sleep 깨어나면 실행) |
| GitHub Actions | 저장소 main 푸시 또는 매주 월요일 09:00 KST | 저장소·Secrets 등록만 필요 |

## 빠른 시작

### 옵션 A — Windows Task Scheduler (권장)

```powershell
# 관리자 PowerShell
cd C:\Users\mapdr\Downloads\dalbitalba
powershell -ExecutionPolicy Bypass -File scripts\scheduler\install-windows-task.ps1
# 기본: 매일 09:00 에 WSL 내부 omni-cycle.sh 호출

# 확인
Get-ScheduledTask -TaskName "OMNI-Audit-Cycle"

# 수동 실행
Start-ScheduledTask -TaskName "OMNI-Audit-Cycle"

# 제거
powershell -File scripts\scheduler\install-windows-task.ps1 -Uninstall
```

**장점**: Windows 네이티브, 머신 sleep 후 깨어나면 자동 실행, WSL2 자동 기동.
**전제**: WSL2 + `Ubuntu` distro + `wsl.exe` PATH 접근.

### 옵션 B — WSL cron

```bash
# WSL 터미널
cd ~/projects/dalbitalba  # 레포 위치
bash scripts/scheduler/install-wsl-cron.sh

# 확인
crontab -l | grep OMNI

# 로그
ls -lt .omc/audit/logs/ | head

# 제거
UNINSTALL=1 bash scripts/scheduler/install-wsl-cron.sh
```

**장점**: Linux 표준, 단순.
**단점**: WSL 세션이 떠 있어야 함 (머신 부팅 후 WSL 터미널 한 번은 열어야).

자동 기동 설정 (선택):
```bash
# /etc/wsl.conf
[boot]
command = service cron start
```

### 옵션 C — GitHub Actions (클라우드)

머신 꺼져 있어도 동작. 이미 `.github/workflows/omni-audit.yml` 등록됨.

```yaml
# 트리거:
#  - schedule: 매주 월요일 00:00 UTC (= 09:00 KST)
#  - push: main 또는 feat/ai-indistinguishability-refine 브랜치
#  - workflow_dispatch: 수동
```

Secrets 등록:
```
Settings → Secrets and variables → Actions →
 - ANTHROPIC_API_KEY
 - OPENAI_API_KEY
 - HF_TOKEN
```

⚠️ 크롤 원본(비공개)을 CI 에서 접근하려면 별도 artifact sync 필요. 없으면 자동 skip.

## 환경 변수

모든 스케줄러가 공유:

| 변수 | 기본 | 설명 |
|---|---|---|
| `CRAWL_SRC` | `/mnt/c/Users/mapdr/Desktop/queenalba-crawler/crawled-data-v2` | 크롤 JSON 디렉토리 |
| `ANTHROPIC_API_KEY` | — | 없으면 judge dry-run |
| `OPENAI_API_KEY` | — | 없으면 GPT judge dry-run |
| `HF_TOKEN` | — | 없으면 HF 탐지기 dry-run |
| `DRIFT_WARN` / `DRIFT_BLOCK` | 0.05 / 0.15 | run-corpus-drift 임계 |
| `INDIST_THRESHOLD` | 0.70 | indist 회귀 gate |

## 로그 & 보고서

| 위치 | 내용 |
|---|---|
| `.omc/audit/logs/cycle-YYYY-MM-DD_HHMMSS.log` | 사이클 전체 stdout |
| `.omc/audit/omni-YYYY-MM-DD_HHMMSS.json` | OMNI 9관점 verdict |
| `.omc/audit/bench-YYYY-MM-DD_HHMMSS/` | 블라인드 bench + judge 출력 |

실패 시:
- BLOCK (exit 3) → `corpus-stats.json` 보강 필요 → [[_system/GAP-ANALYSIS-2026-04-21]]
- WARN (exit 2) → 다음 sprint 백로그
- 블라인드 정확도 > 65% → 회귀 → [[연구/실험/실패-사례/00-MOC]]

## 수동 테스트

각 레이어 개별 검증:

```bash
# 1) OMNI-AUDIT 단독
node apps/web/scripts/run-omni-audit.mjs

# 2) 페어드 사이클 (로컬, API 키 있으면 실 judge)
bash scripts/scheduler/omni-cycle.sh

# 3) Makefile
cd apps/web/scripts/analysis
make cycle             # OMNI + 블라인드
make omni-audit        # OMNI 만
make blind-cycle       # 블라인드 만
```

## 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `CRAWL_SRC 미존재` | 경로 다름 | 환경변수 export 또는 omni-cycle.sh 기본값 수정 |
| `crontab: command not found` | WSL cron 미설치 | `sudo apt-get install -y cron && sudo service cron start` |
| Task Scheduler 실행 안 됨 | WSL2 미기동 | PowerShell 에서 `wsl` 한 번 실행 |
| HF judge 401 | 무료 tier rate limit | 간격 늘리거나 유료 tier |
| 모든 judge dry | API 키 미설정 | `.env.local` 확인 |

## 🔗

- `apps/web/scripts/run-omni-audit.mjs` — 측정기
- `apps/web/scripts/build-scaleup-bench.mjs` — bench 빌더
- `apps/web/scripts/llm-judge.mjs` — multi-judge
- `.github/workflows/omni-audit.yml` — CI workflow
- 저장소 외부 문서:
  - `_system/OMNI-AUDIT-FRAMEWORK.md` — 12관점 정의
  - `_system/API-KEY-CONFIG.md` — 키 구성
