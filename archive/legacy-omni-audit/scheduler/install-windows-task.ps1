# Windows Task Scheduler 에 OMNI-AUDIT + 블라인드 페어드 사이클 등록.
# WSL2 내부의 omni-cycle.sh 를 주기 호출.
#
# 사용:
#   powershell -ExecutionPolicy Bypass -File scripts/scheduler/install-windows-task.ps1
#
# 옵션:
#   -Time "09:00"             # 실행 시각 (로컬 KST)
#   -DistroName "Ubuntu"      # wsl.exe -d <name>
#   -TaskName "OMNI-Audit"    # Task Scheduler 이름
#   -Uninstall                # 제거

param(
    [string]$Time = "09:00",
    [string]$DistroName = "Ubuntu",
    [string]$TaskName = "OMNI-Audit-Cycle",
    [string]$RepoPath = "/mnt/c/Users/mapdr/Downloads/dalbitalba",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "✓ Task 제거: $TaskName" -ForegroundColor Green
    } else {
        Write-Host "ℹ️ Task 미존재" -ForegroundColor Yellow
    }
    exit 0
}

# WSL cron 호출 명령
$BashCmd = "cd '$RepoPath' && bash scripts/scheduler/omni-cycle.sh"
$WslArgs = "-d $DistroName -- bash -lc `"$BashCmd`""

$Action = New-ScheduledTaskAction `
    -Execute "wsl.exe" `
    -Argument $WslArgs

# 매일 지정 시각
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time

# 머신 켜져 있고 유저 로그인 없어도 실행
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# 현재 유저 컨텍스트 (WSL 접근 권한)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# 기존 동일 이름 제거
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "OMNI-AUDIT 12관점 + 블라인드 페어드 사이클 (세션 독립, 머신 켜있으면 자동 실행)" | Out-Null

Write-Host "✓ Task Scheduler 등록 완료" -ForegroundColor Green
Write-Host ""
Write-Host "이름:    $TaskName"
Write-Host "시간:    매일 $Time"
Write-Host "명령:    wsl $WslArgs"
Write-Host ""
Write-Host "확인:    Get-ScheduledTask -TaskName $TaskName"
Write-Host "수동실행: Start-ScheduledTask -TaskName $TaskName"
Write-Host "제거:    .\install-windows-task.ps1 -Uninstall"
Write-Host ""
Write-Host "※ 머신이 꺼져 있으면 실행 안 됨. Sleep 상태면 깨어난 후 실행 (StartWhenAvailable)"
