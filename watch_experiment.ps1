# 실험 진행 상황을 터미널에서 지켜본다.
#
#   .\watch_experiment.ps1              센서 고장 실험
#   .\watch_experiment.ps1 train.log    학습 로그
#
# Ctrl+C 로 빠져나온다. 실험 자체는 계속 돈다.

param(
    [string]$Log = "logs/fault_out.log",
    [int]$Every = 5
)

$path = Join-Path $PSScriptRoot $Log
if (-not (Test-Path $path)) {
    Write-Host "$Log 가 없습니다." -ForegroundColor Red
    exit 1
}

# 이 실험이 돌려야 할 전체 조건 수 (깨끗 상태 제외)
$total = 24

while ($true) {
    Clear-Host
    $lines = Get-Content $path -Encoding UTF8
    $done = @($lines | Where-Object { $_ -match '완주.*유지' }).Count

    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like '*experiment_*' } |
            Select-Object -First 1

    Write-Host ("=" * 62) -ForegroundColor DarkGray
    if ($proc) {
        $started = (Get-Process -Id $proc.ProcessId).StartTime
        $elapsed = (Get-Date) - $started
        Write-Host ("  실험 진행 중      경과 {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Green
        if ($done -gt 0 -and $done -lt $total) {
            $per = $elapsed.TotalSeconds / $done
            $left = [TimeSpan]::FromSeconds($per * ($total - $done))
            Write-Host ("  조건 {0}/{1}          남음 약 {2:hh\:mm\:ss}" -f $done, $total, $left)
        } else {
            Write-Host ("  조건 {0}/{1}" -f $done, $total)
        }
        $filled = [int](20 * $done / $total)
        Write-Host ("  [" + ("#" * $filled) + ("." * (20 - $filled)) + "]")
    } else {
        Write-Host "  실험 종료됨" -ForegroundColor Yellow
    }
    Write-Host ("=" * 62) -ForegroundColor DarkGray
    Write-Host ""

    # 결과 줄만 추려서 보여준다
    $lines | Where-Object {
        $_ -match '완주.*유지' -or $_ -match '^[가-힣].*$' -or $_ -match '손상 없음'
    } | Select-Object -Last 30 | ForEach-Object {
        if ($_ -match '유지\s+(\d+)%') {
            $keep = [int]$Matches[1]
            $color = if ($keep -ge 80) { "Green" }
                     elseif ($keep -ge 50) { "Yellow" } else { "Red" }
            Write-Host $_ -ForegroundColor $color
        } else {
            Write-Host $_ -ForegroundColor Cyan
        }
    }

    if (-not $proc) { break }
    Write-Host ""
    Write-Host "  $Every 초마다 새로고침.  Ctrl+C 로 나가기" -ForegroundColor DarkGray
    Start-Sleep -Seconds $Every
}
