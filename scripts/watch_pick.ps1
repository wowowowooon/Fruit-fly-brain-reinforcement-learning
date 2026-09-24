# 넓은 재측정 진행 상황. 20초마다 새로 고친다.
#   .\watch_pick.ps1
$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [Text.Encoding]::UTF8

while ($true) {
    Clear-Host

    $pick = @(Get-ChildItem (Join-Path $root "models\pick_grid_*.json")).Count
    $race = @(Get-ChildItem (Join-Path $root "models\race_grid_*.json")).Count

    Write-Host "격자 18판 넓은 재측정" -ForegroundColor Cyan
    Write-Host ("=" * 78)
    Write-Host ("  일반 45트랙 x 섭동 2 + 지그재그 12   {0,2}/36" -f $pick) -ForegroundColor $(if ($pick -ge 36) { "Green" } else { "Yellow" })
    Write-Host ("  2026 대회 코스 노면 7종              {0,2}/36" -f $race) -ForegroundColor $(if ($race -ge 36) { "Green" } elseif ($race -gt 0) { "Yellow" } else { "DarkGray" })

    # 지금 돌고 있는 모델
    $live = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -like '*grid_pick*--model*' -or $_.CommandLine -like '*grid_race*--model*' } |
        ForEach-Object {
            $m = [regex]::Match($_.CommandLine, '--model (\S+)\.pt').Groups[1].Value
            $k = if ($_.CommandLine -like '*grid_race*') { "대회" } else { "일반" }
            "$k $m"
        } | Select-Object -Unique | Sort-Object

    Write-Host ""
    Write-Host ("지금 도는 것 ({0}개)" -f $live.Count) -ForegroundColor Cyan
    foreach ($m in $live) { Write-Host ("  " + $m) -ForegroundColor DarkGray }

    # 표는 grid_report.py 가 그린다 (같은 코드로 최종 판정도 한다)
    Write-Host ""
    & $py (Join-Path $root "evaluate\grid_report.py") 2>$null |
        Select-Object -Skip 4

    Write-Host ""
    Write-Host ("새로 고침 " + (Get-Date -Format "HH:mm:ss") + "   Ctrl+C 로 종료") -ForegroundColor DarkGray
    Start-Sleep -Seconds 20
}
