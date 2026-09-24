# 격자 18판 진행 상황. 5초마다 새로 고친다.
#   .\watch_grid.ps1
$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

while ($true) {
    Clear-Host
    Write-Host "격자 18판  —  CAPS 3종 x 감마 2종 x 시드 3개" -ForegroundColor Cyan
    Write-Host ("=" * 78)

    # 지금 돌고 있는 것
    $live = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -like '*--out grid_*' } |
        ForEach-Object { [regex]::Match($_.CommandLine, '--out grid_(\S+)\.pt').Groups[1].Value } |
        Select-Object -Unique

    # 18판 전부를 격자 순서대로 보여준다. 로그가 아직 없으면 '대기'.
    $order = @()
    foreach ($seed in 0, 1, 2) {
        foreach ($gam in "g99", "g998") {
            foreach ($caps in "none", "mid", "high") {
                $order += "${caps}_${gam}_s${seed}"
            }
        }
    }

    $done = 0
    foreach ($name in $order) {
        $path = Join-Path $root "logs\grid_$name.log"
        $it = "-"; $rate = "-"; $spd = "-"; $eta = "-"; $best = "-"; $now = "-"
        $state = "대기"
        if (Test-Path $path) {
            # UTF-8 로 명시해야 한다. 안 하면 ANSI 로 읽어 한글이 깨지고
            # 정규식이 하나도 안 맞는다.
            $text = (Get-Content $path -Raw -Encoding UTF8) -replace "`r", "`n"
            $lines = $text -split "`n" | Where-Object { $_ -match '강화 ' }
            if ($lines) {
                $last = $lines[-1]
                if ($last -match '강화 +(\d+)/(\d+)') { $it = "$($Matches[1])/$($Matches[2])" }
                if ($last -match '완주 +(\d+)%') { $rate = $Matches[1] + "%" }
                if ($last -match '평균속도 +([\d.]+)') { $spd = $Matches[1] }
                if ($last -match '남음 +([\d:]+)') { $eta = $Matches[1] }
                $valids = [regex]::Matches($text, '검증 +(\d+)%') |
                    ForEach-Object { [int]$_.Groups[1].Value }
                if ($valids) {
                    $best = "$(($valids | Measure-Object -Maximum).Maximum)%"
                    $now = "$($valids[-1])%"
                }
                $state = if ($text -match '강화 완료') { "끝" } else { "진행" }
                if ($state -eq "끝") { $done++ }
            }
        }
        $color = switch ($state) { "끝" { "Green" } "진행" { "Yellow" } default { "DarkGray" } }
        Write-Host ("{0,-16} {1,-4} {2,9} 완주{3,5} 속도{4,5} 검증 최고{5,5} 지금{6,5}  남음 {7}" -f
            $name, $state, $it, $rate, $spd, $best, $now, $eta) -ForegroundColor $color
    }
    Write-Host ("=" * 78)
    Write-Host ("끝난 판 {0}/18    {1}" -f $done, (Get-Date -Format "HH:mm:ss"))
    Write-Host "Ctrl+C 로 종료" -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}
