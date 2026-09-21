# 학습 진행 상황을 실시간으로 본다.
# 다른 터미널에서 학습을 돌리고, 이 창에서 지켜보면 된다.
#
#   .\watch.ps1
#
# 종료는 Ctrl+C. 학습 자체는 멈추지 않는다.
#
# 주의: 이 파일은 UTF-8 BOM 으로 저장해야 한다.
# Windows PowerShell 5.1 은 BOM 이 없으면 cp949 로 읽어서 한글이 깨진다.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$log = Join-Path $PSScriptRoot "logs/train.log"

Write-Host "학습 로그를 지켜봅니다: $log" -ForegroundColor Cyan
Write-Host "(Ctrl+C 로 종료. 학습 자체는 멈추지 않습니다)" -ForegroundColor DarkGray
Write-Host ""

while (-not (Test-Path $log)) {
    Write-Host "학습 시작을 기다리는 중..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 2
}

# -Encoding UTF8 이 없으면 로그 안의 한글이 깨진다.
Get-Content -Path $log -Encoding UTF8 -Wait -Tail 60
