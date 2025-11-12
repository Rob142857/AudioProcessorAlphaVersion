Param(
    [switch]$Preview
)

Write-Host "=== Speech to Text Transcription Tool — Bootstrap ===" -ForegroundColor Green

try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force | Out-Null
} catch {}

$repoUrl = 'https://github.com/Rob142857/AudioProcessorAlphaVersion'
$installScript = 'install.ps1'

if (Test-Path $installScript) {
    Write-Host "Running local installer: $installScript" -ForegroundColor Yellow
    .\install.ps1
    return
}

Write-Host "Cloning or downloading repository..." -ForegroundColor Yellow
$work = Join-Path $env:TEMP "speech2text-setup"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null
Set-Location $work

if (Get-Command git -ErrorAction SilentlyContinue) {
    git clone $repoUrl repo | Out-Null
    Set-Location .\repo
} else {
    Invoke-WebRequest "$repoUrl/archive/refs/heads/main.zip" -OutFile repo.zip
    Expand-Archive -Path repo.zip -DestinationPath . -Force
    Set-Location .\AudioProcessorAlphaVersion-main
}

Write-Host "Running installer..." -ForegroundColor Yellow
& powershell -ExecutionPolicy Bypass -File .\install.ps1

Write-Host "Done." -ForegroundColor Green
