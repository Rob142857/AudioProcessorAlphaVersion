# Remote bootstrap entrypoint for irm|iex usage.
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force | Out-Null
} catch {}

Write-Host "Fetching installer..." -ForegroundColor Yellow
try {
    $script = Invoke-RestMethod -Uri "https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1"
    Invoke-Expression $script
} catch {
    Write-Host "Failed to fetch installer: $($_.Exception.Message)" -ForegroundColor Red
    throw
}
