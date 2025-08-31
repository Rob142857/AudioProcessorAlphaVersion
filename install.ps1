# Speech to Text Transcription Tool v1.0Beta Complete Installer for Virgin Windows
# Handles: Python, Visual C++, Git, repository setup, and launch

Write-Host "=== Speech to Text Transcription Tool v1.0Beta Complete Installer ===" -ForegroundColor Green
Write-Host "Installing prerequisites for virgin Windows..." -ForegroundColor Yellow

# Set execution policy if needed
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "✓ PowerShell execution policy set" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not set execution policy - continuing..." -ForegroundColor Yellow
}

# Install Python 3.11 x64 (latest stable)
Write-Host "Installing Python 3.11 x64..." -ForegroundColor Yellow
try {
    winget install --id Python.Python.3.11 --scope user --force --accept-package-agreements --accept-source-agreements
    Write-Host "✓ Python 3.11 x64 installed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Python install failed - may already be installed" -ForegroundColor Yellow
}

# Install Visual Studio Build Tools (required for compiling packages like webrtcvad)
Write-Host "Installing Visual Studio Build Tools..." -ForegroundColor Yellow
try {
    winget install --id Microsoft.VisualStudio.2022.BuildTools --force --accept-package-agreements --accept-source-agreements
    Write-Host "✓ Visual Studio Build Tools installed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Build Tools install failed - webrtcvad may not install" -ForegroundColor Yellow
}

# Install Git (optional but recommended)
Write-Host "Installing Git..." -ForegroundColor Yellow
try {
    winget install --id Git.Git --force --accept-package-agreements --accept-source-agreements
    Write-Host "✓ Git installed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Git install failed - will use ZIP download" -ForegroundColor Yellow
}

# Refresh PATH to pick up new installations
Write-Host "Refreshing environment variables..." -ForegroundColor Yellow
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "Machine")

# Navigate to Downloads and setup repository
Write-Host "Setting up repository..." -ForegroundColor Yellow
Set-Location "$env:USERPROFILE\Downloads"

if (!(Test-Path .\speech2textrme)) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host "Cloning repository with Git..." -ForegroundColor Yellow
        git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme
    } else {
        Write-Host "Downloading repository ZIP..." -ForegroundColor Yellow
        Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile repo.zip
        Expand-Archive repo.zip -Force
        Move-Item "repo\AudioProcessorAlphaVersion-main" "speech2textrme"
        Remove-Item repo.zip, repo -Recurse -Force
    }
    Write-Host "✓ Repository downloaded" -ForegroundColor Green
} else {
    Write-Host "✓ Repository already exists" -ForegroundColor Green
}

Set-Location .\speech2textrme

Write-Host "=== Prerequisites Complete! ===" -ForegroundColor Green
Write-Host "Launching Speech to Text Transcription Tool installer..." -ForegroundColor Yellow

# Launch the main installer
.\run.bat

Write-Host "=== Installation Complete! ===" -ForegroundColor Green
Write-Host "You can now run: python gui_transcribe.py" -ForegroundColor Cyan
