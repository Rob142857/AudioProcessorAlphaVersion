# One-liner installer for Speech to Text Transcription Tool
# Supports both ARM64 and x64 architectures

param(
    [switch]$PyTorch,
    [switch]$WhisperCPP,
    [switch]$ARM64
)

Write-Host "=== Speech to Text Transcription Tool - Universal Installer ===" -ForegroundColor Green

# Detect architecture
$arch = $env:PROCESSOR_ARCHITECTURE
Write-Host "Detected architecture: $arch" -ForegroundColor Yellow

if ($ARM64 -or $arch -eq "ARM64") {
    Write-Host "Installing ARM64 version with NPU acceleration..." -ForegroundColor Cyan

    # ARM64 installation
    $armPath = "whispercpp_arm"
    if (!(Test-Path $armPath)) {
        Write-Host "ARM64 setup not found in repository." -ForegroundColor Red
        Write-Host "Please ensure you're using the latest version from GitHub." -ForegroundColor Red
        exit 1
    }

    Set-Location $armPath

    # Setup ARM environment
    if (!(Test-Path ".venv_arm\Scripts\Activate.ps1")) {
        Write-Host "Creating ARM64 virtual environment..." -ForegroundColor Yellow
        python -m venv .venv_arm
    }

    Write-Host "Activating ARM64 environment..." -ForegroundColor Yellow
    & .venv_arm\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    Write-Host "ARM64 NPU Setup Complete!" -ForegroundColor Green
    Write-Host "Your ARM64 device is now configured with NPU acceleration!" -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "To launch the GUI:" -ForegroundColor Cyan
    Write-Host "  cd whispercpp_arm" -ForegroundColor White
    Write-Host "  python gui_transcribe.py" -ForegroundColor White
    Write-Host "" -ForegroundColor White
    Write-Host "The GUI will show 'Whisper.cpp ARM64 Transcription (NPU Accelerated)'" -ForegroundColor Green
    Write-Host "" -ForegroundColor White

} elseif ($arch -eq "AMD64") {
    Write-Host "Installing x64 version..." -ForegroundColor Cyan

    if ($WhisperCPP) {
        Write-Host "Installing WhisperCPP version..." -ForegroundColor Cyan
        # Simulate choice 2 in install.bat
        $choice = "2"
    } elseif ($PyTorch) {
        Write-Host "Installing PyTorch version..." -ForegroundColor Cyan
        $choice = "1"
    } else {
        # Interactive choice
        Write-Host "Choose your preferred backend:" -ForegroundColor Yellow
        Write-Host "[1] PyTorch Whisper (GPU/CPU, large model)" -ForegroundColor White
        Write-Host "[2] WhisperCPP Turbo v3 (CPU-only, fastest)" -ForegroundColor White
        $choice = Read-Host "Enter your choice (1 or 2)"
    }

    if ($choice -eq "1") {
        Write-Host "Installing PyTorch version..." -ForegroundColor Cyan
        & .\install.bat
    } elseif ($choice -eq "2") {
        Write-Host "Installing WhisperCPP version..." -ForegroundColor Cyan
        # Pass choice to install.bat
        Write-Output "2" | & .\install.bat
    } else {
        Write-Host "Invalid choice. Defaulting to PyTorch..." -ForegroundColor Yellow
        & .\install.bat
    }

} else {
    Write-Host "Unsupported architecture: $arch" -ForegroundColor Red
    Write-Host "This installer supports ARM64 and x64 only." -ForegroundColor Red
    exit 1
}

Write-Host "=== Installation Complete ===" -ForegroundColor Green
