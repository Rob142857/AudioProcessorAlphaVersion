param([switch]$Force)

Write-Host "[INFO] Starting conda bootstrap..." -ForegroundColor Cyan

# Detect architecture
$procArch = $env:PROCESSOR_ARCHITECTURE
$isArm = $procArch -match 'ARM'

if ($isArm) { 
    $uri = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-arm64.exe'
} else { 
    $uri = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'
}

$mf = Join-Path $env:TEMP (Split-Path $uri -Leaf)

Write-Host "[INFO] Detected arch: $procArch -> downloading $uri" -ForegroundColor Cyan

# Download with curl first
if (-not (Test-Path $mf) -or $Force) {
    Write-Host "[INFO] Downloading Miniforge installer..." -ForegroundColor Cyan
    try {
        & curl.exe -L -o $mf $uri
    } catch {
        Write-Host "[ERROR] Download failed" -ForegroundColor Red
        return
    }
}

# Validate size
try {
    $sizeMB = [math]::Round((Get-Item $mf).Length/1MB,2)
    Write-Host "[INFO] Downloaded: $mf ($sizeMB MB)" -ForegroundColor Cyan
} catch {
    Write-Host "[ERROR] Download missing or inaccessible: $mf" -ForegroundColor Red
    return
}

if ($sizeMB -lt 1) {
    Write-Host "[WARN] File suspiciously small - may be HTML error page" -ForegroundColor Yellow
    return
}

# Run installer
Write-Host "[INFO] Running installer elevated..." -ForegroundColor Cyan
try {
    Start-Process -FilePath $mf -Verb RunAs -Wait
} catch {
    Write-Host "[ERROR] Failed to start installer: $_" -ForegroundColor Red
    return
}

# Find conda
$condaExe = Join-Path $env:USERPROFILE 'Miniforge3\Scripts\conda.exe'
if (-not (Test-Path $condaExe)) {
    Write-Host "[WARN] conda.exe not found at $condaExe" -ForegroundColor Yellow
    Write-Host "Please open a new PowerShell and run: conda activate speech2textrme"
    return
}

Write-Host "[INFO] Using conda at: $condaExe" -ForegroundColor Cyan

# Create environment
Write-Host "[INFO] Creating conda environment..." -ForegroundColor Cyan
& "$condaExe" create -n speech2textrme python=3.11 -y

# Install packages
Write-Host "[INFO] Installing binary packages..." -ForegroundColor Cyan
& "$condaExe" run -n speech2textrme conda install -c conda-forge numpy numba meson ninja -y

# Install requirements
$reqPath = Join-Path $PSScriptRoot 'requirements.txt'
if (Test-Path $reqPath) {
    Write-Host "[INFO] Installing requirements.txt..." -ForegroundColor Cyan
    & "$condaExe" run -n speech2textrme python -m pip install --upgrade pip
    & "$condaExe" run -n speech2textrme python -m pip install -r "$reqPath"
}

# Preload Whisper medium model
$preloadPath = Join-Path $PSScriptRoot 'preload_models.py'
if (Test-Path $preloadPath) {
    Write-Host "[INFO] Preloading Whisper medium model..." -ForegroundColor Cyan
    & "$condaExe" run -n speech2textrme python "$preloadPath"
}

Write-Host "Bootstrap complete!" -ForegroundColor Green
Write-Host "To activate: conda activate speech2textrme" -ForegroundColor Cyan
