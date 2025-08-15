<#
run_conda.ps1
Robust Miniforge -> conda bootstrap for Windows (x86_64 or ARM64)
Place this script in the project root (next to requirements.txt) and run from PowerShell.
Usage (from project root):
  powershell -ExecutionPolicy RemoteSigned -File .\run_conda.ps1
This script will:
 - detect platform (ARM64 vs x86_64)
 - try multiple download methods for the Miniforge installer
 - validate the downloaded file (size + PE header)
 - run the installer elevated (interactive)
 - locate conda.exe and create a conda env named 'speech2textrme'
 - install binary deps from conda-forge (numpy, numba, meson, ninja)
 - pip-install the remaining requirements.txt into the conda env

If automatic download fails, the script will instruct you to download manually.
#>

param(
    [switch]$Force
)

function Write-Info($m){ Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Warn($m){ Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err($m){ Write-Host "[ERROR] $m" -ForegroundColor Red }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$reqPath = Join-Path $scriptDir 'requirements.txt'

Write-Info "Starting conda bootstrap (script dir: $scriptDir)"

# Detect architecture (best-effort)
$procArch = $env:PROCESSOR_ARCHITECTURE
if ($procArch -match 'ARM') { $isArm = $true } else { $isArm = $false }

if ($isArm) { $uri = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-arm64.exe' }
else { $uri = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe' }

$mf = Join-Path $env:TEMP (Split-Path $uri -Leaf)

Write-Info "Detected arch: $procArch -> downloading $uri"

function Invoke-Download($uri,$out){
    # Attempt curl.exe -L
    if (Get-Command curl.exe -ErrorAction SilentlyContinue){
        Write-Info "Trying curl.exe -L ..."
        try{
            & curl.exe -L -o $out $uri
            if (Test-Path $out) { return $true }
        } catch { }
    }
    # Try Start-BitsTransfer if available
    if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue){
        Write-Info "Trying Start-BitsTransfer ..."
        try{ Start-BitsTransfer -Source $uri -Destination $out -ErrorAction Stop; if (Test-Path $out) { return $true } } catch { }
    }
    # Try Invoke-WebRequest with UA (older PowerShell)
    Write-Info "Trying Invoke-WebRequest ..."
    try{
        $hdr = @{ 'User-Agent' = 'Mozilla/5.0 (Windows NT) PowerShell' }
        Invoke-WebRequest -Uri $uri -Headers $hdr -OutFile $out -UseBasicParsing -ErrorAction Stop
        if (Test-Path $out) { return $true }
    } catch { }
    return $false
}
if (-not (Test-Path $mf) -or $Force) {
    Write-Info "Downloading Miniforge installer to: $mf"
    if (-not (Invoke-Download $uri $mf)){
        Write-Warn "Automatic download failed. The release page may be blocking non-browser clients."
        Write-Host "Please open the following URL in a browser and download the appropriate installer, then re-run this script when the file is saved to:`n  $mf`n"
        Write-Host "Release page: https://github.com/conda-forge/miniforge/releases/latest" -ForegroundColor Yellow
        return
    }
}

# Validate file size
try{
    $sizeMB = [math]::Round((Get-Item $mf).Length/1MB,2)
    Write-Info "Downloaded: $mf ($sizeMB MB)"
} catch {
    Write-Err "Download missing or inaccessible: $mf"
    return
}

if ($sizeMB -lt 1){
    Write-Warn "Downloaded file is suspiciously small (<1MB). It may be an HTML error page."
    Write-Host "Open the file in Notepad to inspect, or download manually from the release page: https://github.com/conda-forge/miniforge/releases/latest" -ForegroundColor Yellow
    return
}

# Validate PE header (MZ)
try{
    $fs = [System.IO.File]::OpenRead($mf)
    $b1 = $fs.ReadByte()
    $b2 = $fs.ReadByte()
    $fs.Close()
    if (($b1 -ne 0x4D) -or ($b2 -ne 0x5A)){
        Write-Warn "Downloaded file does not look like a Windows PE executable (missing 'MZ' header)."
        Write-Host "Open $mf in Notepad and check if it's HTML. If so, download manually from the release page." -ForegroundColor Yellow
        return
    }
} catch {
    Write-Err "Failed to read downloaded file header: $_"
    return
}

Write-Info "Installer appears valid. Running installer elevated (interactive). Follow prompts in the installer window..."
try{
    Start-Process -FilePath $mf -Verb RunAs -Wait
} catch {
    Write-Err "Failed to start the installer: $_"
    Write-Host "If Start-Process reported the file is not valid for this OS platform, ensure you downloaded the correct installer for your CPU architecture." -ForegroundColor Yellow
    return
}

Write-Info "Installer process finished. Locating conda.exe (this may take a few seconds)..."

$possible = @(
    (Join-Path $env:USERPROFILE 'Miniforge3\Scripts\conda.exe'),
    (Join-Path $env:ProgramFiles 'Miniforge3\Scripts\conda.exe'),
    (Join-Path $env:ProgramFiles 'Miniconda3\Scripts\conda.exe'),
    (Join-Path $env:USERPROFILE 'Miniconda3\Scripts\conda.exe')
)

# Filter existing
$condaExe = $possible | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $condaExe){
    Write-Info "Searching recursively under $env:USERPROFILE for conda.exe (this may take a minute)..."
    try{
        $found = Get-ChildItem -Path $env:USERPROFILE -Filter 'conda.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { $condaExe = $found.FullName }
    } catch { }
}

if (-not $condaExe){
    Write-Warn "conda.exe was not found automatically. Common location: $env:USERPROFILE\Miniforge3\Scripts\conda.exe"
    Write-Host "If you installed Miniforge to a custom path, either open a NEW PowerShell (so conda is on PATH) and run 'conda activate' there, or paste the full path to conda.exe now."
    $manual = Read-Host "Enter full path to conda.exe (or press Enter to abort)"
    if ([string]::IsNullOrWhiteSpace($manual)) { Write-Host "Aborting: installer ran but conda not found. Open a new shell or provide the path manually."; return }
    if (-not (Test-Path $manual)) { Write-Err "Provided path not found: $manual"; return }
    $condaExe = $manual
}

Write-Info "Using conda at: $condaExe"

# Create environment
$envName = 'speech2textrme'
Write-Info "Creating conda environment '$envName' with python=3.11 (may take a minute)..."
try{
    & "$condaExe" create -n $envName python=3.11 -y | Write-Host
} catch {
    Write-Err "Failed to create conda env: $_"
    return
}

Write-Info "Installing binary packages from conda-forge: numpy, numba, meson, ninja"
try{
    & "$condaExe" run -n $envName conda install -c conda-forge numpy numba meson ninja -y | Write-Host
} catch {
    Write-Warn "conda install failed or was interrupted. You can try running:`n  & `"$condaExe`" run -n $envName conda install -c conda-forge numpy numba meson ninja -y`nManually in a new shell."
}

# Upgrade pip and pip-install remaining requirements
if (Test-Path $reqPath){
    Write-Info "Upgrading pip inside the new env and installing $reqPath via pip"
    try{ & "$condaExe" run -n $envName python -m pip install --upgrade pip | Write-Host } catch {}
    try{
        & "$condaExe" run -n $envName python -m pip install -r "$reqPath" | Write-Host
    } catch {
        Write-Warn "pip install -r requirements.txt failed. Try running the command manually in an activated conda shell." 
    }
} else {
    Write-Warn "requirements.txt not found at $reqPath — skipping pip install step."
}

Write-Host "[INFO] Done. To start using the environment in a NEW PowerShell session, run:`n  conda activate $envName" -ForegroundColor Cyan
Write-Host '[INFO] If you plan to install a specific PyTorch wheel (CUDA/DirectML), install it after activating the env as documented in the README.' -ForegroundColor Cyan

