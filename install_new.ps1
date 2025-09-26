#requires -version 5.1
<#
.SYNOPSIS
  Fresh Windows x64 installer for the AudioProcessor toolchain.

.DESCRIPTION
  - Verifies x64 Windows and (optionally) admin rights for system installs
  - Ensures Python 3.11+, creates a local venv (.venv)
  - Installs PyTorch for CPU or CUDA (auto-detect by default)
  - Installs requirements.txt
  - Ensures ffmpeg (via winget if missing; otherwise uses bundled ffmpeg.exe)
  - Optionally installs C++ Build Tools (VS 2022 Build Tools VCTools workload)
  - Preloads Whisper model (prefers large-v3-turbo)
  - Writes a short summary and how-to-run instructions

.PARAMETER Device
  Select target: cpu | cuda | auto (default: auto). Auto uses NVIDIA detection.

.PARAMETER InstallBuildTools
  Install Microsoft Visual Studio 2022 Build Tools (VCTools workload). Requires admin + winget.

.PARAMETER NoWinget
  Skip winget usage entirely (useful in restricted environments).

.PARAMETER SkipModelPreload
  Skip model preloading step.

.PARAMETER Quiet
  Reduce output verbosity.

.EXAMPLE
  # Run in current repo folder
  powershell -ExecutionPolicy Bypass -File .\install_new.ps1 -Device auto

.NOTES
  - Focuses on Windows x64. DirectML/ARM paths are not included here.
  - CUDA Toolkit is NOT required; the PyTorch CUDA wheels include needed runtime libs.
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [ValidateSet('auto','cpu','cuda')]
  [string]$Device = 'auto',

  [switch]$InstallBuildTools,
  [switch]$NoWinget,
  [switch]$SkipModelPreload,
  [switch]$Quiet
)

function Write-Info($msg){ if(-not $Quiet){ Write-Host $msg -ForegroundColor Cyan } }
function Write-Ok($msg){ Write-Host $msg -ForegroundColor Green }
function Write-Warn($msg){ Write-Warning $msg }
function Write-Err($msg){ Write-Host $msg -ForegroundColor Red }

# Basic environment checks
if(-not [Environment]::Is64BitOperatingSystem){
  Write-Err "This installer targets Windows x64 only."; exit 1
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
Write-Info "Repo root: $RepoRoot"

# Helper: test command
function Test-Command($name){ $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }

# Helper: run and fail on error (approved verb; avoid $Args automatic var)
function Invoke-ProcessChecked([string]$Cmd, [string[]]$ArgList){
  Write-Info ("`n> {0} {1}" -f $Cmd, ($ArgList -join ' '))
  $p = Start-Process -FilePath $Cmd -ArgumentList $ArgList -NoNewWindow -PassThru -Wait
  if($p.ExitCode -ne 0){ throw "Command failed ($Cmd) with exit code $($p.ExitCode)" }
}

# Admin check for system installs
$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if($InstallBuildTools -and -not $IsAdmin){
  Write-Warn "Build Tools installation requested but this session is not elevated. Please re-run PowerShell as Administrator or omit -InstallBuildTools."
}

# 1) Ensure winget if needed
if(-not $NoWinget){
  if(-not (Test-Command 'winget')){
    Write-Warn "winget not found; skipping winget-based installs."
    $NoWinget = $true
  }
}

# 2) Ensure Python 3.11+
$PythonCmd = $null
foreach($cand in @('python','py')){
  if(Test-Command $cand){ $PythonCmd = $cand; break }
}
if(-not $PythonCmd -and -not $NoWinget){
  if($IsAdmin){
    Write-Info "Installing Python 3.11 via winget..."
  try{ Invoke-ProcessChecked 'winget' @('install','-e','--id','Python.Python.3.11','-s','winget','--silent') } catch { Write-Err $_ }
    if(Test-Command 'python'){ $PythonCmd = 'python' }
  } else {
    Write-Warn "Python not found and not elevated. Please install Python 3.11+ (Microsoft Store or python.org) and re-run."
  }
}
if(-not $PythonCmd){ $PythonCmd = 'python' } # try anyway

# Resolve python exe path
try {
  $pyver = & $PythonCmd -c "import sys;print(sys.version)" 2>$null
  Write-Info "Python: $pyver"
} catch {
  Write-Warn "Python command not responding; will attempt to continue."
}

# 3) Create virtual environment
$VenvDir = Join-Path $RepoRoot '.venv'
if(-not (Test-Path $VenvDir)){
  Write-Info "Creating virtual environment in .venv..."
  & $PythonCmd -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
if(-not (Test-Path $VenvPython)){
  Write-Err "Virtual environment not created correctly. Aborting."; exit 1
}

# Upgrade pip/setuptools/wheel
Invoke-ProcessChecked $VenvPython @('-m','pip','install','--upgrade','pip','setuptools','wheel')

# 4) Decide Torch variant (auto/cpu/cuda)
function Test-NvidiaGpu(){
  try{
    $gpu = Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'NVIDIA' }
    if($gpu){ return $true }
  } catch {}
  if(Test-Command 'nvidia-smi'){ return $true }
  return $false
}

$tgt = $Device
if($tgt -eq 'auto'){
  $tgt = (Test-NvidiaGpu) ? 'cuda' : 'cpu'
}
Write-Info "Selected Torch target: $tgt"

# 5) Install Torch
if($tgt -eq 'cpu'){
  Invoke-ProcessChecked $VenvPython @('-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch','torchvision','torchaudio')
} elseif($tgt -eq 'cuda'){
  try{
    # Default to cu121 wheels (adjust if your drivers require a different build)
    Invoke-ProcessChecked $VenvPython @('-m','pip','install','--index-url','https://download.pytorch.org/whl/cu121','torch','torchvision','torchaudio')
  } catch {
    Write-Warn "PyTorch CUDA install failed, falling back to CPU wheels..."
    Invoke-ProcessChecked $VenvPython @('-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch','torchvision','torchaudio')
    $tgt = 'cpu'
  }
}

# 6) Install project requirements
if(Test-Path (Join-Path $RepoRoot 'requirements.txt')){
  Invoke-ProcessChecked $VenvPython @('-m','pip','install','-r',(Join-Path $RepoRoot 'requirements.txt'))
} else {
  Write-Warn "requirements.txt not found; installing main libs directly"
  Invoke-ProcessChecked $VenvPython @('-m','pip','install','openai-whisper','python-docx','deepmultilingualpunctuation','moviepy','imageio-ffmpeg','psutil')
}

# 7) Ensure ffmpeg
function Install-FFmpegIfNeeded(){
  if(Test-Command 'ffmpeg'){ Write-Info 'ffmpeg already on PATH'; return }
  $bundled = Join-Path $RepoRoot 'ffmpeg.exe'
  if(Test-Path $bundled){
    Write-Info "Using bundled ffmpeg.exe; adding repo path to user PATH"
    $current = [Environment]::GetEnvironmentVariable('PATH','User')
    if([string]::IsNullOrEmpty($current)){ $current = '' }
    if($current -notmatch [Regex]::Escape($RepoRoot)){
      [Environment]::SetEnvironmentVariable('PATH', ($current + ';' + $RepoRoot), 'User')
      Write-Info "Added $RepoRoot to user PATH (new shells required)"
    }
    return
  }
  if($NoWinget){ Write-Warn 'ffmpeg not found and winget disabled; please install ffmpeg manually.'; return }
  if(-not $IsAdmin){ Write-Warn 'ffmpeg install via winget may require elevation; attempting user install...' }
  try{
    Write-Info 'Installing ffmpeg via winget (Gyan.FFmpeg)...'
    Invoke-ProcessChecked 'winget' @('install','-e','--id','Gyan.FFmpeg','-s','winget','--silent')
  } catch {
    Write-Warn "ffmpeg (Gyan) failed: $_; trying FFmpeg.FFmpeg"
    try{ Invoke-ProcessChecked 'winget' @('install','-e','--id','FFmpeg.FFmpeg','-s','winget','--silent') } catch { Write-Warn "ffmpeg install failed: $_" }
  }
}
Install-FFmpegIfNeeded

# 8) (Optional) Install C++ Build Tools
if($InstallBuildTools -and -not $NoWinget){
  try{
    Write-Info 'Installing Visual Studio 2022 Build Tools (VCTools)...'
    $override = '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
  Invoke-ProcessChecked 'winget' @('install','-e','--id','Microsoft.VisualStudio.2022.BuildTools','-s','winget','--override',$override)
  } catch { Write-Warn "Build Tools install failed: $_" }
} elseif($InstallBuildTools) {
  Write-Warn 'Skipping Build Tools: winget is disabled (-NoWinget).'
}

# 9) Preload Whisper model (large-v3-turbo preferred)
if(-not $SkipModelPreload){
  try{
    Write-Info 'Preloading Whisper model (this may take a while)...'
    Invoke-ProcessChecked $VenvPython @('preload_models.py')
  } catch { Write-Warn "Model preload failed: $_" }
}

# 10) Optional environment check
try{ Invoke-ProcessChecked $VenvPython @('check_env.py') } catch { Write-Warn "Environment check failed: $_" }

# Summary
Write-Ok "`nInstallation complete!"
try {
  $pyv = & $VenvPython -c "import sys;print(sys.version.split()[0])"
  Write-Host ("Python: {0}" -f $pyv) -ForegroundColor Gray
} catch {}
try {
  $torchv = & $VenvPython -c "import torch;print(torch.__version__+' '+('CUDA' if torch.cuda.is_available() else 'CPU'))"
  Write-Host ("Torch:  {0}" -f $torchv) -ForegroundColor Gray
} catch {}

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1) Activate venv in a new PowerShell:  .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Yellow
Write-Host "2) Launch GUI:                       python gui_transcribe.py" -ForegroundColor Yellow
Write-Host "   (Headless example):              python gui_transcribe.py --input \"C:\\path\\to\\file.mp4\"" -ForegroundColor Yellow

exit 0
