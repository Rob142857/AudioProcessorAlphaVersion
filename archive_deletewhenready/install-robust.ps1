#Requires -Version 5.1

<#
.SYNOPSIS
    Streamlined installer for Audio Processor Alpha Version - Robust installation for new x64 machines

.DESCRIPTION
    This script performs a comprehensive, error-resilient installation of the Audio Processor Alpha Version
    on fresh Windows x64 systems. Key improvements:
    - Uses minimal dependencies (no legacy webrtcvad, moviepy, imageio)
    - Robust error handling with retry mechanisms
    - Proper validation at each step
    - Fallback options for failed components
    - Hardware detection for optimal PyTorch installation
    - Clear progress reporting and diagnostics

.PARAMETER SkipSystemComponents
    Skip system-level installations (Python, Git, FFmpeg) - useful if already installed

.PARAMETER SkipModelPreload
    Skip Whisper model preloading (saves time and bandwidth)

.PARAMETER ForceReinstall
    Force reinstallation even if components are detected

.PARAMETER CpuOnly
    Force CPU-only PyTorch installation regardless of hardware

.EXAMPLE
    # Complete installation on new machine (recommended)
    irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install-robust.ps1 | iex

    # Development mode (skip system components)
    .\install-robust.ps1 -SkipSystemComponents

.NOTES
    Version: 2.0.0
    Requires: Windows 10/11 x64, PowerShell 5.1+, Internet connection
    Runtime: 5-15 minutes depending on hardware and connection
#>

param(
    [switch]$SkipSystemComponents,
    [switch]$SkipModelPreload,
    [switch]$ForceReinstall,
    [switch]$CpuOnly
)

# ============================================================================
# Configuration and Constants
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Speed up downloads

# Repository information
$REPO_URL = "https://github.com/Rob142857/AudioProcessorAlphaVersion.git"
$REPO_NAME = "AudioProcessorAlphaVersion"
$SCRIPT_VERSION = "2.0.0"

# Required software
$PYTHON_VERSION = "3.11"
$MIN_PYTHON_VERSION = [Version]"3.11.0"

# Installation paths
$PROJECT_DIR = Join-Path $env:USERPROFILE "AudioProcessor"
$VENV_PATH = Join-Path $PROJECT_DIR ".venv"

# ============================================================================
# Utility Functions
# ============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n>>> $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-InternetConnection {
    try {
        $null = Test-NetConnection -ComputerName "8.8.8.8" -Port 53 -InformationLevel Quiet -WarningAction SilentlyContinue
        return $true
    } catch {
        return $false
    }
}

function Invoke-WithRetry {
    param(
        [scriptblock]$ScriptBlock,
        [int]$MaxRetries = 3,
        [int]$DelaySeconds = 2,
        [string]$Operation = "operation"
    )

    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            return & $ScriptBlock
        } catch {
            if ($i -eq $MaxRetries) {
                Write-Error "Failed $Operation after $MaxRetries attempts: $($_.Exception.Message)"
                throw
            } else {
                Write-Warning "Attempt $i failed for $Operation. Retrying in $DelaySeconds seconds..."
                Start-Sleep -Seconds $DelaySeconds
                $DelaySeconds *= 2  # Exponential backoff
            }
        }
    }
}

function Get-HardwareCapabilities {
    $hardware = @{
        HasNvidiaGPU = $false
        HasAMDGPU = $false
        HasIntelGPU = $false
        CudaSupported = $false
        DirectMLSupported = $false
        RecommendedTorch = "cpu"
        ProcessorName = ""
        TotalRAM_GB = 0
        AvailableRAM_GB = 0
    }

    try {
        # Get system info
        $computerInfo = Get-ComputerInfo -Property @("CsProcessors", "TotalPhysicalMemory") -ErrorAction SilentlyContinue
        if ($computerInfo) {
            $hardware.ProcessorName = $computerInfo.CsProcessors[0].Name
            $hardware.TotalRAM_GB = [math]::Round($computerInfo.TotalPhysicalMemory / 1GB, 1)
        }

        # Get available memory
        $memory = Get-WmiObject -Class Win32_OperatingSystem
        $hardware.AvailableRAM_GB = [math]::Round($memory.FreePhysicalMemory / 1MB, 1)

        # Check for GPUs
        $gpus = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -notmatch "Microsoft|Remote|Virtual" }
        
        foreach ($gpu in $gpus) {
            $gpuName = $gpu.Name.ToLower()
            if ($gpuName -match "nvidia|geforce|rtx|gtx|quadro|tesla") {
                $hardware.HasNvidiaGPU = $true
                $hardware.CudaSupported = $true
                $hardware.RecommendedTorch = "cuda"
            } elseif ($gpuName -match "amd|radeon") {
                $hardware.HasAMDGPU = $true
                $hardware.DirectMLSupported = $true
                if ($hardware.RecommendedTorch -eq "cpu") {
                    $hardware.RecommendedTorch = "directml"
                }
            } elseif ($gpuName -match "intel") {
                $hardware.HasIntelGPU = $true
                $hardware.DirectMLSupported = $true
                if ($hardware.RecommendedTorch -eq "cpu") {
                    $hardware.RecommendedTorch = "directml"
                }
            }
        }

        Write-Info "Hardware Detection Results:"
        Write-Info "  Processor: $($hardware.ProcessorName)"
        Write-Info "  Total RAM: $($hardware.TotalRAM_GB) GB"
        Write-Info "  Available RAM: $($hardware.AvailableRAM_GB) GB"
        if ($hardware.HasNvidiaGPU) { Write-Info "  NVIDIA GPU detected (CUDA recommended)" }
        if ($hardware.HasAMDGPU) { Write-Info "  AMD GPU detected (DirectML supported)" }
        if ($hardware.HasIntelGPU) { Write-Info "  Intel GPU detected (DirectML supported)" }
        Write-Info "  Recommended PyTorch: $($hardware.RecommendedTorch)"

    } catch {
        Write-Warning "Hardware detection failed: $($_.Exception.Message)"
        Write-Warning "Will default to CPU-only installation"
    }

    return $hardware
}

function Test-PythonVersion {
    try {
        $version = & python --version 2>$null
        if ($version -match "Python (\d+\.\d+\.\d+)") {
            $pythonVersion = [Version]$matches[1]
            if ($pythonVersion -ge $MIN_PYTHON_VERSION -and $pythonVersion.Major -eq 3 -and $pythonVersion.Minor -eq 11) {
                Write-Success "Python $pythonVersion detected (compatible)"
                return $true
            } else {
                Write-Warning "Python $pythonVersion detected (requires Python 3.11.x)"
                return $false
            }
        }
    } catch {
        # Python not found
    }

    return $false
}

function Install-SystemComponent {
    param(
        [string]$PackageId,
        [string]$DisplayName,
        [switch]$Essential
    )

    Write-Step "Installing $DisplayName"
    
    # Check if already installed
    try {
        $installed = winget list --id $PackageId 2>$null
        if ($LASTEXITCODE -eq 0 -and $installed -match $PackageId) {
            Write-Success "$DisplayName is already installed"
            return $true
        }
    } catch {
        # Continue with installation
    }

    # Install with retry logic
    try {
        Invoke-WithRetry -Operation "Installing $DisplayName" -ScriptBlock {
            $result = winget install --id $PackageId --accept-source-agreements --accept-package-agreements --silent 2>$null
            if ($LASTEXITCODE -ne 0) {
                throw "winget install failed with exit code $LASTEXITCODE"
            }
        }
        
        Write-Success "$DisplayName installed successfully"
        return $true
        
    } catch {
        if ($Essential) {
            Write-Error "Essential component $DisplayName failed to install: $($_.Exception.Message)"
            throw
        } else {
            Write-Warning "$DisplayName installation failed but is not critical: $($_.Exception.Message)"
            return $false
        }
    }
}

function Install-PythonPackages {
    param([string]$RequirementsFile)

    Write-Step "Installing Python packages from $RequirementsFile"

    if (!(Test-Path $RequirementsFile)) {
        Write-Error "Requirements file not found: $RequirementsFile"
        return $false
    }

    try {
        # Upgrade pip first
        Invoke-WithRetry -Operation "Upgrading pip" -ScriptBlock {
            & python -m pip install --upgrade pip
            if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
        }

        # Install packages with retry logic
        Invoke-WithRetry -Operation "Installing Python packages" -ScriptBlock {
            & python -m pip install -r $RequirementsFile
            if ($LASTEXITCODE -ne 0) { throw "Package installation failed" }
        }

        Write-Success "Python packages installed successfully"
        return $true

    } catch {
        Write-Error "Python package installation failed: $($_.Exception.Message)"
        return $false
    }
}

function Install-PyTorch {
    param([hashtable]$Hardware)

    Write-Step "Installing PyTorch with hardware optimization"

    if ($CpuOnly) {
        $torchCommand = "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        Write-Info "Installing CPU-only PyTorch (forced)"
    } else {
        switch ($Hardware.RecommendedTorch) {
            "cuda" {
                $torchCommand = "python -m pip install torch --index-url https://download.pytorch.org/whl/cu118"
                Write-Info "Installing CUDA PyTorch for NVIDIA GPU"
            }
            "directml" {
                $torchCommand = "python -m pip install torch-directml"
                Write-Info "Installing DirectML PyTorch for AMD/Intel GPU"
            }
            default {
                $torchCommand = "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
                Write-Info "Installing CPU-only PyTorch"
            }
        }
    }

    try {
        Invoke-WithRetry -Operation "Installing PyTorch" -MaxRetries 2 -ScriptBlock {
            Invoke-Expression $torchCommand
            if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
        }

        Write-Success "PyTorch installed successfully"
        return $true

    } catch {
        Write-Warning "Hardware-optimized PyTorch failed, falling back to CPU-only"
        try {
            Invoke-WithRetry -Operation "Installing CPU PyTorch fallback" -ScriptBlock {
                & python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
                if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed" }
            }
            Write-Success "CPU-only PyTorch installed as fallback"
            return $true
        } catch {
            Write-Error "PyTorch installation completely failed: $($_.Exception.Message)"
            return $false
        }
    }
}

function Test-Installation {
    Write-Step "Validating installation"

    $tests = @(
        @{ Name = "Python"; Command = "python --version" },
        @{ Name = "Pip"; Command = "python -m pip --version" },
        @{ Name = "Torch"; Command = "python -c `"import torch; print(f'PyTorch {torch.__version__}')`"" },
        @{ Name = "Whisper"; Command = "python -c `"import whisper; print('Whisper available')`"" },
        @{ Name = "GUI Script"; Command = "python gui_transcribe.py --help" }
    )

    $allPassed = $true
    foreach ($test in $tests) {
        try {
            $result = Invoke-Expression $test.Command 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "$($test.Name): OK"
            } else {
                Write-Warning "$($test.Name): Failed (exit code $LASTEXITCODE)"
                $allPassed = $false
            }
        } catch {
            Write-Warning "$($test.Name): Error - $($_.Exception.Message)"
            $allPassed = $false
        }
    }

    return $allPassed
}

function Start-Application {
    Write-Step "Launching Audio Processor GUI"

    try {
        # Try to launch the GUI
        Start-Process -FilePath "python" -ArgumentList "gui_transcribe.py" -WorkingDirectory $PROJECT_DIR -WindowStyle Normal
        Write-Success "GUI application launched successfully"
        Write-Info "The Audio Processor GUI should now be visible"
        return $true
    } catch {
        Write-Error "Failed to launch GUI: $($_.Exception.Message)"
        Write-Info "You can manually run: python gui_transcribe.py"
        return $false
    }
}

# ============================================================================
# Main Installation Logic
# ============================================================================

function Main {
    Write-Header "Audio Processor Alpha v$SCRIPT_VERSION - Robust Installation"
    Write-Info "Starting streamlined installation for Windows x64 systems..."

    # Preliminary checks
    Write-Step "Performing system checks"

    if (!(Test-AdminRights)) {
        Write-Warning "Not running as Administrator"
        Write-Info "Some system components may fail to install. For best results, run as Administrator."
    } else {
        Write-Success "Running with Administrator privileges"
    }

    if (!(Test-InternetConnection)) {
        Write-Error "No internet connection detected. Installation requires internet access."
        exit 1
    }
    Write-Success "Internet connection verified"

    # Hardware detection
    $hardware = Get-HardwareCapabilities

    # System component installation
    if (-not $SkipSystemComponents) {
        Write-Header "Installing System Components"

        # Set execution policy
        try {
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
            Write-Success "PowerShell execution policy configured"
        } catch {
            Write-Warning "Could not set execution policy: $($_.Exception.Message)"
        }

        # Install essential components
        Install-SystemComponent -PackageId "Python.Python.3.11" -DisplayName "Python 3.11" -Essential
        Install-SystemComponent -PackageId "Git.Git" -DisplayName "Git" -Essential  
        Install-SystemComponent -PackageId "Gyan.FFmpeg" -DisplayName "FFmpeg" -Essential

        # Optional components
        Install-SystemComponent -PackageId "Microsoft.VCRedist.2015+.x64" -DisplayName "Visual C++ Redistributables"

        # Refresh environment
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        # Verify Python
        if (!(Test-PythonVersion)) {
            Write-Error "Python 3.11 installation verification failed. Please restart and try again."
            exit 1
        }
    } else {
        Write-Info "Skipping system component installation"
        if (!(Test-PythonVersion)) {
            Write-Error "Python 3.11 is required but not found. Install Python 3.11 first."
            exit 1
        }
    }

    # Project setup
    Write-Header "Setting Up Project"

    # Create project directory
    if (!(Test-Path $PROJECT_DIR)) {
        New-Item -ItemType Directory -Path $PROJECT_DIR -Force | Out-Null
        Write-Success "Created project directory: $PROJECT_DIR"
    }

    # Clone or update repository
    Push-Location $PROJECT_DIR
    try {
        if (Test-Path $REPO_NAME -PathType Container) {
            if ($ForceReinstall) {
                Remove-Item -Path $REPO_NAME -Recurse -Force
                Write-Info "Removed existing repository for clean installation"
            } else {
                Push-Location $REPO_NAME
                try {
                    & git pull 2>$null
                    Write-Success "Repository updated"
                } catch {
                    Write-Warning "Git pull failed, using existing code"
                } finally {
                    Pop-Location
                }
            }
        }

        if (!(Test-Path $REPO_NAME -PathType Container)) {
            Invoke-WithRetry -Operation "Cloning repository" -ScriptBlock {
                & git clone $REPO_URL
                if ($LASTEXITCODE -ne 0) { throw "Git clone failed" }
            }
            Write-Success "Repository cloned successfully"
        }

        # Move into repository
        Push-Location $REPO_NAME

        # Python environment setup
        Write-Header "Setting Up Python Environment"

        # Create virtual environment
        if (!(Test-Path $VENV_PATH) -or $ForceReinstall) {
            if (Test-Path $VENV_PATH) {
                Remove-Item -Path $VENV_PATH -Recurse -Force
            }
            
            & python -m venv .venv
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to create virtual environment"
                exit 1
            }
            Write-Success "Virtual environment created"
        }

        # Activate virtual environment
        $activateScript = Join-Path ".venv" "Scripts" "Activate.ps1"
        if (Test-Path $activateScript) {
            & $activateScript
            Write-Success "Virtual environment activated"
        } else {
            Write-Error "Virtual environment activation script not found"
            exit 1
        }

        # Install Python packages
        Write-Header "Installing Python Dependencies"

        # Use minimal requirements if available, otherwise fall back to original
        $reqFile = if (Test-Path "requirements-minimal.txt") { "requirements-minimal.txt" } else { "requirements.txt" }
        Write-Info "Using requirements file: $reqFile"

        if (!(Install-PythonPackages -RequirementsFile $reqFile)) {
            Write-Error "Python package installation failed"
            exit 1
        }

        # Install PyTorch
        Write-Header "Installing PyTorch"
        if (!(Install-PyTorch -Hardware $hardware)) {
            Write-Error "PyTorch installation failed"
            exit 1
        }

        # Model preloading
        if (-not $SkipModelPreload) {
            Write-Header "Preloading AI Models"
            if (Test-Path "preload_models.py") {
                try {
                    Invoke-WithRetry -Operation "Preloading models" -MaxRetries 2 -ScriptBlock {
                        & python preload_models.py
                        if ($LASTEXITCODE -ne 0) { throw "Model preloading failed" }
                    }
                    Write-Success "Models preloaded successfully"
                } catch {
                    Write-Warning "Model preloading failed but continuing: $($_.Exception.Message)"
                    Write-Info "Models will be downloaded on first use"
                }
            } else {
                Write-Warning "Model preloading script not found, skipping"
            }
        } else {
            Write-Info "Skipping model preloading (models will download on first use)"
        }

        # Installation validation
        Write-Header "Validating Installation"
        if (Test-Installation) {
            Write-Success "All installation tests passed"
        } else {
            Write-Warning "Some validation tests failed - see details above"
        }

        # Launch application
        Write-Header "Launching Application"
        Start-Application

        # Installation complete
        Write-Header "Installation Complete!"
        Write-Success "Audio Processor Alpha Version is ready to use"
        Write-Info ""
        Write-Info "Installation Summary:"
        Write-Info "  Project Location: $PROJECT_DIR\$REPO_NAME"
        Write-Info "  Python Environment: Activated (.venv)"
        Write-Info "  PyTorch: $($hardware.RecommendedTorch.ToUpper()) optimized"
        Write-Info "  Launch Command: python gui_transcribe.py"
        Write-Info ""
        
        if ($hardware.TotalRAM_GB -lt 8) {
            Write-Warning "System has less than 8GB RAM - consider using smaller Whisper models for better performance"
        }

    } finally {
        Pop-Location  # Exit repository directory
        Pop-Location  # Exit project directory
    }
}

# ============================================================================
# Script Entry Point
# ============================================================================

try {
    Main
    Write-Host "`nInstallation completed successfully! 🎉" -ForegroundColor Green
} catch {
    Write-Host "`nInstallation failed! ❌" -ForegroundColor Red
    Write-Error "Error: $($_.Exception.Message)"
    Write-Host "Stack trace:" -ForegroundColor Gray
    Write-Host $_.ScriptStackTrace -ForegroundColor Gray
    exit 1
}