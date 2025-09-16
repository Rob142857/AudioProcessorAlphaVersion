#Requires -Version 5.1

<#
.SYNOPSIS
    Complete installer for Audio Processor Alpha Version
    Installs all prerequisites, sets up virtual environment, installs dependencies,
    preloads models, and launches the GUI application.

.DESCRIPTION
    This script performs a complete installation of the Audio Processor Alpha Version
    application on Windows systems. It handles:
    - System prerequisite installation (Python, Git, VS Build Tools, etc.)
    - Hardware detection for optimal PyTorch installation
    - Virtual environment setup
    - Python package installation
    - Model preloading
    - GUI application launch

.PARAMETER SkipPrerequisites
    Skip system prerequisite installation (for development/testing)

.PARAMETER SkipModelPreload
    Skip Whisper model preloading

.PARAMETER ForceCpuTorch
    Force CPU-only PyTorch installation regardless of hardware detection

.EXAMPLE
    # Complete installation (recommended)
    irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex

    # Skip prerequisites (if already installed)
    .\install.ps1 -SkipPrerequisites

.NOTES
    - Must be run as Administrator
    - Requires internet connection
    - May take 10-30 minutes depending on hardware and internet speed
#>

param(
    [switch]$SkipPrerequisites,
    [switch]$SkipModelPreload,
    [switch]$ForceCpuTorch
)

# ============================================================================
# Configuration and Constants
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Speed up downloads

# Repository information
$REPO_URL = "https://github.com/Rob142857/AudioProcessorAlphaVersion.git"
$REPO_NAME = "AudioProcessorAlphaVersion"
$SCRIPT_VERSION = "1.0.0"

# Required software versions
$PYTHON_VERSION = "3.11"
$PYTHON_FULL_VERSION = "3.11.9"
$GIT_VERSION = "2.51.0"

# ============================================================================
# Utility Functions
# ============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
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

function Test-Command {
    param([string]$Command)
    try {
        $null = Invoke-Expression "$Command 2>&1"
        return $true
    } catch {
        return $false
    }
}

function Get-HardwareInfo {
    $hardware = @{
        HasNvidiaGPU = $false
        HasAMDGPU = $false
        HasIntelGPU = $false
        CudaVersion = $null
        ProcessorName = ""
    }

    try {
        # Check for NVIDIA GPU
        $nvidiaOutput = wmic path win32_VideoController get name /value 2>$null
        if ($nvidiaOutput -match "NVIDIA|GeForce|RTX|GTX|Quadro|Tesla") {
            $hardware.HasNvidiaGPU = $true
        }

        # Check for AMD GPU
        if ($nvidiaOutput -match "AMD| Radeon") {
            $hardware.HasAMDGPU = $true
        }

        # Check for Intel GPU
        if ($nvidiaOutput -match "Intel") {
            $hardware.HasIntelGPU = $true
        }

        # Get processor info
        $cpuInfo = Get-WmiObject -Class Win32_Processor
        $hardware.ProcessorName = $cpuInfo.Name

        # Try to detect CUDA version (basic check)
        if ($hardware.HasNvidiaGPU) {
            try {
                $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
                if (Test-Path $cudaPath) {
                    $cudaDirs = Get-ChildItem $cudaPath -Directory | Sort-Object Name -Descending
                    if ($cudaDirs) {
                        $hardware.CudaVersion = $cudaDirs[0].Name
                    }
                }
            } catch {
                # CUDA detection failed, will use default
            }
        }
    } catch {
        Write-Warning "Hardware detection failed, will use CPU-only PyTorch: $($_.Exception.Message)"
    }

    return $hardware
}

function Install-Package {
    param(
        [string]$PackageId,
        [string]$PackageName,
        [switch]$SkipIfInstalled
    )

    Write-Info "Installing $PackageName..."

    if ($SkipIfInstalled) {
        $installed = winget list --id $PackageId 2>$null
        if ($LASTEXITCODE -eq 0 -and $installed -match $PackageId) {
            Write-Success "$PackageName is already installed"
            return $true
        }
    }

    $result = winget install --id $PackageId --accept-source-agreements --accept-package-agreements --silent
    if ($LASTEXITCODE -eq 0) {
        Write-Success "$PackageName installed successfully"
        return $true
    } else {
        Write-Error "Failed to install $PackageName (exit code: $LASTEXITCODE)"
        return $false
    }
}

function Test-PythonInstallation {
    try {
        $pythonVersion = & python --version 2>$null
        if ($pythonVersion -match "Python $PYTHON_VERSION") {
            Write-Success "Python $PYTHON_VERSION is available"
            return $true
        }
    } catch {
        # Python not found in PATH
    }

    # Try python3
    try {
        $pythonVersion = & python3 --version 2>$null
        if ($pythonVersion -match "Python $PYTHON_VERSION") {
            Write-Success "Python $PYTHON_VERSION is available (as python3)"
            return $true
        }
    } catch {
        # python3 not found
    }

    return $false
}

function Install-PythonPackages {
    param([string]$RequirementsFile)

    Write-Info "Installing Python packages from $RequirementsFile..."

    if (!(Test-Path $RequirementsFile)) {
        Write-Error "Requirements file not found: $RequirementsFile"
        return $false
    }

    # Install packages excluding torch (we handle that separately)
    $result = & python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to upgrade pip"
        return $false
    }

    # Install packages, excluding torch-related lines
    $requirements = Get-Content $RequirementsFile | Where-Object {
        $_ -notmatch "^#" -and
        $_ -notmatch "^\s*$" -and
        $_ -notmatch "torch|PyTorch"
    }

    foreach ($package in $requirements) {
        Write-Info "Installing $package..."
        $result = & python -m pip install $package
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install $package"
            return $false
        }
    }

    Write-Success "Python packages installed successfully"
    return $true
}

function Install-PyTorch {
    param([hashtable]$Hardware)

    Write-Info "Installing PyTorch for detected hardware..."

    if ($ForceCpuTorch) {
        Write-Info "Forcing CPU-only PyTorch installation"
        $torchCommand = "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    } elseif ($Hardware.HasNvidiaGPU) {
        Write-Info "NVIDIA GPU detected, installing CUDA PyTorch"
        # Use CUDA 11.8 for broad compatibility
        $torchCommand = "python -m pip install torch --index-url https://download.pytorch.org/whl/cu118"
    } elseif ($Hardware.HasAMDGPU -or $Hardware.HasIntelGPU) {
        Write-Info "AMD/Intel GPU detected, installing DirectML PyTorch"
        $torchCommand = "python -m pip install torch-directml"
    } else {
        Write-Info "No GPU detected, installing CPU-only PyTorch"
        $torchCommand = "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    }

    Write-Info "Running: $torchCommand"
    $result = Invoke-Expression $torchCommand
    if ($LASTEXITCODE -eq 0) {
        Write-Success "PyTorch installed successfully"
        return $true
    } else {
        Write-Error "Failed to install PyTorch"
        return $false
    }
}

function Preload-Models {
    Write-Info "Preloading Whisper models..."

    $preloadScript = "preload_models.py"
    if (!(Test-Path $preloadScript)) {
        Write-Error "Model preload script not found: $preloadScript"
        return $false
    }

    $result = & python $preloadScript
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Models preloaded successfully"
        return $true
    } else {
        Write-Error "Failed to preload models"
        return $false
    }
}

function Start-GUIApplication {
    Write-Info "Launching GUI application..."

    $guiScript = "gui_transcribe.py"
    if (!(Test-Path $guiScript)) {
        Write-Error "GUI script not found: $guiScript"
        return $false
    }

    # Launch GUI in background
    Start-Process -FilePath "python" -ArgumentList $guiScript, "--gui" -NoNewWindow

    Write-Success "GUI application launched"
    Write-Info "The application should now be running. Check for the GUI window."
    return $true
}

# ============================================================================
# Main Installation Logic
# ============================================================================

function Main {
    Write-Header "Audio Processor Alpha Version v$SCRIPT_VERSION Complete Installer"
    Write-Info "Starting installation process..."

    # Check if running as administrator
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdmin) {
        Write-Error "This script must be run as Administrator. Please right-click PowerShell and select 'Run as Administrator', then run the command again."
        exit 1
    }

    Write-Success "Running as Administrator"

    # Detect hardware for PyTorch selection
    $hardware = Get-HardwareInfo
    Write-Info "Hardware detected: $($hardware.ProcessorName)"
    if ($hardware.HasNvidiaGPU) { Write-Info "  - NVIDIA GPU detected" }
    if ($hardware.HasAMDGPU) { Write-Info "  - AMD GPU detected" }
    if ($hardware.HasIntelGPU) { Write-Info "  - Intel GPU detected" }

    # Step 1: Install system prerequisites
    if (-not $SkipPrerequisites) {
        Write-Header "Installing System Prerequisites"

        # Set execution policy
        try {
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
            Write-Success "PowerShell execution policy set"
        } catch {
            Write-Warning "Could not set execution policy: $($_.Exception.Message)"
        }

        # Install Python
        if (-not (Install-Package -PackageId "Python.Python.$PYTHON_VERSION" -PackageName "Python $PYTHON_VERSION" -SkipIfInstalled)) {
            Write-Error "Python installation failed. Please install Python $PYTHON_VERSION manually."
            exit 1
        }

        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

        # Verify Python installation
        if (-not (Test-PythonInstallation)) {
            Write-Error "Python $PYTHON_VERSION verification failed. Please restart PowerShell and try again."
            exit 1
        }

        # Install Visual C++ Redistributables
        if (-not (Install-Package -PackageId "Microsoft.VCRedist.2015+.x64" -PackageName "Visual C++ Redistributables" -SkipIfInstalled)) {
            Write-Warning "Visual C++ Redistributables installation failed, but continuing..."
        }

        # Install Git
        if (-not (Install-Package -PackageId "Git.Git" -PackageName "Git" -SkipIfInstalled)) {
            Write-Error "Git installation failed. Please install Git manually."
            exit 1
        }

        # Install ffmpeg (required for audio processing)
        if (-not (Install-Package -PackageId "Gyan.FFmpeg" -PackageName "FFmpeg" -SkipIfInstalled)) {
            Write-Error "FFmpeg installation failed. Audio processing will not work without FFmpeg."
            exit 1
        }

        Write-Success "System prerequisites installed"
    } else {
        Write-Info "Skipping system prerequisite installation"
    }

    # Step 2: Clone/update repository
    Write-Header "Setting up Repository"

    if (Test-Path $REPO_NAME) {
        Write-Info "Repository directory exists, updating..."
        Push-Location $REPO_NAME
        try {
            & git pull
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Git pull failed, but continuing with existing code"
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Info "Cloning repository..."
        & git clone $REPO_URL
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to clone repository"
            exit 1
        }
    }

    # Change to repository directory
    if (!(Test-Path $REPO_NAME)) {
        Write-Error "Repository directory not found after clone/update"
        exit 1
    }

    Push-Location $REPO_NAME
    Write-Success "Repository ready"

    # Step 3: Set up virtual environment
    Write-Header "Setting up Virtual Environment"

    if (Test-Path ".venv") {
        Write-Info "Virtual environment already exists"
    } else {
        Write-Info "Creating virtual environment..."
        & python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create virtual environment"
            Pop-Location
            exit 1
        }
    }

    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    & .\.venv\Scripts\Activate.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to activate virtual environment"
        Pop-Location
        exit 1
    }

    Write-Success "Virtual environment activated"

    # Step 4: Install Python packages
    Write-Header "Installing Python Dependencies"

    if (-not (Install-PythonPackages -RequirementsFile "requirements.txt")) {
        Write-Error "Python package installation failed"
        Pop-Location
        exit 1
    }

    # Step 5: Install PyTorch
    Write-Header "Installing PyTorch"

    if (-not (Install-PyTorch -Hardware $hardware)) {
        Write-Error "PyTorch installation failed"
        Pop-Location
        exit 1
    }

    # Step 6: Preload models
    if (-not $SkipModelPreload) {
        Write-Header "Preloading Models"
        if (-not (Preload-Models)) {
            Write-Warning "Model preloading failed, but continuing..."
        }
    } else {
        Write-Info "Skipping model preloading"
    }

    # Step 7: Launch GUI
    Write-Header "Launching Application"

    if (-not (Start-GUIApplication)) {
        Write-Error "Failed to launch GUI application"
        Pop-Location
        exit 1
    }

    # Installation complete
    Write-Header "Installation Complete!"
    Write-Success "Audio Processor Alpha Version is now installed and running"
    Write-Info "The GUI application should be visible. If not, check the taskbar."
    Write-Info "You can run the application again with: python gui_transcribe.py --gui"

    Pop-Location
}

# ============================================================================
# Script Entry Point
# ============================================================================

try {
    Main
} catch {
    Write-Error "Installation failed with error: $($_.Exception.Message)"
    Write-Error "Stack trace: $($_.ScriptStackTrace)"
    exit 1
}
