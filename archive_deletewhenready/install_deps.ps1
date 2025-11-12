# Speech to Text Transcription Tool v1.1 Dependency Installer
# Handles Python packages and PyTorch installation

Write-Host "=== Installing Python Dependencies ===" -ForegroundColor Green

# Function to run Python command and check result
function Invoke-PythonCommand {
    param($Command, $Description)
    Write-Host "Running: $Description" -ForegroundColor Yellow
    try {
        $result = & python $Command.Split() 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $Description completed" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ $Description failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "Output: $result" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ $Description failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Create virtual environment if it doesn't exist
if (!(Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    if (!(Invoke-PythonCommand "-m venv .venv" "Create virtual environment")) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Invoke-PythonCommand "-m pip install --upgrade pip" "Upgrade pip"

# Install core packages
Write-Host "Installing core Python packages..." -ForegroundColor Yellow
$packages = @(
    'openai-whisper==20250625',
    'deepmultilingualpunctuation==1.0.1',
    'moviepy>=2.1.0,<3.0.0',
    'imageio-ffmpeg>=0.6.0,<0.7.0',
    'python-docx>=1.2.0,<2.0.0',
    'psutil>=7.0.0,<8.0.0',
    'tqdm>=4.60.0,<5.0.0'
)

foreach ($pkg in $packages) {
    Invoke-PythonCommand "-m pip install $pkg" "Install $pkg"
}

# Install CPU PyTorch (safe default)
Write-Host "Installing CPU PyTorch..." -ForegroundColor Yellow
Invoke-PythonCommand "-m pip install torch --index-url https://download.pytorch.org/whl/cpu" "Install CPU PyTorch"

# Test installation
Write-Host "Testing installation..." -ForegroundColor Yellow
$testResult = Invoke-PythonCommand "-c import sys; import whisper, psutil; import docx; print('Core imports successful')" "Test core imports"

if ($testResult) {
    Write-Host "✓ Installation successful!" -ForegroundColor Green
} else {
    Write-Host "⚠ Some tests failed - installation may be incomplete" -ForegroundColor Yellow
}

# Try to preload model
Write-Host "Attempting to preload Whisper model..." -ForegroundColor Yellow
Invoke-PythonCommand "preload_models.py" "Preload Whisper model"

Write-Host "=== Dependencies Installation Complete! ===" -ForegroundColor Green
Write-Host "You can now run: python gui_transcribe.py --gui" -ForegroundColor Cyan