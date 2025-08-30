@echo off
REM Simple bootstrap: create venv if missing, activate and launch GUI
if not exist ".venv\Scripts\Activate.ps1" (
  echo Creating virtual environment...
  python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\Activate.bat

echo Installing requirements (skips torch so you can pick GPU build)...
python -m pip install --upgrade pip

REM Try to install all requirements, but handle webrtcvad failure gracefully
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ⚠️  Some packages failed to install (likely webrtcvad build tools issue)
  echo Trying to install packages individually, skipping problematic ones...
  echo.
  
  REM Install packages one by one, skipping webrtcvad if it fails
  python -m pip install "openai-whisper==20250625"
  python -m pip install "deepmultilingualpunctuation==1.0.1"
  python -m pip install "moviepy>=2.1.0,<3.0.0"
  python -m pip install "imageio-ffmpeg>=0.6.0,<0.7.0"
  python -m pip install "python-docx>=1.2.0,<2.0.0"
  python -m pip install "psutil>=7.0.0,<8.0.0"
  python -m pip install "tqdm>=4.60.0,<5.0.0"
  
  echo Attempting webrtcvad installation...
  python -m pip install "webrtcvad==2.0.10"
  if errorlevel 1 (
    echo.
    echo ❌ webrtcvad failed to install (requires Visual Studio Build Tools)
    echo ℹ️  Voice Activity Detection will be disabled, but transcription will still work
    echo ℹ️  To fix this: Install Visual Studio Build Tools or run install.bat again
    echo.
  ) else (
    echo ✅ webrtcvad installed successfully
  )
)

@echo off
REM Simple bootstrap: create venv if missing, activate and launch GUI
if not exist ".venv\Scripts\Activate.ps1" (
  echo Creating virtual environment...
  python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\Activate.bat

echo Installing requirements (skips torch so you can pick GPU build)...
python -m pip install --upgrade pip

REM Try to install all requirements, but handle webrtcvad failure gracefully
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ⚠️  Some packages failed to install (likely webrtcvad build tools issue)
  echo Trying to install packages individually, skipping problematic ones...
  echo.
  
  REM Install packages one by one, skipping webrtcvad if it fails
  python -m pip install "openai-whisper==20250625"
  python -m pip install "deepmultilingualpunctuation==1.0.1"
  python -m pip install "moviepy>=2.1.0,<3.0.0"
  python -m pip install "imageio-ffmpeg>=0.6.0,<0.7.0"
  python -m pip install "python-docx>=1.2.0,<2.0.0"
  python -m pip install "psutil>=7.0.0,<8.0.0"
  python -m pip install "tqdm>=4.60.0,<5.0.0"
  
  echo Attempting webrtcvad installation...
  python -m pip install "webrtcvad==2.0.10"
  if errorlevel 1 (
    echo.
    echo ❌ webrtcvad failed to install (requires Visual Studio Build Tools)
    echo ℹ️  Voice Activity Detection will be disabled, but transcription will still work
    echo ℹ️  To fix this: Install Visual Studio Build Tools or run install.bat again
    echo.
  ) else (
    echo ✅ webrtcvad installed successfully
  )
)

REM Automated hardware detection and PyTorch installation
echo.
echo 🔍 Detecting hardware and installing optimal PyTorch build...
echo.

python -c "
import subprocess
import sys
import platform

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except:
        return False, '', ''

def detect_nvidia_gpu():
    # Check for NVIDIA GPU using nvidia-smi
    success, stdout, stderr = run_command('nvidia-smi --query-gpu=name --format=csv,noheader,nounits')
    if success and stdout.strip():
        gpu_name = stdout.strip().split('\n')[0]
        print(f'✅ NVIDIA GPU detected: {gpu_name}')
        return True, gpu_name
    return False, None

def detect_amd_gpu():
    # Check for AMD GPU using dxdiag or wmic
    success, stdout, stderr = run_command('wmic path win32_VideoController get name')
    if success:
        for line in stdout.split('\n'):
            line = line.strip()
            if 'amd' in line.lower() or 'radeon' in line.lower():
                print(f'✅ AMD GPU detected: {line}')
                return True, line
    return False, None

def detect_intel_gpu():
    # Check for Intel GPU
    success, stdout, stderr = run_command('wmic path win32_VideoController get name')
    if success:
        for line in stdout.split('\n'):
            line = line.strip()
            if 'intel' in line.lower():
                print(f'✅ Intel GPU detected: {line}')
                return True, line
    return False, None

def install_pytorch(build_type, description):
    print(f'📦 Installing {description}...')
    if build_type == 'cuda':
        cmd = 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
    elif build_type == 'directml':
        cmd = 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
        success, _, _ = run_command(cmd)
        if success:
            cmd = 'python -m pip install torch-directml'
    else:  # cpu
        cmd = 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
    
    success, stdout, stderr = run_command(cmd)
    if success:
        print(f'✅ {description} installed successfully!')
        return True
    else:
        print(f'❌ {description} installation failed')
        print(f'Error: {stderr}')
        return False

# Main detection and installation logic
print('🔍 Scanning hardware...')
print()

# Check for NVIDIA GPU first (best performance)
nvidia_detected, nvidia_name = detect_nvidia_gpu()
if nvidia_detected:
    if install_pytorch('cuda', 'CUDA PyTorch for NVIDIA GPU'):
        print('🚀 CUDA acceleration ready!')
        sys.exit(0)

# Check for AMD GPU
amd_detected, amd_name = detect_amd_gpu()
if amd_detected:
    if install_pytorch('directml', 'DirectML PyTorch for AMD GPU'):
        print('🚀 DirectML acceleration ready!')
        sys.exit(0)

# Check for Intel GPU
intel_detected, intel_name = detect_intel_gpu()
if intel_detected:
    if install_pytorch('directml', 'DirectML PyTorch for Intel GPU'):
        print('🚀 DirectML acceleration ready!')
        sys.exit(0)

# Fallback to CPU-only
print('💻 No compatible GPU detected, installing CPU-only PyTorch...')
if install_pytorch('cpu', 'CPU-only PyTorch'):
    print('✅ CPU PyTorch ready!')
    print('ℹ️  CPU-only mode will work but may be slower for large files')
else:
    print('❌ All PyTorch installation attempts failed!')
    print('ℹ️  You can manually install PyTorch later with:')
    print('   pip install torch --index-url https://download.pytorch.org/whl/cpu')
"

if errorlevel 1 (
  echo.
  echo ===== PYTORCH INSTALLATION FAILED =====
  echo Manual installation may be required.
  echo Try: pip install torch --index-url https://download.pytorch.org/whl/cpu
  echo.
) else (
  echo.
  echo ===== PYTORCH INSTALLED SUCCESSFULLY =====
  echo Hardware acceleration should now be available.
  echo.
)

echo Preloading Whisper large model...
python preload_models.py
if errorlevel 1 (
  echo Model preloading failed. Check PyTorch installation above.
)
goto :launch

:launch
echo Launching GUI...
python gui_transcribe.py
pause
echo Preloading Whisper large model...
python preload_models.py
if errorlevel 1 (
  echo Model preloading failed. Check PyTorch installation above.
)
goto :launch

:launch
echo Launching GUI...
python gui_transcribe.py
pause
