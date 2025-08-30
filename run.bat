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

REM Offer an optional guided PyTorch install
echo.
echo 🔍 Detecting hardware and installing optimal PyTorch build...
echo.

REM Automatic hardware detection and PyTorch installation
python -c "
import subprocess
import sys
import os

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, '', str(e)

def detect_cuda():
    try:
        success, stdout, stderr = run_command('nvidia-smi --query-gpu=name --format=csv,noheader,nounits')
        if success and stdout.strip():
            gpu_name = stdout.strip().split('\n')[0]
            print(f'🎯 CUDA GPU detected: {gpu_name}')
            return True, gpu_name
    except:
        pass
    return False, None

def detect_directml():
    try:
        # Check for AMD/Intel GPUs that support DirectML
        success, stdout, stderr = run_command('wmic path win32_videocontroller get name')
        if success:
            gpu_info = stdout.lower()
            if 'amd' in gpu_info or 'radeon' in gpu_info or 'intel' in gpu_info or 'arc' in gpu_info:
                print('🎯 DirectML-compatible GPU detected')
                return True
    except:
        pass
    return False, None

print('🔍 Scanning for GPU acceleration options...')

# Priority: CUDA > DirectML > CPU
cuda_available, cuda_gpu = detect_cuda()
directml_available, _ = detect_directml()

if cuda_available:
    print(f'✅ Installing CUDA PyTorch for {cuda_gpu}...')
    success, stdout, stderr = run_command('python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    if success:
        print('✅ CUDA PyTorch installed successfully!')
        print('🚀 Your NVIDIA GPU will accelerate transcription by 5-15x')
    else:
        print('⚠️  CUDA PyTorch installation failed, falling back to CPU')
        run_command('python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu')
        
elif directml_available:
    print('✅ Installing DirectML PyTorch for AMD/Intel GPU...')
    success1, _, _ = run_command('python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu')
    success2, _, _ = run_command('python -m pip install --no-warn-script-location torch-directml')
    if success1 and success2:
        print('✅ DirectML PyTorch installed successfully!')
        print('🚀 Your AMD/Intel GPU will accelerate transcription')
    else:
        print('⚠️  DirectML installation failed, using CPU-only')
        
else:
    print('📊 No GPU acceleration available, installing CPU-only PyTorch...')
    success, _, _ = run_command('python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu')
    if success:
        print('✅ CPU PyTorch installed successfully!')
        print('⚡ CPU processing ready (works on any computer)')

print()
print('📥 Downloading Whisper Large model (3GB)...')
print('   This is required for best transcription quality')
print('   Download may take 5-15 minutes depending on your internet speed...')
success, _, _ = run_command('python preload_models.py')
if success:
    print('✅ Large AI model downloaded and cached successfully!')
    print('   Ready for high-quality transcription')
else:
    print('⚠️  Model download failed - will download on first use')
    print('   This will delay your first transcription')
"

:launch
echo Launching GUI...
python gui_transcribe.py
pause
