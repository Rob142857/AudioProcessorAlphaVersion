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

python detect_hardware.py
if errorlevel 1 (
  echo.
  echo ❌ Hardware detection failed, falling back to CPU-only PyTorch...
  python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
  echo.
  echo ✅ Hardware detection completed successfully!
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
