@echo off
REM Simple bootstrap: create venv if missing, install deps, attempt hardware-optimized torch, preload model, launch GUI

if not exist ".venv\Scripts\Activate.ps1" (
  echo Creating virtual environment...
  python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\Activate.bat

echo Installing base Python packages (will skip torch so you can choose the right build)...
python -m pip install --upgrade pip

REM Attempt to install all requirements; if webrtcvad fails, continue without it
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ⚠️  Some packages failed to install (often webrtcvad). Installing others individually...
  echo.
  python -m pip install "openai-whisper==20250625"
  python -m pip install "deepmultilingualpunctuation==1.0.1"
  python -m pip install "moviepy>=2.1.0,<3.0.0"
  python -m pip install "imageio-ffmpeg>=0.6.0,<0.7.0"
  python -m pip install "python-docx>=1.2.0,<2.0.0"
  python -m pip install "psutil>=7.0.0,<8.0.0"
  python -m pip install "tqdm>=4.60.0,<5.0.0"
  echo Attempting webrtcvad installation...
  python -m pip install "webrtcvad==2.0.10"
)

echo.
echo 🔍 If you have an NVIDIA/AMD/Intel GPU, install a matching PyTorch build now (see README). Otherwise CPU-only works.
echo.

echo Preloading Whisper model (turbo preferred)...
python preload_models.py
if errorlevel 1 (
  echo Model preloading failed. You may need to install torch (see README) before preloading.
)

echo Launching GUI...
python gui_transcribe.py --gui
pause
