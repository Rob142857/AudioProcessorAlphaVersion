@echo off
REM Simple bootstrap for WhisperCPP version
if not exist ".venv_whispercpp\Scripts\Activate.ps1" (
  echo Creating WhisperCPP virtual environment...
  python -m venv .venv_whispercpp
)

echo Activating WhisperCPP virtual environment...
call .venv_whispercpp\Scripts\Activate.bat

echo Checking WhisperCPP setup...
if not exist "whisper-cli.exe" (
  echo ❌ whisper-cli.exe not found!
  echo Please run install.bat and choose option 2 for WhisperCPP setup.
  pause
  exit /b 1
)

if not exist "models\ggml-large-v3-turbo.bin" (
  echo ❌ Model file not found!
  echo Please run install.bat and choose option 2 for WhisperCPP setup.
  pause
  exit /b 1
)

echo ✅ WhisperCPP environment ready
echo Launching WhisperCPP GUI...
python gui_transcribe_whispercpp.py
pause
