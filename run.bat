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
python -m pip install -r requirements.txt

REM Offer an optional guided PyTorch install for Surface-style Windows machines
echo.
echo Optional: install a recommended PyTorch build for this machine.
echo   1) Skip (recommended if you will install a specific wheel later)
echo   2) Install CPU-only PyTorch wheel (safe default)
echo   3) Install CPU PyTorch + torch-directml (attempt DirectML support)
set /p PTCHOICE="Choose an option [1/2/3] (default 1): "
if "%PTCHOICE%"=="" set PTCHOICE=1

if "%PTCHOICE%"=="2" (
  echo Installing CPU-only PyTorch wheel...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  echo Preloading Whisper medium model...
  python preload_models.py
  goto :launch
)

if "%PTCHOICE%"=="3" (
  echo Installing CPU-only PyTorch wheel and torch-directml...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  echo Installing torch-directml...
  python -m pip install --no-warn-script-location torch-directml
  echo Preloading Whisper medium model...
  python preload_models.py
  goto :launch
)

echo Skipping PyTorch install. You can install a custom wheel later.
echo Note: Run "python preload_models.py" after installing PyTorch to preload models.

:launch
echo Launching GUI...
python gui_transcribe.py
pause
