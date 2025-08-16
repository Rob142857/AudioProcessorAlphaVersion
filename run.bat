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

REM Detect if running on ARM64
for /f "tokens=*" %%i in ('echo %PROCESSOR_ARCHITECTURE%') do set PROC_ARCH=%%i
if /i "%PROC_ARCH%"=="ARM64" (
  echo Detected ARM64 processor - using conda environment is recommended for this platform.
  echo Consider using the conda bootstrap instead: powershell -ExecutionPolicy RemoteSigned -File .\run_conda.ps1
  echo.
)

REM Offer an optional guided PyTorch install for Surface-style Windows machines
echo.
echo Optional: install a recommended PyTorch build for this machine.
echo   1) Skip PyTorch install (you'll install manually later)
echo   2) Install CPU-only PyTorch wheel (safe default for x64)
echo   3) Install CPU PyTorch + torch-directml (attempt DirectML support for x64)
set /p PTCHOICE="Choose an option [1/2/3] (default 1): "
if "%PTCHOICE%"=="" set PTCHOICE=1

if "%PTCHOICE%"=="2" (
  echo Installing CPU-only PyTorch wheel...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo PyTorch installation failed. On ARM64, try the conda approach instead.
    goto :launch
  )
  echo Preloading Whisper medium model...
  python preload_models.py
  if errorlevel 1 (
    echo Model preloading failed. Check PyTorch installation.
  )
  goto :launch
)

if "%PTCHOICE%"=="3" (
  echo Installing CPU-only PyTorch wheel and torch-directml...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo PyTorch installation failed. On ARM64, try the conda approach instead.
    goto :launch
  )
  echo Installing torch-directml...
  python -m pip install --no-warn-script-location torch-directml
  if errorlevel 1 (
    echo torch-directml installation failed. Continuing without DirectML support.
  )
  echo Preloading Whisper medium model...
  python preload_models.py
  if errorlevel 1 (
    echo Model preloading failed. Check PyTorch installation.
  )
  goto :launch
)

echo Skipping PyTorch install. You can install a custom wheel later.
echo NOTE: You must install PyTorch and run "python preload_models.py" before using the application.
echo For ARM64/Surface: Use conda approach - run: powershell -ExecutionPolicy RemoteSigned -File .\run_conda.ps1

:launch
echo Launching GUI...
python gui_transcribe.py
pause
