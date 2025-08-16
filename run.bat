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
  echo.
  echo ===== ARM64 DETECTED =====
  echo NOTE: PyTorch does not officially support ARM64 Windows.
  echo Options 2 and 3 use x64 emulation - this may work but could be slower.
  echo For best results, consider installing x64 Python instead of ARM64 Python.
  echo.
)

REM Offer an optional guided PyTorch install for Surface-style Windows machines
echo.
echo Optional: install a recommended PyTorch build for this machine.
echo   1) Skip PyTorch install (you'll install manually later)
echo   2) Install CPU-only PyTorch wheel (works via x64 emulation on ARM64)
echo   3) Install CPU PyTorch + torch-directml (x64 emulation, may not work on ARM64)
set /p PTCHOICE="Choose an option [1/2/3] (default 1): "
if "%PTCHOICE%"=="" set PTCHOICE=1

if "%PTCHOICE%"=="2" (
  echo Installing CPU-only PyTorch wheel...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo.
    echo ===== PYTORCH INSTALLATION FAILED =====
    echo This likely means you're on ARM64 and x64 emulation didn't work.
    echo Try installing x64 Python from python.org instead of ARM64 Python.
    echo.
    pause
    goto :launch
  )
  echo Preloading Whisper medium model...
  python preload_models.py
  if errorlevel 1 (
    echo Model preloading failed. Check PyTorch installation above.
  )
  goto :launch
)

if "%PTCHOICE%"=="3" (
  echo Installing CPU-only PyTorch wheel and torch-directml...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo.
    echo ===== PYTORCH INSTALLATION FAILED =====
    echo This likely means you're on ARM64 and x64 emulation didn't work.
    echo Try installing x64 Python from python.org instead of ARM64 Python.
    echo.
    pause
    goto :launch
  )
  echo Installing torch-directml...
  python -m pip install --no-warn-script-location torch-directml
  if errorlevel 1 (
    echo torch-directml installation failed. This is expected on ARM64. Continuing without DirectML support.
  )
  echo Preloading Whisper medium model...
  python preload_models.py
  if errorlevel 1 (
    echo Model preloading failed. Check PyTorch installation above.
  )
  goto :launch
)

echo Skipping PyTorch install. You can install a custom wheel later.
echo NOTE: You must install PyTorch and run "python preload_models.py" before using the application.
if /i "%PROC_ARCH%"=="ARM64" (
  echo For ARM64: Consider installing x64 Python, then use: pip install torch --index-url https://download.pytorch.org/whl/cpu
) else (
  echo For x64: pip install torch --index-url https://download.pytorch.org/whl/cpu
)

:launch
echo Launching GUI...
python gui_transcribe.py
pause
