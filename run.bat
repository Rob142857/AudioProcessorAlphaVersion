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

REM Offer an optional guided PyTorch install
echo.
echo Optional: install a recommended PyTorch build for this machine.
echo   1) Skip PyTorch install (you'll install manually later)
echo   2) Install CPU-only PyTorch (most compatible)
echo   3) Install CUDA PyTorch (for NVIDIA GPU acceleration)
echo   4) Install CPU + DirectML (for AMD/Intel/NVIDIA GPU acceleration)
set /p PTCHOICE="Choose an option [1/2/3/4] (default 1): "
if "%PTCHOICE%"=="" set PTCHOICE=1

if "%PTCHOICE%"=="2" (
  echo Installing CPU-only PyTorch wheel...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo.
    echo ===== PYTORCH INSTALLATION FAILED =====
    echo Check your internet connection and try again.
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
  echo Installing CUDA PyTorch for NVIDIA GPU acceleration...
  python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  if errorlevel 1 (
    echo.
    echo ===== CUDA PYTORCH INSTALLATION FAILED =====
    echo This may happen if you don't have an NVIDIA GPU or drivers.
    echo Falling back to CPU-only PyTorch...
    python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  ) else (
    echo.
    echo ===== CUDA PYTORCH INSTALLED SUCCESSFULLY =====
    echo Your NVIDIA GPU should now be available for acceleration.
  )
  echo Preloading Whisper medium model...
  python preload_models.py
  if errorlevel 1 (
    echo Model preloading failed. Check PyTorch installation above.
  )
  goto :launch
)

if "%PTCHOICE%"=="4" (
  echo Installing CPU-only PyTorch wheel and torch-directml...
  python -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo.
    echo ===== PYTORCH INSTALLATION FAILED =====
    echo Check your internet connection and try again.
    echo.
    pause
    goto :launch
  )
  echo Installing torch-directml for DirectML GPU acceleration...
  python -m pip install --no-warn-script-location torch-directml
  if errorlevel 1 (
    echo torch-directml installation failed. Continuing with CPU-only support.
  ) else (
    echo.
    echo ===== DIRECTML INSTALLED SUCCESSFULLY =====
    echo DirectML GPU acceleration should now be available.
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
echo.
echo Manual PyTorch installation options:
echo   CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
echo   CUDA GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo   DirectML: pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install torch-directml

:launch
echo Launching GUI...
python gui_transcribe.py
pause
