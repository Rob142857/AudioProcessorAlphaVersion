@echo off
REM One-liner installer for Speech to Text Transcription Tool
REM Supports both ARM64 and x64 architectures

echo === Speech to Text Transcription Tool - Universal Installer ===
echo.

REM Detect architecture
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    echo Detected ARM64 architecture
    goto :arm64_install
) else if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    echo Detected x64 architecture
    goto :x64_install
) else (
    echo Unsupported architecture: %PROCESSOR_ARCHITECTURE%
    echo This installer supports ARM64 and x64 only.
    pause
    exit /b 1
)

:arm64_install
echo.
echo === ARM64 Installation ===
echo This will install the WhisperCPP version optimized for ARM64 with NPU acceleration.
echo.

REM Check if ARM files exist
if exist "whispercpp_arm" (
    echo Found existing ARM64 setup. Using it...
    cd whispercpp_arm

    REM Install Python requirements for ARM
    if not exist ".venv_arm\Scripts\Activate.ps1" (
        echo Creating ARM64 virtual environment...
        python -m venv .venv_arm
    )

    echo Activating ARM64 environment...
    call .venv_arm\Scripts\Activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    echo Downloading ARM64 models...
    python download_ggml_model.py

    echo.
    echo === ARM64 NPU Setup Complete ===
    echo Your ARM64 device is now configured with NPU acceleration!
    echo.
    echo To launch the GUI:
    echo   cd whispercpp_arm
    echo   python gui_transcribe.py
    echo.
    echo The GUI will show "Whisper.cpp ARM64 Transcription (NPU Accelerated)"
    echo.
) else (
    echo ARM64 setup not found in repository.
    echo This should not happen - ARM64 files should be included.
    echo Please ensure you're using the latest version from GitHub.
    pause
    exit /b 1
)
goto :end

:x64_install
echo.
echo === x64 Installation ===
echo Choose your preferred backend:
echo [1] PyTorch Whisper (GPU/CPU, large model)
echo [2] WhisperCPP Turbo v3 (CPU-only, fastest)
echo.

set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" goto :x64_pytorch
if "%choice%"=="2" goto :x64_whispercpp
echo Invalid choice. Defaulting to PyTorch...
goto :x64_pytorch

:x64_pytorch
echo Installing PyTorch version...
call install.bat
goto :end

:x64_whispercpp
echo Installing WhisperCPP version...
REM Simulate choice 2 in install.bat
echo 2 | install.bat
goto :end

:end
echo.
echo === Installation Complete ===
pause
