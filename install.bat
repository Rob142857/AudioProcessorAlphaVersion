@echo off
echo === Speech to Text Transcription Tool v1.0Beta Complete Installer ===
echo Choose your preferred backend:
echo [1] PyTorch Whisper (GPU/CPU, default large-v3-turbo)
echo [2] WhisperCPP (CPU-only, turbo v3 model)
echo.

set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" goto :pytorch
if "%choice%"=="2" goto :whispercpp
echo Invalid choice. Defaulting to PyTorch...
goto :pytorch

:pytorch
echo.
echo === Installing PyTorch Whisper Version (turbo default) ===
echo Installing prerequisites for Windows...

REM Check if we're in the project directory
if not exist "run.bat" (
    echo.
    echo ❌ ERROR: run.bat not found in current directory!
    echo ℹ️  You need to download the project files first.
    echo.
    echo 📥 Downloading project files...
    echo.

    REM Download the project zip
    curl -L https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip -o AudioProcessorAlphaVersion.zip

    REM Extract the zip (requires PowerShell)
    powershell -command "Expand-Archive -Path AudioProcessorAlphaVersion.zip -DestinationPath . -Force"

    REM Move into the extracted directory
    cd AudioProcessorAlphaVersion-main

    echo ✅ Project files downloaded and extracted.
    echo.
)

REM Install Python 3.11 x64 (if not already installed)
echo Checking for Python 3.11...
python --version >nul 2>&1
if errorlevel 1 (
    echo Installing Python 3.11 x64...
    winget install --id Python.Python.3.11 --scope user --force --accept-package-agreements --accept-source-agreements
) else (
    echo ✅ Python already installed
)

REM Install Visual C++ Redistributables (if not already installed)
echo Checking for Visual C++ Redistributables...
if not exist "C:\Windows\System32\vcruntime140.dll" (
    echo Installing Visual C++ Redistributables...
    winget install --id Microsoft.VCRedist.2015+.x64 --force --accept-package-agreements --accept-source-agreements
) else (
    echo ✅ Visual C++ Redistributables already installed
)

REM Install Visual Studio Build Tools (required for compiling packages like webrtcvad)
echo Checking for Visual Studio Build Tools...
vswhere.exe >nul 2>&1
if errorlevel 1 (
    echo Installing Visual Studio Build Tools...
    winget install --id Microsoft.VisualStudio.2022.BuildTools --force --accept-package-agreements --accept-source-agreements
) else (
    echo ✅ Visual Studio Build Tools already installed
)

echo.
echo === Prerequisites Complete! ===
echo Launching Speech to Text Transcription Tool setup...

REM Launch the main setup (which handles venv creation, requirements, and hardware-optimized PyTorch)
call run.bat

echo.
echo === Installation Complete! ===
echo You can now run: python gui_transcribe.py --gui
pause
goto :end

:whispercpp
echo.
echo === Installing WhisperCPP Version ===
echo This version uses the fast turbo v3 model with CPU-only processing.
echo.

REM Check if we're in the project directory
if not exist "run_whispercpp.bat" (
    echo.
    echo ❌ ERROR: run_whispercpp.bat not found in current directory!
    echo ℹ️  You need to download the project files first.
    echo.
    echo 📥 Downloading project files...
    echo.

    REM Download the project zip
    curl -L https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip -o AudioProcessorAlphaVersion.zip

    REM Extract the zip (requires PowerShell)
    powershell -command "Expand-Archive -Path AudioProcessorAlphaVersion.zip -DestinationPath . -Force"

    REM Move into the extracted directory
    cd AudioProcessorAlphaVersion-main

    echo ✅ Project files downloaded and extracted.
    echo.
)

REM Install Python 3.11 x64 (if not already installed)
echo Checking for Python 3.11...
python --version >nul 2>&1
if errorlevel 1 (
    echo Installing Python 3.11 x64...
    winget install --id Python.Python.3.11 --scope user --force --accept-package-agreements --accept-source-agreements
) else (
    echo ✅ Python already installed
)

REM Install Visual C++ Redistributables (if not already installed)
echo Checking for Visual C++ Redistributables...
if not exist "C:\Windows\System32\vcruntime140.dll" (
    echo Installing Visual C++ Redistributables...
    winget install --id Microsoft.VCRedist.2015+.x64 --force --accept-package-agreements --accept-source-agreements
) else (
    echo ✅ Visual C++ Redistributables already installed
)

echo.
echo === Setting up WhisperCPP Environment ===

REM Create virtual environment if it doesn't exist
if not exist ".venv_whispercpp\Scripts\Activate.ps1" (
    echo Creating WhisperCPP virtual environment...
    python -m venv .venv_whispercpp
)

echo Activating WhisperCPP virtual environment...
call .venv_whispercpp\Scripts\Activate.bat

echo Installing WhisperCPP requirements...
python -m pip install --upgrade pip
python -m pip install onnxruntime transformers numpy python-docx psutil tqdm

echo Downloading WhisperCPP turbo v3 model...
if not exist "models" mkdir models
if not exist "models\ggml-large-v3-turbo.bin" (
    echo Downloading ggml-large-v3-turbo.bin model...
    curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin -o models\ggml-large-v3-turbo.bin
) else (
    echo ✅ Model already downloaded
)

echo Downloading WhisperCPP binaries for Windows x64...
if not exist "whispercpp_x64" (
    mkdir whispercpp_x64
    cd whispercpp_x64

    echo Downloading whisper-cli.exe...
    curl -L https://github.com/ggerganov/whisper.cpp/releases/download/v1.5.5/whisper-cli-x64.zip -o whisper-cli-x64.zip
    powershell -command "Expand-Archive -Path whisper-cli-x64.zip -DestinationPath . -Force"

    REM Copy the executable to parent directory
    if exist "whisper-cli.exe" (
        copy whisper-cli.exe ..
    ) else (
        REM Try to find it in subdirectories
        for /r %%i in (whisper-cli.exe) do copy "%%i" ..
    )

    cd ..
)

if exist "whisper-cli.exe" (
    echo ✅ WhisperCPP setup complete!
) else (
    echo ❌ WhisperCPP setup failed - executable not found
    echo ℹ️  You may need to download whisper-cli.exe manually
)

echo.
echo === WhisperCPP Installation Complete! ===
echo You can now run: python gui_transcribe_whispercpp.py
echo Or use: run_whispercpp.bat
pause
goto :end

:end
