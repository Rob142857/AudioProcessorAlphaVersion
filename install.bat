@echo off
echo === Speech to Text Transcription Tool v1.0Beta Complete Installer ===
echo Installing prerequisites for Windows...

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

REM Install Git (if not already installed)
echo Checking for Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo Installing Git...
    winget install --id Git.Git --force --accept-package-agreements --accept-source-agreements
) else (
    echo ✅ Git already installed
)

echo.
echo === Prerequisites Complete! ===
echo Launching Speech to Text Transcription Tool setup...

REM Launch the main setup (which handles venv creation, requirements, and hardware-optimized PyTorch)
call run.bat

echo.
echo === Installation Complete! ===
echo You can now run: python gui_transcribe.py
pause
