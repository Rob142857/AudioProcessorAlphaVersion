@echo off
echo === AudioProcessor Simple Installer ===
echo Installing only essential prerequisites...

REM Install Python 3.11 x64 (if not already installed)
echo Checking for Python 3.11...
python --version >nul 2>&1
if errorlevel 1 (
    echo Installing Python 3.11 x64...
    winget install --id Python.Python.3.11 --scope user --force --accept-package-agreements --accept-source-agreements
) else (
    echo ✅ Python already installed
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
echo === Basic Prerequisites Complete! ===
echo.
echo ⚠️  NOTE: This installer skips Visual Studio Build Tools
echo ⚠️  webrtcvad may fail to install, but the app will work without it
echo ℹ️  Voice Activity Detection will use simple duration-based segmentation
echo.

REM Launch the main setup (which handles venv creation, requirements, and hardware-optimized PyTorch)
call run.bat

echo.
echo === Installation Complete! ===
echo You can now run: launch_gui.bat
echo Or run: python gui_transcribe.py
pause
