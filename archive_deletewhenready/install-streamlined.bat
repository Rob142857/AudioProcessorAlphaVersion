@echo off
REM Quick launcher for streamlined installation
REM Uses the new robust installation script with minimal dependencies

echo.
echo ================================================================
echo  Audio Processor Alpha Version - Streamlined Installation
echo ================================================================
echo.
echo This will install Audio Processor with:
echo  - Minimal dependencies (no legacy webrtcvad/moviepy)
echo  - Robust error handling and retries
echo  - Hardware-optimized PyTorch
echo  - Proper validation
echo.

choice /C YN /M "Continue with installation"
if errorlevel 2 exit /b 0

echo.
echo Starting installation...
echo.

REM Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell available'" >nul 2>&1
if errorlevel 1 (
    echo Error: PowerShell is required but not found.
    pause
    exit /b 1
)

REM Run the robust installation script
powershell -ExecutionPolicy Bypass -File "install-robust.ps1"

echo.
if errorlevel 1 (
    echo Installation failed. Check the error messages above.
) else (
    echo Installation completed successfully!
    echo.
    echo You can now run the application with: python gui_transcribe.py
)

echo.
pause