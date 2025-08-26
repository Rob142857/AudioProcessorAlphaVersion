@echo off
echo === Speech2Text Complete Installer ===
echo Installing prerequisites for virgin Windows...

REM Install Python 3.11 x64
echo Installing Python 3.11 x64...
winget install --id Python.Python.3.11 --scope user --force --accept-package-agreements --accept-source-agreements

REM Install Visual C++ Redistributables
echo Installing Visual C++ Redistributables...
winget install --id Microsoft.VCRedist.2015+.x64 --force --accept-package-agreements --accept-source-agreements

REM Install Git
echo Installing Git...
winget install --id Git.Git --force --accept-package-agreements --accept-source-agreements

REM Navigate to Downloads
cd /d "%USERPROFILE%\Downloads"

REM Clone or download repository
if not exist "speech2textrme" (
    git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme
    if errorlevel 1 (
        echo Git failed, downloading ZIP...
        powershell -Command "Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile repo.zip; Expand-Archive repo.zip -Force; Move-Item 'repo\AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item repo.zip, repo -Recurse -Force"
    )
)

cd speech2textrme

echo === Prerequisites Complete! ===
echo Launching Speech2Text installer...

REM Launch the main installer
call run.bat

echo === Installation Complete! ===
echo You can now run: launch_gui.bat
pause
