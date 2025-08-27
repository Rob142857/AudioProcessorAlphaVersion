@echo off
echo === Speech2Text Simple Installer ===
echo Installing only essential prerequisites...

REM Install Python 3.11 x64
echo Installing Python 3.11 x64...
winget install --id Python.Python.3.11 --scope user --force --accept-package-agreements --accept-source-agreements

REM Install Git for downloading the repository
echo Installing Git...
winget install --id Git.Git --force --accept-package-agreements --accept-source-agreements

REM Navigate to Downloads
cd /d "%USERPROFILE%\Downloads"

REM Navigate to Downloads
cd /d "%USERPROFILE%\Downloads"

REM Clone or update repository
if exist "speech2textrme" (
    echo Found existing installation, updating...
    cd speech2textrme
    git pull origin main
    if errorlevel 1 (
        echo Git pull failed, downloading fresh copy...
        cd ..
        rmdir /s /q speech2textrme
        git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme
        if errorlevel 1 (
            echo Git clone failed, downloading ZIP...
            powershell -Command "Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile repo.zip; Expand-Archive repo.zip -Force; Move-Item 'repo\AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item repo.zip, repo -Recurse -Force"
        )
    )
) else (
    echo Fresh installation...
    git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme
    if errorlevel 1 (
        echo Git failed, downloading ZIP...
        powershell -Command "Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile repo.zip; Expand-Archive repo.zip -Force; Move-Item 'repo\AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item repo.zip, repo -Recurse -Force"
    )
)

cd speech2textrme

echo === Basic Prerequisites Complete! ===
echo.
echo ⚠️  NOTE: This installer skips Visual Studio Build Tools
echo ⚠️  webrtcvad may fail to install, but the app will work without it
echo ℹ️  Voice Activity Detection will use simple duration-based segmentation
echo.

REM Launch the main installer
call run.bat

echo === Installation Complete! ===
echo You can now run: launch_gui.bat
pause
