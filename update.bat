@echo off
echo === Speech2Text Code Updater ===
echo Updating to latest version...

REM Save current directory and navigate to the speech2textrme folder
cd /d "%USERPROFILE%\Downloads\speech2textrme"

REM Check if this is a git repository
if not exist ".git" (
    echo This doesn't appear to be a git repository.
    echo Downloading fresh copy...
    cd ..
    rmdir /s /q speech2textrme
    git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme
    if errorlevel 1 (
        echo Git clone failed, downloading ZIP...
        powershell -Command "Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile repo.zip; Expand-Archive repo.zip -Force; Move-Item 'repo\AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item repo.zip, repo -Recurse -Force"
    )
    cd speech2textrme
    goto :complete
)

REM Try to update via git pull
echo Pulling latest changes...
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
    cd speech2textrme
)

:complete
echo.
echo === Update Complete! ===
echo Latest code has been downloaded.
echo Your virtual environment and models are preserved.
echo You can now run: launch_gui.bat
pause
