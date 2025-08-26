@echo off
REM GUI Launcher with proper virtual environment activation
echo Launching Speech-to-Text GUI with CUDA support...
cd /d "%~dp0"

REM Activate virtual environment and prioritize its Python
if exist ".venv\Scripts\Activate.bat" (
    call .venv\Scripts\Activate.bat
    REM Force the venv Python to be first in PATH
    set "PATH=%~dp0.venv\Scripts;%PATH%"
) else (
    echo ERROR: Virtual environment not found!
    echo Please run run.bat first to set up the environment.
    pause
    exit /b 1
)

REM Verify we're using the right Python
echo Verifying Python environment...
python -c "import sys, torch; print('Python:', sys.executable); print('CUDA available:', torch.cuda.is_available())"

REM Launch GUI
echo Starting GUI with CUDA-enabled environment...
python gui_transcribe.py

pause
