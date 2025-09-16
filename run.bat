@echo off
REM Simple launcher: activate venv and run GUI (assumes dependencies are installed)

if not exist ".venv\Scripts\Activate.bat" (
  echo Error: Virtual environment not found. Run install.ps1 first.
  pause
  exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\Activate.bat

echo Checking for required packages...
python -c "import sys; import whisper, psutil, docx; print('✓ All required packages found')" 2>nul
if errorlevel 1 (
  echo.
  echo ❌ Required packages not found. Run install.ps1 first to install dependencies.
  echo.
  pause
  exit /b 1
)

echo.
echo � Launching Speech to Text Transcription Tool GUI...
echo.
python gui_transcribe.py --gui

if errorlevel 1 (
  echo.
  echo ❌ GUI failed to start. Check the error messages above.
  echo.
)
pause
