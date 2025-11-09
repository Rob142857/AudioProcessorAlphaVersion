@echo off
cd /d "%~dp0"
echo Starting AudioProcessor GUI...
.venv\Scripts\python.exe gui_transcribe.py
pause