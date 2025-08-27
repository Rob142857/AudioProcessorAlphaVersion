@echo off
REM Troubleshooting batch script for transcription issues
echo 🔧 AudioProcessorAlphaVersion - Transcription Troubleshooting
echo.

if "%~1"=="" (
    echo Usage: troubleshoot.bat "path\to\your\audio_file.mp3"
    echo.
    echo This script will run multiple transcription tests to identify why transcripts are too short.
    echo It will test different combinations of VAD segmentation and model sizes.
    echo.
    pause
    exit /b 1
)

set INPUT_FILE=%~1

if not exist "%INPUT_FILE%" (
    echo ❌ Error: Input file not found: %INPUT_FILE%
    echo.
    pause
    exit /b 1
)

echo 📁 Input file: %INPUT_FILE%
echo 📂 Output will be saved to Downloads folder
echo.

echo 🧪 Running troubleshooting tests...
echo This will take several minutes depending on your hardware and audio length.
echo.

python troubleshoot_transcription.py "%INPUT_FILE%"

echo.
echo ✅ Troubleshooting complete!
echo 📊 Check your Downloads folder for multiple test results
echo 💡 Compare the file sizes and content to see which method captured the most text
echo.

pause
