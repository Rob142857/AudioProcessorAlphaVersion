# Speech-to-Text Transcription Tool

**What it does:** This program converts audio and video files into text documents. Simply drag in your file, and it creates a Word document with the transcription in your Downloads folder.

**Input files:** Accepts most audio formats (MP3, WAV, M4A, FLAC) and video formats (MP4, AVI, MKV, MOV, WMV)

**Output:** Creates a clean Word document (.docx) and text file (.txt) with the transcription, automatically saved to your Downloads folder

**AI Models:** Uses OpenAI's Whisper AI models:
- **Medium** - Good quality, faster processing (~1.4GB download)
- **Large** - Best quality, slower processing (~2.9GB download) - **Recommended**

**Processing Options:**
1. **CPU Processing** - Works on any computer, slower but reliable
2. **NVIDIA GPU** - 2-5x faster if you have an NVIDIA graphics card (GTX 1060+ or RTX series)
3. **DirectML GPU** - 1.5-3x faster for AMD/Intel/NVIDIA graphics cards on Windows

**System Requirements:** Windows 10/11 (64-bit), 4GB+ RAM (8GB+ recommended)

---

## Installation (Copy & Paste One Command)

### Super-Simple Install (Handles Everything)
**For any Windows computer - installs Python, everything needed:**

**Option 1 - PowerShell:**
```powershell
irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex
```

**Option 2 - Command Prompt:**
```cmd
curl -L https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.bat -o install.bat && install.bat
```

### What Happens During Install:
1. Installs Python and required components automatically
2. Downloads the speech-to-text program to your Downloads folder
3. Asks you to choose your processing method (CPU, NVIDIA GPU, or DirectML GPU)
4. Downloads the AI model (Medium by default, you can choose Large for better quality)
5. Ready to use!

### NVIDIA GPU Users
**For best performance with NVIDIA graphics cards:**
- Install [GeForce Experience](https://www.nvidia.com/geforce/geforce-experience/) for automatic driver updates
- Or download drivers manually from [nvidia.com/drivers](https://nvidia.com/drivers)
- Requires GTX 1060+ or any RTX series card

---

## How to Use

### Quick Start
1. **Run the installer** (copy/paste command above)
2. **Choose your processing method** when prompted (GPU recommended if available)
3. **Launch the program:** Double-click `launch_gui.bat` in the `speech2textrme` folder

### Using the Program
1. **Click "Browse"** to select your audio/video file
2. **Choose model:** Medium (faster) or Large (better quality)
3. **Select processing device:** Auto (recommended), NVIDIA GPU, DirectML GPU, or CPU
4. **Click "Run"** and wait for transcription to complete
5. **Find your files** in the Downloads folder: `filename_transcription.docx` and `filename_transcription.txt`

### Recommended Settings for Best Quality
- **Model:** Large
- **Processing:** Auto (automatically picks fastest available)
- **Enable checkboxes:** Preprocess, VAD segmentation, Restore punctuation

---

## Performance Guide

**Processing Times (approximate for 5-minute audio):**
- **CPU Only:** 15 minutes (slow but works everywhere)
- **NVIDIA GPU (GTX 1070 Ti):** 2.5 minutes (fast)
- **DirectML GPU:** 3.5 minutes (good for AMD/Intel cards)

**Quality Comparison:**
- **Medium Model:** Good transcription quality, smaller download
- **Large Model:** Excellent transcription quality, better with accents and difficult audio

---

## Troubleshooting

**Installation Issues:**
- **"winget not recognized"** - Update Windows or install from Microsoft Store
- **Python errors** - The installer handles this, but make sure you're using 64-bit Windows
- **Download fails** - Check your internet connection and try again

**Program Issues:**
- **"No module found"** - Run `launch_gui.bat` instead of trying to run Python directly
- **GPU not working** - Install latest graphics drivers
- **Out of memory** - Use Medium model instead of Large, or switch to CPU processing

**Need Help?** Open an issue on the GitHub repository with your error message.

---

## Advanced Users

**Command Line Usage:**
```powershell
# Basic transcription
python transcribe.py "C:\path\to\audio.mp3"

# Full quality processing
python transcribe.py "video.mp4" --model large --preprocess --vad --punctuate
```

**Manual Installation:** See the original README for step-by-step manual setup instructions.

**File Locations:**
- **Program folder:** `%USERPROFILE%\Downloads\speech2textrme`
- **Output files:** `%USERPROFILE%\Downloads\`
- **Models cached:** `%USERPROFILE%\.cache\whisper\`
