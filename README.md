# Audio Processor Alpha Version (Whisper large‑v3‑turbo)

High‑quality local transcription using Whisper large‑v3‑turbo by default. Outputs .txt and .docx next to the source file.

## Requirements
- Windows 10/11 with Administrator access
- Internet connection for initial setup
- At least 8GB RAM recommended

## Install

### One-Command Complete Installation (Recommended)
Run this single PowerShell command as Administrator to install everything automatically:
```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex"
```

**What this does:**
- ✅ Installs Python 3.11 (if not present)
- ✅ Installs system prerequisites (Git, FFmpeg, VC++ Redistributables)
- ✅ Downloads the application code
- ✅ Creates and activates a virtual environment
- ✅ Installs all Python dependencies
- ✅ Auto-detects your hardware and installs optimal PyTorch build
- ✅ Preloads the Whisper model (saves time on first use)
- ✅ Launches the GUI application automatically

**Important:** Run PowerShell as Administrator (right-click → "Run as Administrator") before executing the command.

### Installation Options
The installer supports these parameters:
```powershell
# Skip system prerequisites (if already installed)
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex -SkipPrerequisites"

# Skip model preloading
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex -SkipModelPreload"

# Force CPU-only PyTorch (ignore GPU detection)
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex -ForceCpuTorch"
```

### Manual Setup (Advanced Users)
If you prefer manual control:
```powershell
# 1. Clone repository
git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git
cd AudioProcessorAlphaVersion

# 2. Run installer with local script
.\install.ps1
```

### Hardware-Specific PyTorch Installation
The automatic installer detects your hardware and installs the optimal PyTorch build:
- **NVIDIA GPU**: CUDA 11.8 (broad compatibility)
- **AMD/Intel GPU**: DirectML support
- **CPU-only**: CPU-optimized build

If you need a different PyTorch version, you can override after installation:
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install specific PyTorch build
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1+
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
python -m pip install torch-directml  # DirectML for AMD/Intel GPUs
```

### Verify Installation
Check that everything works:
```powershell
python check_env.py  # Writes env_report.json with system info
```

## Run
The installer launches the GUI automatically. To run manually:

**GUI Mode:**
```powershell
python gui_transcribe.py
```

**Headless Mode (save next to source):**
```powershell
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --threads 16 --ram-gb 8 --vram-fraction 0.75
```

## Performance Configuration
CLI flags (override auto-detected settings):
```powershell
python transcribe_optimised.py --input "C:\path\to\file.mp3" --threads 16 --ram-gb 8 --ram-fraction 0.8 --vram-gb 6 --vram-fraction 0.75
```

Environment variables (persist for the session):
```powershell
$env:TRANSCRIBE_THREADS = '16'
$env:TRANSCRIBE_RAM_GB = '8'
$env:TRANSCRIBE_RAM_FRACTION = '0.8'
$env:TRANSCRIBE_VRAM_GB = '6'
$env:TRANSCRIBE_VRAM_FRACTION = '0.75'
python gui_transcribe.py
```

## Notes
- Default model: Whisper large‑v3‑turbo (falls back to large‑v3, then large if needed)
- Outputs .txt and .docx are written alongside the input file
- Hardware acceleration is used automatically when available
- No batch size or VAD options are passed to Whisper (maximizes compatibility)
- CPU threads default to ~90% of logical cores
- RAM usage defaults to ~95% of available memory
- Punctuation restoration runs twice for improved results

### Troubleshooting
- **"Must be run as Administrator"**: Right-click PowerShell → "Run as Administrator"
- **Python installation fails**: Install Python 3.11 manually from python.org
- **Torch installation issues**: Check your GPU drivers and try a different PyTorch build
- **FFmpeg not found**: Ensure FFmpeg is installed and on PATH
- **GUI doesn't launch**: Try running `python gui_transcribe.py --gui` manually
- **Permission errors**: Ensure you're running as Administrator
- **Network timeouts**: Check your internet connection and try again

### System Requirements Details
- **OS**: Windows 10 version 1903+ or Windows 11
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 2GB free space (plus model cache)
- **Network**: Required for initial setup and model downloads
- **GPU**: Optional but recommended (NVIDIA, AMD, or Intel with DirectML)
