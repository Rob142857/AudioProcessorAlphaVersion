# Speech to Text Transcription (Whisper large‑v3‑turbo)

High‑quality local transcription using Whisper large‑v3‑turbo by default. Outputs .txt and .docx next to the source file.

## Requirements
- Windows with Python 3.10+ (PowerShell recommended)
- ffmpeg on PATH

## Install

Quick install (PowerShell one‑liner):
```powershell
irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex
```

Local script runner:
```powershell
./run.bat
```

Manual setup (if you want control over Torch):
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch (choose ONE that matches your hardware):
```powershell
# CPU-only (universal):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA CUDA 11.8 (good for GTX 10xx/16xx, RTX 20xx):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

# NVIDIA CUDA 12.1+ (newer RTX 30/40 series):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Windows DirectML (AMD/Intel GPUs):
python -m pip install torch-directml
```

Preload the turbo model cache (recommended to avoid first‑run delay):
```powershell
python preload_models.py
```

Verify environment (writes env_report.json):
```powershell
python check_env.py
```

## Run
GUI:
```powershell
python gui_transcribe.py
```

Headless (save next to source):
```powershell
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --threads 16 --ram-gb 8 --vram-fraction 0.75
```

## Performance knobs
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
- Default model: Whisper large‑v3‑turbo (falls back to large‑v3, then large if needed).
- Outputs .txt and .docx are written alongside the input file.
- If CUDA is available, it will be used; otherwise CPU or DirectML is used automatically.
- No batch size or VAD options are passed to Whisper (maximizes compatibility across versions).
- CPU threads default to ~90% of logical cores (override via --threads or TRANSCRIBE_THREADS).
- RAM plan defaults to ~95% of currently available memory (override via env/CLI). 
- Punctuation restoration runs twice for improved results.

### Troubleshooting
- If you see errors about missing torch, install PyTorch using one of the commands above that matches your hardware, then re-run preload and the GUI.
- Ensure ffmpeg is on PATH. A local ffmpeg.exe is included here; you can add it to PATH or install via winget.
