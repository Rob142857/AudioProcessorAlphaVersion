# Speech to Text Transcription (Large Model)

High‑quality local transcription using the Large Whisper model. Outputs .txt and .docx next to the source file.

## Requirements
- Windows with Python 3.10+ (PowerShell recommended)
- ffmpeg on PATH

## Install
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

Optional Torch builds:
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Preload the Large model cache:
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
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --threads 16 --batch-size 24 --ram-gb 8 --vram-fraction 0.75
```

## Performance knobs
CLI flags (override auto-detected settings):
```powershell
python transcribe_optimised.py --input "C:\path\to\file.mp3" --threads 16 --batch-size 24 --ram-gb 8 --ram-fraction 0.8 --vram-gb 6 --vram-fraction 0.75
```

Environment variables (persist for the session):
```powershell
$env:TRANSCRIBE_THREADS = '16'
$env:TRANSCRIBE_BATCH_SIZE = '24'
$env:TRANSCRIBE_RAM_GB = '8'
$env:TRANSCRIBE_RAM_FRACTION = '0.8'
$env:TRANSCRIBE_VRAM_GB = '6'
$env:TRANSCRIBE_VRAM_FRACTION = '0.75'
python gui_transcribe.py
```

## Notes
- Large model is used by default for best accuracy.
- Outputs .txt and .docx are written alongside the input file.
- If CUDA is available, it will be used; otherwise CPU is used automatically.
