# Speech-to-Text Transcription (local, Whisper-based)

This repository provides a local speech-to-text pipeline using OpenAI Whisper plus optional preprocessing (FFmpeg), VAD segmentation, punctuation restoration, and a small Tkinter GUI wrapper.

## Quick Start (Windows x64)

### 1. PowerShell Setup (one-time)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Everything
```powershell
Set-Location "$env:USERPROFILE\Downloads"; if (Test-Path .\speech2textrme) { Set-Location .\speech2textrme; if (Get-Command git -ErrorAction SilentlyContinue) { git pull } } else { if (Get-Command git -ErrorAction SilentlyContinue) { git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme } else { Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile main.zip; Expand-Archive main.zip -Force; Rename-Item 'AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item main.zip }; Set-Location .\speech2textrme }; .\run.bat
```

### 3. Choose PyTorch Option
When prompted, choose **option 2** (CPU-only PyTorch) for reliable performance, or **option 3** if you want DirectML GPU acceleration.

---

## Manual Setup (Advanced Users)

### 1. Clone Repository
```powershell
Set-Location "$env:USERPROFILE\Downloads"; git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme; Set-Location .\speech2textrme
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Install PyTorch (Optional)
**CPU-only:**
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA (example for CUDA 11.8):**
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**DirectML:** Install `torch-directml` manually and use `--device dml`

### 5. Preload Models (Recommended)
```powershell
python preload_models.py
```

### 6. Install FFmpeg
```powershell
winget install gyan.ffmpeg
```

---

## Usage

### GUI Mode
```powershell
python gui_transcribe.py
```

### Command Line Examples
**Basic transcription:**
```powershell
python transcribe.py "C:\path\to\audio.mp3"
```

**Full processing pipeline:**
```powershell
python transcribe.py "video.mp4" --outdir "C:\Output" --model medium --preprocess --vad --punctuate --keep-temp
```

**Headless GUI mode:**
```powershell
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --model medium --preprocess --vad --punctuate --keep-temp
```

## Features

- **Local transcription**: Uses OpenAI Whisper models (no cloud API calls)
- **Preprocessing**: FFmpeg-based audio extraction and VAD (voice activity detection)
- **Post-processing**: Punctuation restoration and capitalization
- **Multi-format**: Outputs to `.txt` and `.docx` with optional timestamps
- **Hardware acceleration**: Supports CPU and DirectML (Windows GPU)
- **GUI Interface**: Simple Tkinter-based file picker and progress display

## Supported Input Formats

Audio: `.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`, `.ogg`, `.wma`  
Video: `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.flv`, `.webm`

## Configuration

**Recommended quality flags:** `--preprocess --vad --punctuate --model medium`

**Output formats:** 
- `.txt` (cleaned & paragraphed)
- `.docx` (Word format, paragraphed) 
- `<input_basename>_transcription_log.json` (run metadata, timings, file paths)

**Helper scripts:**
- `run.bat` — Windows bootstrap: creates `.venv`, installs requirements, launches GUI
- `check_env.py` — writes `env_report.json` with system capability report

## Troubleshooting

- **Import failures:** Confirm `.venv` is activated and `pip install -r requirements.txt` succeeded
- **FFmpeg errors:** Ensure `ffmpeg.exe` is on PATH or install via `winget install gyan.ffmpeg`
- **Model download failures:** Retry on stable network (files are large: medium ~1.4GB, punctuation model ~2+GB)
- **DirectML setup:** Install `torch-directml` manually, then use `--device dml` flag

## Next Steps

- For DirectML default on Windows: install `torch` + `torch-directml` manually, run with `--device dml`
- For production deployment: Consider model quantization for faster inference
- For custom models: Replace Whisper model paths in the code

## Contact

Open an issue or create a PR with improvements. For specific PyTorch builds or automated DirectML installer requests, specify target hardware.
