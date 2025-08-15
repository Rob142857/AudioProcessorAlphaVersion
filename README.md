# Speech-to-Text Transcription (local, Whisper-based)

This repository provides a local speech-to-text pipeline using OpenAI Whisper plus optional preprocessing (FFmpeg), VAD segmentation, punctuation restoration, and a small Tkinter GUI wrapper.

## Quick Start

### 1. PowerShell Setup (one-time)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Windows x64 - Quick Bootstrap
```powershell
Set-Location "$env:USERPROFILE\Downloads"; if (Test-Path .\speech2textrme) { Set-Location .\speech2textrme; if (Get-Command git -ErrorAction SilentlyContinue) { git pull } } else { if (Get-Command git -ErrorAction SilentlyContinue) { git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme } else { Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile main.zip; Expand-Archive main.zip -Force; Rename-Item 'AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item main.zip }; Set-Location .\speech2textrme }; if (-not (Test-Path .venv)) { python -m venv .venv }; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; python -m pip install -r requirements.txt
```

### 3. Run the Application
```batch
.\run.bat
```

### Windows ARM64 (Surface) - Quick Bootstrap
```powershell
Set-Location "$env:USERPROFILE\Downloads"; if (Test-Path .\speech2textrme) { Set-Location .\speech2textrme; if (Get-Command git -ErrorAction SilentlyContinue) { git pull } } else { if (Get-Command git -ErrorAction SilentlyContinue) { git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme } else { Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile main.zip; Expand-Archive main.zip -Force; Rename-Item 'AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item main.zip }; Set-Location .\speech2textrme }; if (Test-Path .\run_conda.ps1) { powershell -ExecutionPolicy RemoteSigned -File .\run_conda.ps1 } else { Write-Host 'ERROR: Bootstrap script not found!' -ForegroundColor Red }
```

After completion: `conda activate speech2textrme`

---

## Manual Setup (Detailed Instructions)

### 0. Clone Repository (The line below clones to your downloads folder)
```powershell
Set-Location "$env:USERPROFILE\Downloads"; git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme; Set-Location .\speech2textrme
```

### 1. Create Virtual Environment
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip
```

### 2. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Install PyTorch (Optional)
**CPU-only:**
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA (example for CUDA 11.8):**
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**DirectML:** Install `torch-directml` manually and use `--device dml`

### 4. Preload Models (Recommended)
```powershell
python preload_models.py
```

Add `--include-large` for large model (~2.9GB):
```powershell
python preload_models.py --include-large
```

### 5. Install FFmpeg
```powershell
winget install gyan.ffmpeg
```

### 6. Environment Check (Optional)
```powershell
python check_env.py
```

## Usage

### GUI Mode
```powershell
python gui_transcribe.py
```

### Headless Mode
```powershell
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --model medium --preprocess --vad --punctuate --keep-temp
```

## Windows ARM (Surface) - Detailed Miniforge Setup

Run in PowerShell (copy-paste each block):

```powershell
$mf = "$env:TEMP\Miniforge3-Windows-arm64.exe"; Invoke-WebRequest "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-arm64.exe" -OutFile $mf; Start-Process -FilePath $mf -Wait
```

After installer completes, open new PowerShell:
```powershell
conda create -n speech2textrme python=3.11 -y; conda activate speech2textrme; conda install -c conda-forge numpy numba meson ninja -y; python -m pip install --upgrade pip; python -m pip install -r requirements.txt
```

## Features & Configuration

**Recommended quality flags:** `--preprocess --vad --punctuate --model medium`

**Input formats:** MP3, WAV, MP4, MOV (video → audio extracted automatically)

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
- For vendor NPUs (ARM): build ONNX exports and vendor runtimes (advanced, not included)

## Contact

Open an issue or create a PR with improvements. For specific PyTorch builds or automated DirectML installer requests, specify target hardware.
