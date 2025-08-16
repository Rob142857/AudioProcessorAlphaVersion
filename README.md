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
When prompted, choose **option 2** (CPU-only PyTorch) for reliable performance.

---

## ARM64 Surface Installation

**For Surface laptops with ARM64 processors (Copilot+ PCs)**

### Option A: Standard Installation (Recommended)
**Step 1: PowerShell Setup**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Step 2: Install with x64 Emulation**  
```powershell
Set-Location "$env:USERPROFILE\Downloads"; if (Test-Path .\speech2textrme) { Set-Location .\speech2textrme; if (Get-Command git -ErrorAction SilentlyContinue) { git pull } } else { if (Get-Command git -ErrorAction SilentlyContinue) { git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme } else { Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile main.zip; Expand-Archive main.zip -Force; Rename-Item 'AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item main.zip }; Set-Location .\speech2textrme }; .\run.bat
```

**Step 3: Choose Option 2**  
When prompted, choose **option 2** (CPU-only PyTorch) - works through x64 emulation.

### Option B: NPU-Accelerated (Experimental)
**Step 1: PowerShell Setup** (same as above)

**Step 2: Clone Repository**  
```powershell
Set-Location "$env:USERPROFILE\Downloads"; if (Test-Path .\speech2textrme) { Set-Location .\speech2textrme; git pull } else { git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme; Set-Location .\speech2textrme }
```

**Step 3: Install NPU Dependencies**  
```powershell
python transcribe_npu.py --install-deps
```

**Usage:** `python transcribe_npu.py "input_file.mp4" --output "C:\Output"`

### Alternative: Use x64 Python
If ARM64 installation fails, install x64 Python from python.org instead of ARM64 Python, then follow the x64 instructions above.

---

## NPU Acceleration (Surface Copilot+ PCs)

**🚀 Experimental NPU Support for 2-5x faster transcription**

For Surface Copilot+ PCs with Qualcomm NPU:
```powershell
python transcribe_npu.py "input_file.mp4" --output "C:\Output" --model medium
```

Or use the GUI: `python gui_transcribe.py --device qnn`

**Requirements:** Surface Pro 11/Surface Laptop 7 or newer with Qualcomm NPU

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

### Headless Mode
```powershell
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --model medium --preprocess --vad --punctuate --keep-temp
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
- **ARM64/Surface Issues:**
  - **"Whisper model couldn't be imported":** PyTorch installation failed - neither PyTorch nor Miniforge support ARM64 Windows
  - **Options 2/3 in run.bat don't work:** Install x64 Python from python.org instead of ARM64 Python
  - **Best ARM64 solution:** Use x64 Python with x64 emulation (Windows 11 handles this well)
  - **Conda doesn't exist for ARM64 Windows:** Use regular pip with x64 Python instead
- **NPU Issues (Surface Copilot+ PCs):**
  - **NPU not detected:** Ensure you have a Qualcomm-powered Surface with NPU support
  - **`onnxruntime-qnn` install fails:** Make sure you're using ARM64 Python (not x64 for NPU support)
  - **Model loading errors with NPU:** Quantized models are required - they're automatically downloaded
  - **Fallback to CPU:** NPU will fallback to CPU if model isn't compatible

## Next Steps

- For DirectML default on Windows: install `torch` + `torch-directml` manually, run with `--device dml`
- For NPU acceleration on Surface Copilot+ PCs: Use `python transcribe_npu.py` for experimental NPU support
- For production NPU deployment: Convert Whisper models to quantized ONNX format using QNN SDK tools
- For vendor NPUs (ARM): Build ONNX exports and vendor runtimes (advanced, not included)

## Contact

Open an issue or create a PR with improvements. For specific PyTorch builds or automated DirectML installer requests, specify target hardware.
