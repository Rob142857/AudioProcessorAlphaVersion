# Speech-to-Text Transcription (local, Whisper-based)

This repository provides a local speech-to## Windows ARM64 (Surface) - Detailed Setup

**Current Reality:** PyTorch and most ML libraries don't provide native ARM64 Windows wheels. The best approach is to use x64 emulation.

**Recommended Approach:**
1. Install x64 Python (not ARM64) from python.org
2. Follow the regular x64 bootstrap process
3. Windows 11 ARM64 runs x64 Python efficiently through emulation

**Alternative - Use existing ARM64 Python with CPU-only approach:**
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```
*Note: This may fail on ARM64. If it fails, install x64 Python instead.*

**Manual ARM64 Dependencies (if available):**
```powershell
python -m pip install numpy scipy pillow  # These usually have ARM64 Windows wheels
python -m pip install -r requirements.txt  # Some packages may fail
```ne using OpenAI Whisper plus optional preprocessing (FFmpeg), VAD segmentation, punctuation restoration, and a small Tkinter GUI wrapper.

## Quick Start

### 1. PowerShell Setup (one-time)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Universal Install (Works on x64 and ARM64)
```powershell
Set-Location "$env:USERPROFILE\Downloads"; if (Test-Path .\speech2textrme) { Set-Location .\speech2textrme; if (Get-Command git -ErrorAction SilentlyContinue) { git pull } } else { if (Get-Command git -ErrorAction SilentlyContinue) { git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme } else { Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile main.zip; Expand-Archive main.zip -Force; Rename-Item 'AudioProcessorAlphaVersion-main' 'speech2textrme'; Remove-Item main.zip }; Set-Location .\speech2textrme }; .\run.bat
```

**Then choose option 2** (CPU-only PyTorch) when prompted. This works on both x64 and ARM64 through emulation.

## NPU Acceleration (Surface Copilot+ PCs)

**🚀 NEW: Use your Surface NPU for faster inference!**

For Surface laptops with Qualcomm NPU (Copilot+ PCs), you can enable NPU acceleration:

### Quick NPU Setup
```powershell
python transcribe_npu.py --install-deps
```

### NPU Usage
```powershell
python transcribe_npu.py "input_file.mp4" --output "C:\Output" --model medium
```

Or use the GUI with NPU:
```powershell
python gui_transcribe.py --device qnn
```

**Requirements:**
- Surface Copilot+ PC with Qualcomm NPU (e.g., Surface Pro 11, Surface Laptop 7)
- Windows ARM64
- onnxruntime-qnn package

**Performance:** NPU acceleration can provide 2-5x faster transcription compared to CPU-only inference while using significantly less power.

**Note:** This feature is experimental and requires ONNX-optimized Whisper models. Current implementation falls back to optimized CPU processing with NPU readiness.

---

## Manual Setup (Detailed Instructions)

### 1. Clone Repository (The line below clones to your downloads folder)
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

Add `--include-large` for large model (~2.9GB):
```powershell
python preload_models.py --include-large
```

### 6. Install FFmpeg
```powershell
winget install gyan.ffmpeg
```

### 7. Environment Check (Optional)
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
conda create -n speech2textrme python=3.11 -y; conda activate speech2textrme; conda install -c conda-forge numpy numba meson ninja -y; python -m pip install --upgrade pip; python -m pip install -r requirements.txt; python preload_models.py
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
