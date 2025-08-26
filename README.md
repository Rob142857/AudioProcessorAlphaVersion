# Speech-to-Text Transcription (local, Whisper-based)

This repository provides a local speech-to-text pipeline using OpenAI Whisper plus optional preprocessing (FFmpeg), VAD segmentation, punctuation restoration, and a small Tkinter GUI wrapper.

**System Requirements:**
- Windows 10/11 x64 architecture
- Python 3.8+ (x64 version) - download from python.org
- 4GB+ RAM (8GB+ recommended for large models)
- Optional: NVIDIA GPU with 4GB+ VRAM for CUDA acceleration
- Optional: Any DirectX 12 compatible GPU for DirectML acceleration

**Prerequisites (Virgin Windows Install):**
The super-simple installer handles everything automatically! If you want to install manually, you'll need:

1. **Python 3.8+ x64**: Download from [python.org](https://python.org) - **must be x64 version**
2. **Visual C++ Redistributables**: For Python package compilation
3. **Git** (optional): Enables automatic updates
4. **PowerShell permissions**: To run installation scripts

**That's it!** The installer handles everything else (Python, Visual C++, Git, packages, PyTorch, FFmpeg, model downloads).

## Ultra-Quick Start (Copy & Paste)

### Super-Simple Install (Handles EVERYTHING)
**For complete virgin Windows - installs Python, Visual C++, Git, everything:**
```powershell
irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex
```

**Or download and run the batch file:**
```cmd
curl -L https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.bat -o install.bat && install.bat
```

### Alternative One-Line Install
**If you already have Python x64 installed:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser; winget install --id Microsoft.VCRedist.2015+.x64 --force; Set-Location "$env:USERPROFILE\Downloads"; if (!(Test-Path .\speech2textrme)) { if (Get-Command git -ErrorAction SilentlyContinue) { git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme } else { Invoke-WebRequest 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile repo.zip; Expand-Archive repo.zip -Force; Move-Item "repo\AudioProcessorAlphaVersion-main" "speech2textrme"; Remove-Item repo.zip, repo -Recurse -Force } }; Set-Location .\speech2textrme; .\run.bat
```

### Hardware Selection
When prompted in `run.bat`, choose:
- **Option 2**: CPU-only processing (reliable, slower)
- **Option 3**: NVIDIA GPU acceleration (requires CUDA-compatible GPU)
- **Option 4**: DirectML GPU acceleration (Windows GPU acceleration for AMD/Intel/NVIDIA)

### NVIDIA GPU Users (Optional)
**For best GPU performance, install NVIDIA drivers:**
- **GeForce Experience** (recommended): Automatically manages drivers
- **Manual install**: Download from [nvidia.com/drivers](https://nvidia.com/drivers)
- **Check compatibility**: GTX 1060+ or RTX series required for CUDA acceleration

---

## Manual Setup (Advanced Users)

If you prefer manual control over the installation process:

```powershell
# 1. Clone and setup
Set-Location "$env:USERPROFILE\Downloads"
git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme
Set-Location .\speech2textrme

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. Install base requirements
python -m pip install -r requirements.txt

# 4. Choose PyTorch build:
# CPU-only: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA:     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# DirectML: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu; python -m pip install torch-directml

# 5. Preload models and install FFmpeg
python preload_models.py
winget install gyan.ffmpeg
```

---

## Hardware Acceleration Options

### CPU Processing
- **Compatibility**: Works on all x64 Windows systems
- **Performance**: Slower but reliable
- **Memory**: Uses system RAM
- **Setup**: Default option, no additional drivers needed

### NVIDIA GPU Acceleration (CUDA)
- **Requirements**: NVIDIA GTX 1060+ or RTX series GPU
- **Performance**: 2-5x faster than CPU
- **Memory**: Uses GPU VRAM (4GB+ recommended)
- **Setup**: Requires NVIDIA drivers (GeForce Experience or NVIDIA website)
- **Device setting**: Use `cuda` or `auto` in GUI

### DirectML GPU Acceleration
- **Compatibility**: AMD, Intel, and NVIDIA GPUs on Windows 10/11
- **Performance**: 1.5-3x faster than CPU (varies by GPU)
- **Memory**: Uses GPU memory
- **Setup**: Built into Windows, no additional drivers needed
- **Device setting**: Use `dml` in GUI

---

## Usage

### GUI Mode (Recommended)
**Using the launcher (ensures proper environment):**
```powershell
.\launch_gui.bat
```

**Or manually:**
```powershell
python gui_transcribe.py
```

### Device Selection in GUI
- **auto**: Automatically selects best available (CUDA > DirectML > CPU)
- **cuda**: Force NVIDIA GPU acceleration (if available)
- **dml**: Force DirectML GPU acceleration (if available)  
- **cpu**: Force CPU-only processing

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
- **Hardware acceleration**: Supports CPU, NVIDIA CUDA, and DirectML (Windows GPU)
- **GUI Interface**: Simple Tkinter-based file picker and progress display
- **x64 optimized**: Built for Windows x64 architecture

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
- `run.bat` — Windows bootstrap: creates `.venv`, installs requirements, offers PyTorch options
- `launch_gui.bat` — Proper GUI launcher ensuring correct Python environment and CUDA support
- `check_env.py` — writes `env_report.json` with system capability report

## Troubleshooting

**Virgin Windows Install Issues:**
- **"winget not recognized"**: Update Windows to latest version or install from Microsoft Store
- **Python package build errors**: Ensure Visual C++ Redistributables are installed: `winget install --id Microsoft.VCRedist.2015+.x64 --force`
- **PowerShell script blocked**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Missing Python**: Download x64 version from [python.org](https://python.org) - **not from Microsoft Store**
- **"Microsoft Visual C++ 14.0 is required"**: Install Visual Studio Build Tools: `winget install --id Microsoft.VisualStudio.2022.BuildTools`

**Environment Issues:**
- **Import failures:** Confirm `.venv` is activated and `pip install -r requirements.txt` succeeded
- **Wrong Python version:** Ensure you're using x64 Python, not x86 or ARM64

**GPU Acceleration Issues:**
- **CUDA not working:** Verify NVIDIA GPU drivers are installed and GPU has 4GB+ VRAM
- **DirectML issues:** Update Windows to latest version, ensure GPU drivers are current
- **Performance problems:** Check Task Manager → Performance → GPU during transcription

**General Issues:**
- **FFmpeg errors:** Install via `winget install gyan.ffmpeg` or ensure `ffmpeg.exe` is on PATH
- **Model download failures:** Retry on stable network (medium model ~1.4GB, large ~2.9GB)
- **Memory errors:** Use smaller model (`base` instead of `large`) or reduce to CPU processing

**Performance Comparison (approximate):**
- **CPU (Intel i7)**: ~0.3x real-time (5min audio = 15min processing)
- **CUDA (GTX 1070 Ti)**: ~2x real-time (5min audio = 2.5min processing)  
- **DirectML (varies)**: ~1.5x real-time (5min audio = 3.5min processing)

## Next Steps & Advanced Usage

**For CUDA users with NVIDIA GPUs:**
- Use `launch_gui.bat` for reliable CUDA environment setup
- Monitor GPU usage in Task Manager → Performance → GPU during transcription
- For maximum performance, use models: `medium` or `large`

**For DirectML users (AMD/Intel/NVIDIA):**
- Set device to `dml` in GUI dropdown
- Performance varies by GPU - RTX/RX 6000+ series work best
- Fallback to CPU if DirectML fails to initialize

**For production deployment:**
- Consider model quantization for faster inference
- Use batch processing for multiple files
- Implement custom post-processing for domain-specific terminology

**Hardware Recommendations:**
- **Budget**: CPU-only with 16GB RAM
- **Balanced**: GTX 1660+ or RTX 3060+ with 6GB+ VRAM  
- **Performance**: RTX 4070+ with 12GB+ VRAM

## Contact

Open an issue or create a PR with improvements. For specific PyTorch builds or automated DirectML installer requests, specify target hardware.
