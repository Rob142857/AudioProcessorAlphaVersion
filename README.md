# AudioProcessor v1.0.0-beta.1

Ultra-high-performance audio and video transcription system for Windows x64 machines, featuring OpenAI Whisper with advanced text processing and intelligent optimization.

> **Latest Update**: v1.0.0-beta.1 - Stable working version with full feature set. One-click x64 GeForce installer coming soon!

## What It Does

**AudioProcessor** converts audio and video files into high-quality text documents with professional formatting. It automatically detects your hardware and uses maximum available resources for optimal performance.

### Key Features

🎯 **Ultra-Aggressive Performance**
- Uses 98% of available RAM and 99% of VRAM for maximum speed
- Automatic hardware detection (CUDA GPU → DirectML → CPU)
- Whisper large-v3-turbo model with English optimization
- Typically processes at 15-40x realtime speed

🧠 **Advanced Text Processing**
- **6-pass text enhancement** with parallel processing across CPU cores
- **Intelligent punctuation** restoration and capitalization
- **Context-aware paragraph** formatting with semantic grouping
- **Quality assessment** and readability optimization
- **Multi-threaded processing** utilizing up to 75% of CPU cores

📄 **Professional Output**
- Clean, properly formatted transcriptions
- Outputs both `.txt` and `.docx` files next to source
- Automatic artifact removal (watermarks, music, etc.)
- Enhanced readability with proper sentence structure

⚡ **Intelligent Optimization**
- **VAD (Voice Activity Detection)** for speech-focused processing
- **Parallel segment processing** for large files
- **Dynamic resource allocation** based on content size
- **Automatic fallbacks** ensure compatibility

🔧 **Hardware Acceleration**
- **NVIDIA GPUs**: CUDA with cuBLAS optimizations
- **AMD/Intel GPUs**: DirectML support
- **CPU-only**: Optimized multi-threading
- **Mixed processing**: GPU + CPU parallel workflows

## Quick Start (Windows x64)

### One-Click Installation (Coming Soon!)
**For NVIDIA GeForce Users:**
```powershell
# One-click installer with automatic CUDA detection (In Development)
# Will detect your GeForce GPU and install optimal CUDA version automatically
```

### Current Installation
- **Windows 10/11 x64** with Administrator access
- **8GB+ RAM recommended** (16GB+ for optimal performance)
- **Internet connection** for initial setup
- **Optional**: NVIDIA/AMD GPU for acceleration

### Current Installation

#### Requirements
```powershell
# Run this single command (auto-detects GPU/CPU and installs accordingly)
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install_new.ps1 | iex"
```

Optional (run with parameters like forcing CPU/CUDA):
```powershell
# Download locally, then run with flags
iwr https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install_new.ps1 -OutFile install_new.ps1
powershell -ExecutionPolicy Bypass -File .\install_new.ps1 -Device cpu   # or -Device cuda
```

This automatically:
- ✅ Installs Python 3.11 and system prerequisites
- ✅ Sets up optimized virtual environment
- ✅ Detects hardware and installs optimal PyTorch
- ✅ Downloads Whisper models
- ✅ Launches the GUI

### Alternative Setup
```powershell
# Quick local setup
.\run.bat
```

**Or manual:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python install_text_processing.py  # Install enhanced text processing
python preload_models.py           # Download Whisper models
```

## Usage

### GUI Mode (Recommended)
```powershell
python gui_transcribe.py
```
- Drag & drop audio/video files
- Automatic optimal settings
- Real-time progress monitoring
- Professional formatted output

### Command Line
```powershell
# Basic usage (auto-optimized)
python transcribe_optimised.py --input "audio.mp4"

# With VAD for speech-heavy content
python transcribe_optimised.py --input "lecture.mp4" --vad

# Custom resource limits
python transcribe_optimised.py --input "large_file.mp4" --threads 16 --ram-gb 12 --vram-fraction 0.9
```

### Biasing recognition with domain terms (awkward words)
You can guide Whisper to “listen out for” special terms using an initial prompt built from your word list. Provide terms via one of these:

- Place an `awkward_words.txt` (or `.md`) next to your input file
- Or place `awkward_words.txt` in the repo root
- Or set an env file path: `TRANSCRIBE_AWKWARD_FILE="C:\\path\\to\\my_terms.txt"`
- Or set a comma list: `TRANSCRIBE_AWKWARD_TERMS="Schrödinger, Euler, Noether, Fourier"`

Each line is one term; bullets like `-` or `*` are ignored. The installer will then pass a concise combined hint to Whisper as the initial prompt to improve recognition of those words.

### Performance Options

**VAD (Voice Activity Detection)**
```powershell
$env:TRANSCRIBE_VAD = '1'          # Enable intelligent speech detection
```
- 20-40% performance boost for content with silence/music
- Automatic fallback if VAD unavailable
- Parallel segment processing

**Resource Control**
```powershell
$env:TRANSCRIBE_RAM_FRACTION = '0.98'    # Use 98% of RAM (default)
$env:TRANSCRIBE_VRAM_FRACTION = '0.99'   # Use 99% of VRAM (default)
$env:TRANSCRIBE_MAX_PERF = '1'           # Ultra-aggressive mode (default)
```

**Text Processing**
```powershell
# Enhanced text processing is enabled by default
# Uses parallel processing with up to 8 worker threads
# 6-pass enhancement for maximum quality
#   Pass 1: Basic punctuation restoration
#   Pass 2: Advanced sentence segmentation  
#   Pass 3: Capitalization and proper nouns
#   Pass 4: Grammar and style improvements
#   Pass 5: Final cleanup and formatting
#   Pass 6: Global coherence and quality check
```

## Performance

**Typical Processing Speeds on Windows x64:**
- **High-end system** (RTX 4080+ / 32GB RAM): 30-40x realtime
- **Mid-range system** (GTX 1070+ / 16GB RAM): 20-30x realtime  
- **CPU-only system** (modern 8+ core): 10-15x realtime

**Resource Utilization:**
- **RAM**: 98% utilization (ultra-aggressive mode)
- **VRAM**: 99% utilization with memory pooling
- **CPU**: Up to 100% cores for Whisper + 75% for text processing
- **Storage**: Outputs saved next to source files

**Quality Features:**
- Whisper large-v3-turbo model (English optimized)
- Multi-pass punctuation and grammar enhancement
- Intelligent paragraph segmentation
- Automatic artifact removal
- Readability optimization

## Output

Creates two files next to your source file:
- **`filename.txt`** - Clean text transcript
- **`filename.docx`** - Professional Word document with metadata

**Example:**
```
input:  C:\Videos\meeting.mp4
output: C:\Videos\meeting.txt
        C:\Videos\meeting.docx
```

## System Requirements

- **OS**: Windows 10 x64 (1903+) or Windows 11 x64
- **RAM**: 8GB minimum, 16GB+ recommended
- **CPU**: Intel/AMD x64 processor (8+ cores recommended)  
- **GPU**: Optional NVIDIA/AMD GPU (significant performance boost)
- **Storage**: 2GB free space for models and cache
- **Network**: Required for initial setup

## Troubleshooting

**Installation Issues:**
- Run PowerShell as Administrator
- Ensure stable internet connection
- Check Windows is up to date

**Performance Issues:**
- Close other applications for maximum resources
- Enable VAD for speech-heavy content
- Check GPU drivers are current

**Output Issues:**
- Ensure sufficient disk space at destination
- Check file permissions in output directory
- Verify input file is not corrupted
