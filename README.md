# Audio Transcription Tool**This automatically:**
- Installs Python and dependencies
- Detects your hardware and installs optimal PyTorch
- Downloads the program and Large AI model (~3GB)
- Creates desktop shortcutsonverts audio and video files to text using OpenAI Whisper AI. Supports multiple formats with automatic hardware detection and optimisation.

## What It Does

- **Transcribes audio/video** to text documents (.txt and .docx)
- **Automatic hardware detection** - works on any Windows PC
- **Multiple processing modes**: Auto (intelligent), Optimised (GPU+CPU), CPU-only
- **Real-time progress** with ETA estimates and performance metrics
- **Voice Activity Detection** for better segmentation
- **Punctuation restoration** for readable output

## Supported Formats

**Audio:** MP3, WAV, FLAC, M4A, AAC, OGG, WMA  
**Video:** MP4, AVI, MKV, MOV, WMV, FLV, WebM

## Installation

Run this command in PowerShell:

```powershell
irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex
```

This automatically:
- Installs Python and dependencies
- Detects your hardware and installs optimal PyTorch
- Downloads the program and AI model
- Creates desktop shortcuts

## Hardware Support

**NVIDIA GPUs:** CUDA acceleration (5-15x faster)  
**AMD/Intel GPUs:** DirectML acceleration  
**CPU-only:** Works on any computer

## Usage

1. Run `launch_gui.bat` or desktop shortcut
2. Click "Browse" to select audio/video file
3. Click "Start Transcription"
4. Find output files in Downloads folder

The program automatically uses the Large model for best quality transcription.

## Troubleshooting

**Short transcripts:** Use "Troubleshoot" mode to test different approaches  
**Installation issues:** Run installer as Administrator  
**Performance problems:** Large model requires more resources than alternatives  
**Memory errors:** Use CPU mode or ensure sufficient RAM (8GB+ recommended)

## File Locations

- **Program:** `%USERPROFILE%\Downloads\speech2textrme`
- **Output:** `%USERPROFILE%\Downloads\`
- **Models:** `%USERPROFILE%\.cache\whisper\`

## Requirements

- Windows 10/11
- Internet connection for initial setup
- 8GB+ RAM recommended (Large model requirement)
- GPU optional but recommended for speed
- ~3GB disk space for Large AI model
