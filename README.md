# AudioProcessor

High-performance audio and video transcription for Windows x64 using OpenAI Whisper with automatic GPU acceleration.

## What It Does

Converts audio/video files into professionally formatted text transcripts. Automatically detects and utilizes NVIDIA GPUs for 3-5x faster processing, with CPU fallback.

## Installation

Run this single command:
`powershell
irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install_geforce.ps1 | iex
`

This installs Python, creates a virtual environment, installs dependencies, downloads models, and launches the GUI.

## Usage

Drag & drop audio/video files into the GUI, or use command line:
`powershell
python gui_transcribe.py --input "audio.mp4"
`

## Requirements

- Windows 10/11 x64
- 8GB+ RAM (16GB recommended)
- Optional: NVIDIA GPU for acceleration
