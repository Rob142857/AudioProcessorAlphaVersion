# Speech-to-Text Transcription (local, Whisper-based)

This repository provides a local speech-to-text pipeline using OpenAI Whisper plus optional preprocessing, VAD, punctuation restoration and a small GUI wrapper for ease of use.

## Features
- Transcribe audio/video (MP3, WAV, MP4, MOV, etc.) to plain text and DOCX.
- Optional preprocessing (ffmpeg filters), VAD segmentation, and punctuation restoration.
- Simple Tkinter-based GUI wrapper for single-click use and a headless CLI mode for automation.
- Local-only processing (models downloaded and cached locally).

## Quickstart (AI agent / fresh machine)
Goal: clone the repo, create a venv, install dependencies, and launch the GUI in one sequence.

1) Clone the repo and change into the project folder:

   ```powershell
   git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git .
   cd AudioProcessorAlphaVersion
   ```

2) Create and activate a Python virtual environment (Windows PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3) Install Python dependencies (this will install Whisper, PyTorch, VAD and punctuation model support):

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Notes on PyTorch: If a GPU is available and you want CUDA acceleration, install the appropriate PyTorch build for your CUDA version. Example (CUDA 11.8):

   ```powershell
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4) Ensure FFmpeg is available. The script will try to use an included or imageio-ffmpeg-provided binary, but system FFmpeg is recommended. On Windows you can:
   - Download an FFmpeg build (e.g., gyan.dev or BtbN) and add the `ffmpeg.exe` folder to your PATH, or
   - Install via winget: `winget install gyan.ffmpeg` (if winget is available)

5) Launch the GUI (default output folder is the user's Downloads directory):

   ```powershell
   python gui_transcribe.py
   ```

6) Or run headless (example):

   ```powershell
   python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "C:\path\to\out" --model medium --preprocess --vad --punctuate --keep-temp
   ```

## Files of interest
- `transcribe.py` — main transcription pipeline and CLI.
- `gui_transcribe.py` — GUI and CLI wrapper that calls `transcribe.py`.
- `requirements.txt` — Python dependencies (use `pip install -r requirements.txt`).
- `.gitignore` — project ignores.

## Recommended default run for high quality
- `--preprocess --vad --punctuate --model medium` (medium model ~1.42GB; GPU recommended)

## Notes about ARM / NPU / DirectML (Surface Laptop 7)
- For Windows DirectML (`--device dml`) you must install `torch-directml` and a compatible PyTorch build. On ARM devices with vendor NPUs, vendor-specific runtimes or ONNX conversions may be required. The code will print guidance when `--device dml` is requested but not available.

## Troubleshooting
- Module import errors: ensure you activated the `.venv` and installed `requirements.txt` into it.
- FFmpeg errors: ensure `ffmpeg` is on PATH or imageio-ffmpeg is available.
- If the punctuation model or Whisper model downloads fail, retry; they may be large files and need stable network connectivity.

## Example automation script (PowerShell)
Save this as `run_example.ps1` in the repo to run a headless transcription on a file and collect outputs in Downloads:

```powershell
.\.venv\Scripts\Activate.ps1
python gui_transcribe.py --input "C:\path\to\your.mp4" --outdir "$env:USERPROFILE\Downloads" --model medium --preprocess --vad --punctuate --keep-temp
```

---

If you'd like, I can also:
- Pin exact versions in `requirements.txt` and produce a small install-check script, or
- Add a Windows `run.bat` that prepares the venv and launches the GUI automatically.
