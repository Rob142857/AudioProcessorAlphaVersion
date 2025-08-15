# Speech-to-Text Transcription (local, Whisper-based)

This repository provides a local speech-to-text pipeline using OpenAI Whisper plus optional preprocessing (FFmpeg), VAD segmentation, punctuation restoration, and a small Tkinter GUI wrapper.

This README focuses on a fast, reproducible Windows bootstrap so an AI agent or a human can get the environment created and a single transcription run working with minimal interaction.

## Quick checklist (what you'll do)
- Clone the repo
- Create & activate a Python virtual environment
- Install Python dependencies (requirements + optional PyTorch build)
- Ensure FFmpeg is available
- Run the included `run.bat` (recommended) or launch the GUI / headless CLI

Quickest possible start: PowerShell (interactive)

Open Powershell as Administrator, then:

Copy and paste this multi-line PowerShell snippet into a PowerShell window (not cmd). It will clone or update the repo in your Downloads folder, create and activate a `.venv`, install the pinned non-Torch requirements, and then prompt you to run `./run.bat`.

```powershell
Set-Location "$env:USERPROFILE\Downloads"

if (Test-Path .\speech2textrme) {
	Write-Host 'Using existing folder: speech2textrme'
	Set-Location .\speech2textrme
	if (Get-Command git -ErrorAction SilentlyContinue) { git pull }
} else {
	if (Get-Command git -ErrorAction SilentlyContinue) {
		git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git speech2textrme
	} else {
		$zip = 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip'
		Invoke-WebRequest $zip -OutFile main.zip
		Expand-Archive main.zip -DestinationPath .
		Rename-Item 'AudioProcessorAlphaVersion-main' 'speech2textrme'
	}
	Set-Location .\speech2textrme
}

if (-not (Test-Path .venv)) { python -m venv .venv }

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Bootstrap complete. Please run .\run.bat now to continue (it will offer optional PyTorch/DirectML install)."
Read-Host -Prompt "Press Enter to close"
```

## Fastest path (one-liner for interactive use)
Open PowerShell in the repo root and run the included `run.bat`. It will create the venv, install the pinned dependencies (non-PyTorch), and launch the GUI.

If you prefer to do the steps manually or want full control, follow the commands below.

## Manual setup (recommended for debugging and custom PyTorch)
1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Install Python packages (requirements.txt contains pinned versions for most packages). This installs Whisper, VAD support, punctuation helpers, and utilities. Note: `torch` is intentionally left unpinned — pick the correct build for your hardware.

```powershell
python -m pip install -r requirements.txt
```

3) Install PyTorch (optional but recommended if you have GPU/DirectML support). Select the correct wheel for your machine:

- CPU-only (simple):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- CUDA (example for CUDA 11.8 GPU):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

- DirectML (Windows without CUDA): install `torch-directml` following Microsoft instructions and use `--device dml` when running. We do not auto-install DirectML because versions must match the local PyTorch build.

4) Ensure FFmpeg is available on PATH. The code can fallback to `imageio-ffmpeg` binaries, but system FFmpeg is preferred. On Windows you can install with winget (if available):

```powershell
winget install gyan.ffmpeg
```

Or download a static build and add its folder to your PATH.

## Run the GUI (recommended for first run)
Launch the GUI which will open a simple window for file selection and options:

```powershell
python gui_transcribe.py
```

## Headless example (automation)
Transcribe a single file to your Downloads folder using a higher-quality preset:

```powershell
.\.venv\Scripts\Activate.ps1
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --model medium --preprocess --vad --punctuate --keep-temp
```

## Useful helper scripts
- `run.bat` — Windows bootstrap: creates `.venv`, installs pinned requirements (non-PyTorch), and launches the GUI. Use this first on a fresh Windows VM.
- `check_env.py` — writes `env_report.json` describing Python version, ffmpeg discoverability, Torch version and CUDA availability, and other quick checks.

## Recommended default flags for quality
- `--preprocess --vad --punctuate --model medium`

Notes: the `medium` Whisper model (~1.4GB) and punctuation model weights (~2+GB) will be downloaded on first use.

## Inputs & outputs
- Inputs: MP3, WAV, MP4, MOV (video → audio is extracted automatically).
- Outputs (by default): `.txt` (cleaned & paragraphed), `.docx` (Word, paragraphed), and a per-run JSON log named `<input_basename>_transcription_log.json` which contains timings, options used, and generated file paths.

## Troubleshooting checklist
- If imports fail, confirm the `.venv` is activated and `pip install -r requirements.txt` succeeded.
- If `ffmpeg` errors occur, make sure `ffmpeg.exe` is on PATH or install via winget / add to PATH.
- If model downloads fail, retry on a stable network; these files are large.

## Next steps and optional improvements
- If you want DirectML to be the default on Windows, install `torch` + `torch-directml` manually to avoid version mismatches, then run with `--device dml`.
- For vendor NPUs (ARM), build ONNX exports and vendor runtimes — this is advanced and not included here.

## Contact / Contributions
Open an issue or create a PR with improvements. If you want me to further pin PyTorch builds or add an automated DirectML installer, say which target hardware you want to support and I will add it.
