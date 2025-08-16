<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->
Quick agent-friendly bootstrap for Windows x64 (fast, predictable):

1) From the repo root, run the provided Windows bootstrap which creates a venv, installs requirements and launches the GUI:

```powershell
.\run.bat
```

2) If you need manual control (recommended when selecting PyTorch / DirectML builds), run these steps instead in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3) Install PyTorch matching the target hardware (only if you want a specific build):
- CPU-only: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
- CUDA (example): `python -m pip install torch --index-url https://download.pytorch.org/whl/cu118`
- DirectML: install `torch-directml` manually and use `--device dml`

4) **Preload Whisper models** (recommended to avoid first-run delays):

```powershell
python preload_models.py
```

This downloads and caches the medium model (~1.4GB). The `run.bat` script automatically does this when PyTorch is installed via options 2 or 3.

5) Ensure `ffmpeg` is on PATH. If not available, install via winget or add a static build to PATH.

6) Optional quick environment check (writes `env_report.json`):

```powershell
.\.venv\Scripts\Activate.ps1
python check_env.py
```

7) Launch GUI or run headless (example):

```powershell
python gui_transcribe.py
# headless example:
python gui_transcribe.py --input "C:\path\to\file.mp4" --outdir "$env:USERPROFILE\Downloads" --model medium --preprocess --vad --punctuate --keep-temp
```

Agent notes and constraints:
- Do NOT auto-install `torch` or `torch-directml` unless you have the target CUDA/driver information. Ask the human or document the recommended wheel.
- Model downloads are large (Whisper medium ~1.4GB; punctuation model multi-GB). Cache model folders when running in ephemeral CI.
- Ensure `ffmpeg` is discoverable for reliable audio preprocessing and VAD extraction.

Completion criteria for automated runs:
- `.venv` exists and `pip install -r requirements.txt` succeeded
- `check_env.py` produced `env_report.json` and reported basic system capability
- A test transcription run produced `.txt` and `.docx` outputs

End of instructions.
