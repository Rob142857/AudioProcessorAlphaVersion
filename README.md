# Speech-to-Text Transcription Project

This project provides a free, local solution for converting audio files (WAV, MP3, etc.) to well-formatted text using OpenAI Whisper.

## Features
- Transcribe audio files to text
- Outputs well-punctuated, readable sentences
- No ongoing costs or cloud dependencies

## Requirements
- Python 3.8+
- pip
- (Recommended) A GPU for faster transcription (CPU also supported)

## Setup
1. Install Python 3.8 or newer from https://www.python.org/
2. (Optional) Set up a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install openai-whisper torch
   ```

## Usage
1. Place your audio file (e.g., `audio.mp3`) in the project folder.
2. Run the transcription script:
   ```powershell
   python transcribe.py audio.mp3
   ```
3. The transcript will be saved as `audio.txt` in the same folder.

## Notes
- For best results, use clear audio recordings.
- Large files may take longer to process, especially on CPU.

## AI agent prerequisites

To run the automated transcription agent locally, ensure the following are installed and configured:

- Python 3.8 or newer
- A virtual environment is recommended. From the project root:

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- FFmpeg must be available. The script will try to use an included binary (if present) or imageio-ffmpeg. For best results, install FFmpeg system-wide and ensure `ffmpeg` is on your PATH. On Windows you can download a build and add it to PATH.

- For higher-performance transcription, install PyTorch with CUDA support matching your GPU. See https://pytorch.org/ for install commands.

- To run the high-quality `medium` or `large` Whisper models you will need sufficient RAM (and preferably a GPU). The `medium` model is ~1.4GB.

If you want me to push this repository to GitHub, I can commit and push using your local git credentials if your environment is configured for it.

---

For help or improvements, just ask Copilot!
