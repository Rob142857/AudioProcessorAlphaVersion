# Whisper.cpp ARM Transcription App

This is a high-quality transcription tool for ARM64, using Whisper.cpp (C++), with support for large models, punctuation restoration, and AI guardrails (mute copyright, filter swearing, etc.).

## Features
- Transcribe individual files or entire folders
- Uses Whisper.cpp for transcription (large model support)
- Punctuation restoration via Hugging Face ONNX or pure Python model
- AI guardrails: mute copyright audio, filter swearing/offensive language
- Modern, simple UI (cross-platform)

## Requirements
- Windows ARM64 (or Linux ARM64)
- whisper.cpp binary (download/build from https://github.com/ggerganov/whisper.cpp)
- ONNX Runtime (for punctuation)
- Python 3.11+ (for orchestration, if using Python)

## Usage
1. Place whisper.cpp binary in this folder (or specify path in config)
2. Run the main script/app
3. Select file or folder for transcription
4. Outputs: .txt, .docx, and optionally .srt

## Next Steps
- Implement main orchestration script
- Integrate punctuation restoration
- Add AI guardrails
- Build simple UI

---

See README.md for more details.
