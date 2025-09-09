# ARM Support Notes

This branch is dedicated to ARM64 compatibility for Windows.

## Issues Detected
- PyTorch and torchaudio wheels for Windows ARM64 are not available via standard channels.
- `webrtcvad` fails to build due to missing ARM architecture support in its C code.
- Whisper and deepmultilingualpunctuation require torch, which cannot be installed.
- The app cannot launch due to missing dependencies.

## Next Steps
- Research and document any available ARM64-compatible wheels for torch/torchaudio.
- Consider using alternative packages or disabling features that require unavailable dependencies.
- Update installation instructions for ARM users.
- Add conditional imports or error handling for missing features.

## Manual Installation Guidance
- Try installing torch with: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- If unavailable, check for community ARM64 wheels or build from source.
- For `webrtcvad`, consider disabling VAD or using a pure Python alternative.

## Status
- Branch created for ARM support. Further commits will address compatibility issues.
