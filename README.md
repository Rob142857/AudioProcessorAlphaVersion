# 🎙️ Professional Speech-to-Text Transcription Tool

**Transform any audio or video into perfect text documents in minutes, not hours.**

This program uses cutting-edge AI to convert your recordings into professionally formatted Word documents with stunning accuracy. Whether you're transcribing interviews, lectures, meetings, podcasts, or YouTube videos, this tool delivers broadcast-quality transcriptions with zero manual effort.

## ✨ What Makes This Special

**🧠 Advanced AI Processing**
- Uses OpenAI's state-of-the-art Whisper AI models
- Optimized parameters bypass content filtering for superior quality
- Handles music, background noise, multiple speakers, and accents effortlessly
- Smart punctuation restoration creates proper sentences and paragraphs

**⚡ Blazing Fast Performance**
- **Auto Mode**: Intelligent optimization using 80-90% of available CPU/GPU resources
- **OPTIMISED Strategy**: Maximum hardware utilization with 1 GPU + 26 CPU workers
- **Optimized Mode**: Enforced GPU + CPU hybrid processing (20-40x realtime)
- **CUDA Acceleration**: NVIDIA GPU processing (5-15x realtime) 
- **CPU Processing**: Universal compatibility with intelligent threading
- Smart preprocessing eliminates noise and normalizes audio automatically

**📊 Real-Time Progress Tracking**
- **Live Progress Bar**: Visual completion percentage with smooth updates
- **ETA Calculator**: Intelligent time-to-completion estimates based on actual processing speed
- **Thread Monitoring**: Real-time display of active worker threads
- **Performance Metrics**: Speed multiplier calculations and processing statistics
- **Smart Log Output**: Clean, readable formatting with proper message parsing

**🎯 Professional Output**
- Clean, formatted Word documents (.docx) with proper headings
- Plain text files (.txt) for universal compatibility
- Intelligent paragraph breaks and sentence structure
- Processing metadata and timestamps included

**🔧 Intelligent Audio Processing**
- **Voice Activity Detection**: Automatically segments audio for parallel processing
- **Audio Enhancement**: Noise reduction, volume normalization, format optimization
- **Content Filtering Bypass**: Transcribes audio with music or copyright content perfectly
- **Multi-format Support**: Handles virtually any audio/video format

**🛠️ Troubleshooting Tools**
- **Built-in Troubleshooting Mode**: Test different transcription approaches automatically
- **VAD vs No-VAD Comparison**: Compare segmented vs continuous transcription
- **Model Size Testing**: Test Medium vs Large models for quality differences
- **Detailed Logging**: Comprehensive logs show processing steps and segment information

## 📁 Supported Files

**Audio Formats:** MP3, WAV, FLAC, M4A, AAC, OGG, WMA  
**Video Formats:** MP4, AVI, MKV, MOV, WMV, FLV, WebM  
**Sources:** Local files, recordings, downloads, streaming captures

## 🚀 Installation (One Command - Works on Any Windows PC)

### For Everyone (CPU Processing)
**Copy and paste this command into PowerShell or Command Prompt:**

```powershell
# PowerShell (Recommended)
irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.ps1 | iex
```

```cmd
# Command Prompt Alternative
curl -L https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install.bat -o install.bat && install.bat
```

**This automatically:**
- Installs Python and all dependencies
- Downloads the program to your Downloads folder  
- Installs the Medium AI model (~1.4GB)
- Creates desktop shortcuts for easy access
- Sets up CPU processing (works on any computer)

### ⚡ GPU Acceleration Setup (Optional - For 3-40x Speed Boost)

**After the basic installation, choose your GPU type:**

#### NVIDIA GPU Users (GTX 1060+, RTX Series)
```powershell
# Run this in the program folder after basic install
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Requirements:**
- NVIDIA GeForce GTX 1060 or newer
- RTX series (any model)
- Latest drivers from [nvidia.com/drivers](https://nvidia.com/drivers)

#### AMD/Intel GPU Users (DirectML)
```powershell
# Run this in the program folder after basic install
.\.venv\Scripts\Activate.ps1
python -m pip install torch-directml
```

**Requirements:**
- AMD Radeon RX series
- Intel Arc or Iris Xe graphics
- Windows 10/11 with latest graphics drivers

## 🎮 How to Use

### Quick Start
1. **Run the installer** command above
2. **Launch the program**: Double-click `launch_gui.bat` or the desktop shortcut
3. **Select your file**: Click "Browse" and choose any audio/video file
4. **Choose settings**: 
   - **Model**: Medium (faster) or Large (best quality)
   - **Processing**: Auto (intelligent), Optimized (GPU+CPU), or CPU only
5. **Click "Start Transcription"** and watch real-time progress:
   - **Progress Bar**: Visual completion percentage
   - **ETA Display**: Intelligent time estimates
   - **Thread Counter**: Live worker monitoring
   - **Speed Metrics**: Processing performance statistics
6. **Find your files** in Downloads folder with perfect formatting

### Processing Modes Explained

**🤖 Auto (Best possible)** - Intelligent system analysis, automatically uses 80-90% of your CPU/GPU for maximum performance with real-time ETA tracking  
**⚡ Optimized (CPU + GPU enforced)** - Forces hybrid GPU+CPU processing for consistent high-speed transcription with progress monitoring  
**🖥️ CPU only** - Pure CPU processing with optimizations, works on any computer

## 📊 Performance Guide

**Real-world processing times for a 30-minute audio file:**

| Processing Mode | Hardware Example | Time | Speed | Features |
|----------------|------------------|------|--------|----------|
| CPU only | Any modern computer | ~45 minutes | 0.7x realtime | Basic progress |
| Auto (GPU detected) | GTX 1070 Ti | ~6 minutes | 5x realtime | ETA + threads |
| OPTIMISED | GTX 1070 Ti + 32 CPU cores | ~2 minutes | 15x realtime | Full monitoring |

**Quality Comparison:**
- **Medium Model**: Excellent quality, faster processing, good for most content
- **Large Model**: Professional-grade accuracy, handles difficult audio perfectly

## 🎯 Advanced Features

**🔧 Always-On Quality Settings:**
- Audio preprocessing (noise reduction, normalization)
- Voice Activity Detection (smart audio segmentation)
- Punctuation restoration (proper sentences and paragraphs)  
- Optimized AI parameters (captures all speech, ignores music filtering)
- Clean startup (suppressed deprecated package warnings)

**📊 Enhanced User Experience:**
- Real-time progress tracking with visual progress bar
- Intelligent ETA calculations based on processing speed
- Live thread monitoring showing active workers
- Clean, formatted log output for easy monitoring
- Professional GUI layout with proper spacing

**📝 Professional Output:**
- Formatted Word documents with metadata
- Processing time and performance statistics
- Intelligent paragraph breaking
- Clean text files for further editing

**🎵 Handles Challenging Audio:**
- Music playing in background
- Multiple speakers and accents
- Phone recordings and poor audio quality
- Copyright content and filtered audio

## 🔧 Troubleshooting

**Installation Issues:**
- **"Command not recognized"**: Make sure you're using PowerShell or Command Prompt as Administrator
- **Download fails**: Check internet connection and Windows security settings
- **Python errors**: The installer handles everything - just run the command again
- **webrtcvad build error**: This package requires Visual Studio Build Tools. The app will work without it using simple duration-based segmentation instead of Voice Activity Detection

**Alternative Installation (if webrtcvad fails):**
```cmd
# Use this if the main installer fails due to build tools
curl -L https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install_simple.bat -o install_simple.bat && install_simple.bat
```

**Performance Issues:**
- **Out of memory**: Switch to Medium model or CPU processing
- **GPU not detected**: Install latest graphics drivers and restart
- **Slow processing**: Enable GPU acceleration with the optional GPU setup commands
- **Progress tracking issues**: Modern GUI includes real-time ETA and thread monitoring

**Quality Issues:**
- **Missing speech**: Try Large model for difficult audio
- **Wrong language**: The AI auto-detects language, but works best with English content
- **Poor punctuation**: This is automatically optimized - output should have proper formatting
- **Transcripts too short**: Use Troubleshooting mode to identify the issue

## 🛠️ Troubleshooting Short Transcripts

If your transcripts are too short or missing content, use the built-in troubleshooting tools:

### Quick Troubleshooting (GUI)
1. **Launch the GUI**: Double-click `launch_gui.bat`
2. **Select your file**: Click "Browse" and choose your audio/video file
3. **Choose "Troubleshoot" mode**: In the Processing dropdown, select "Troubleshoot"
4. **Click "Start Transcription"**: This will run 4 different tests automatically
5. **Compare results**: Check your Downloads folder for files ending in:
   - `_transcription.txt` (VAD + Medium model)
   - `_no_vad_transcription.txt` (No VAD + Medium model) 
   - Large model versions with similar suffixes

### Command-Line Troubleshooting
```cmd
# Run troubleshooting on your audio file
troubleshoot.bat "path\to\your\audio_file.mp3"
```

### Common Issues & Solutions

**VAD Too Aggressive (Most Common)**
- **Symptom**: No-VAD versions are much longer than VAD versions
- **Solution**: Use "No VAD" transcription or adjust VAD settings

**Model Size Issue**
- **Symptom**: Large model versions are significantly longer
- **Solution**: Always use Large model for best quality

**Audio Quality Issues**
- **Symptom**: All methods produce short transcripts
- **Solution**: Check audio quality, try different preprocessing

### Manual Testing
```bash
# Test without VAD segmentation
python transcribe.py "your_file.mp3" --model large

# Test with VAD disabled
python -c "from transcribe import transcribe_file_no_vad; transcribe_file_no_vad('your_file.mp3', model_name='large')"
```

## 💡 Pro Tips

- **Long files (>30 min)**: Use Auto mode for intelligent optimization with ETA tracking
- **Poor audio quality**: Choose Large model for better accuracy  
- **Multiple files**: Process them one at a time for best results with full progress monitoring
- **Best quality**: Use Large model + Auto processing for optimal performance
- **Monitor progress**: Watch the progress bar, ETA estimates, and active thread count
- **System utilization**: OPTIMISED mode targets 80-90% hardware utilization
- **Keep originals**: The program never modifies your original files

## 🗂️ File Locations

**Program**: `%USERPROFILE%\Downloads\speech2textrme`  
**Output files**: `%USERPROFILE%\Downloads\` (your Downloads folder)  
**AI models**: `%USERPROFILE%\.cache\whisper\` (automatically managed)

## 🆘 Need Help?

**Common Questions:**
- Files appear in your Downloads folder with "_transcription", "_auto", or "_optimised" suffixes
- Processing time varies greatly based on audio length and hardware
- The first run downloads AI models, subsequent runs are much faster
- All processing happens locally - your audio never leaves your computer
- Progress bar and ETA estimates provide real-time feedback during transcription
- Thread monitoring shows system utilization and worker activity

**Still stuck?** Open an issue on the GitHub repository with your error message and system details.

---

*Transform your audio into perfect text documents with the power of professional AI transcription. Fast, accurate, and incredibly easy to use.*
