"""
Optimised transcription using Large model only.
Simplified, reliable transcription with automatic device detection.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

import sys
import os
import shutil
import subprocess
import tempfile
import argparse
import whisper
import torch
from docx import Document
from moviepy import VideoFileClip, AudioFileClip
import imageio_ffmpeg as iio_ffmpeg
from deepmultilingualpunctuation import PunctuationModel
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    print("⚠️  webrtcvad not available - Voice Activity Detection disabled")
import re
import time
import json


def get_media_duration(input_path):
    """Get duration of media file in seconds."""
    try:
        if input_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')):
            clip = VideoFileClip(input_path)
            duration = clip.duration
            clip.close()
            return duration
        elif input_path.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma')):
            clip = AudioFileClip(input_path)
            duration = clip.duration
            clip.close()
            return duration
    except Exception as e:
        print(f"⚠️  Could not get duration: {e}")
    return None


def split_into_paragraphs(text, max_length=500):
    """Split text into paragraphs."""
    if not text:
        return ""

    # Split by double newlines first
    paragraphs = text.split('\n\n')

    result = []
    for para in paragraphs:
        if len(para) <= max_length:
            result.append(para)
        else:
            # Split long paragraphs
            sentences = para.split('. ')
            current_para = ""
            for sentence in sentences:
                if len(current_para) + len(sentence) <= max_length:
                    current_para += sentence + '. '
                else:
                    if current_para:
                        result.append(current_para.strip())
                    current_para = sentence + '. '
            if current_para:
                result.append(current_para.strip())

    return '\n\n'.join(result) if result else text


def format_duration(seconds):
    """Format seconds into HH:MM:SS."""
    if not seconds:
        return "00:00:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return "02d"


def transcribe_file_simple_auto(input_path, output_dir=None):
    """
    Simplified transcription using Large model only.
    - Automatic device detection (CUDA > DirectML > CPU)
    - Large model for best quality
    - No VAD segmentation
    - Basic preprocessing
    """
    start_time = time.time()

    try:
        print("🚀 Starting Large model transcription...")
        print(f"📁 Input: {os.path.basename(input_path)}")

        # Get media duration
        duration = get_media_duration(input_path)
        if duration:
            print(f"⏱️  Duration: {format_duration(duration)}")

        # Device detection
        device = "cpu"
        device_name = "CPU"

        # Try CUDA first
        if torch.cuda.is_available():
            device = "cuda"
            device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
            print(f"🎯 Using CUDA GPU for acceleration")
        else:
            # Try DirectML
            try:
                import torch_directml
                dml_device = torch_directml.device()
                device = dml_device
                device_name = "DirectML GPU"
                print("🎯 Using DirectML GPU for acceleration")
            except ImportError:
                print("📊 Using CPU processing")

        print("📥 Loading Large model...")
        model = whisper.load_model("large", device=device)
        print("✅ Large model loaded successfully")

        if not output_dir:
            output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

        # Transcribe with optimal settings for Large model
        print("🎙️  Transcribing...")
        result = model.transcribe(input_path,
                                language=None,
                                compression_ratio_threshold=float('inf'),
                                logprob_threshold=-1.0,
                                no_speech_threshold=0.1,
                                condition_on_previous_text=False,
                                temperature=0.0)

        # Extract text
        if isinstance(result, dict):
            text_result = result.get("text", "")
            full_text = text_result.strip() if isinstance(text_result, str) else str(text_result).strip()
        else:
            full_text = str(result).strip()

        print(f"📝 Transcription complete: {len(full_text)} characters")

        # Post-processing
        print("📝 Adding punctuation...")
        try:
            punctuation_model = PunctuationModel()
            full_text = punctuation_model.restore_punctuation(full_text)
        except Exception as e:
            print(f"⚠️  Punctuation restoration failed: {e}")

        # Format text
        formatted_text = split_into_paragraphs(full_text, max_length=500)
        if isinstance(formatted_text, list):
            formatted_text = '\n\n'.join(formatted_text)

        # Save outputs
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}_large.txt")
        docx_path = os.path.join(output_dir, f"{base_name}_large.docx")

        # Save text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)

        # Create Word document
        doc = Document()
        doc.add_heading(f'Large Model Transcription: {base_name}', 0)

        if duration:
            doc.add_paragraph(f'Duration: {format_duration(duration)}')

        elapsed = time.time() - start_time
        doc.add_paragraph(f'Processing time: {format_duration(elapsed)}')
        doc.add_paragraph(f'Model: Large')
        doc.add_paragraph(f'Hardware: {device_name}')

        if duration and elapsed > 0:
            speedup = duration / elapsed
            doc.add_paragraph(f'Processing speed: {speedup:.1f}x realtime')

        doc.add_paragraph('')

        for para in formatted_text.split('\n\n'):
            if para.strip():
                doc.add_paragraph(para.strip())

        doc.save(docx_path)

        # Final stats
        print("\n🎉 Transcription complete!")
        print(f"📄 Text file: {txt_path}")
        print(f"📄 Word document: {docx_path}")
        print(f"⏱️  Total time: {format_duration(elapsed)}")
        print(f"🚀 Hardware: {device_name}")

        if duration:
            speedup = duration / elapsed
            print(f"⚡ Speed: {speedup:.1f}x realtime")

        return txt_path

    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Large model transcription")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--output-dir", help="Output directory (default: Downloads)")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    try:
        transcribe_file_simple_auto(args.input, output_dir=args.output_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
