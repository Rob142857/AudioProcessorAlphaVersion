#!/usr/bin/env python3
"""
Test script for lecture transcription to diagnose why 2.5 hour lectures produce very little content.
This script tests the lecture-optimized transcription function.
"""

import os
import sys

def test_lecture_transcription(input_file, output_dir=None):
    """Test the lecture transcription function."""

    if not os.path.isfile(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return

    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    print("🎓 LECTURE TRANSCRIPTION TEST")
    print(f"📁 Input: {os.path.basename(input_file)}")
    print(f"📂 Output: {output_dir}")
    print()

    # Import the lecture transcription function
    try:
        from transcribe import transcribe_lecture
    except ImportError as e:
        print(f"❌ Error importing lecture transcription function: {e}")
        return

    # Get file size and duration info
    file_size = os.path.getsize(input_file) / (1024**3)  # GB
    print(f"📊 File size: {file_size:.2f} GB")

    try:
        from moviepy import VideoFileClip, AudioFileClip
        ext = os.path.splitext(input_file)[1].lower()
        duration = None
        if ext in [".mp4", ".mov", ".mkv", ".avi"]:
            clip = VideoFileClip(input_file)
            duration = clip.duration
            clip.close()
        else:
            clip = AudioFileClip(input_file)
            duration = clip.duration
            clip.close()

        if duration:
            print(f"⏱️  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    except Exception as e:
        print(f"⚠️  Could not get duration: {e}")
        duration = None

    print()
    print("🧪 Testing lecture-optimized transcription...")

    try:
        result_file = transcribe_lecture(
            input_file,
            model_name="medium",
            output_dir=output_dir,
            preprocess=True,
            punctuate=True
        )

        print(f"\n✅ Lecture transcription completed!")
        print(f"📄 Output file: {result_file}")

        # Analyze the result
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"\n📊 RESULTS ANALYSIS:")
        print(f"   • Characters: {len(content):,}")
        print(f"   • Words: {len(content.split()):,}")
        print(f"   • Lines: {len(content.split(chr(10))):,}")
        print(f"   • Paragraphs: {len([p for p in content.split(chr(10)*2) if p.strip()]):,}")

        # Estimate speaking rate
        try:
            if duration and duration > 0:
                words_per_minute = len(content.split()) / (duration / 60)
                print(f"   • Speaking rate: {words_per_minute:.0f} words per minute")
        except:
            pass

        # Show first 500 characters as preview
        print(f"\n📝 CONTENT PREVIEW (first 500 chars):")
        print("-" * 50)
        preview = content[:500].replace('\n', ' ')
        print(f"   {preview}{'...' if len(content) > 500 else ''}")
        print("-" * 50)

        # Check for common issues
        print(f"\n🔍 DIAGNOSTICS:")
        if len(content) < 1000:
            print("   ⚠️  WARNING: Very little content transcribed!")
            print("   💡 This suggests VAD is too optimised or audio quality issues")

        if "No speech detected" in content:
            print("   ❌ ISSUE: Whisper detected no speech at all")

        if len(content.split()) < 100:
            print("   ⚠️  WARNING: Less than 100 words transcribed")
            print("   💡 Try: No-VAD mode, Large model, or check audio quality")

        print(f"\n📋 RECOMMENDATIONS:")
        if len(content) < 1000:
            print("   • Try 'No VAD' mode in the GUI (disables voice activity detection)")
            print("   • Use 'Large' model instead of 'Medium' for better accuracy")
            print("   • Check audio quality - ensure clear speech without heavy background noise")
            print("   • Try the troubleshooting mode to compare different approaches")

        print(f"\n📄 Full transcription saved to: {result_file}")

    except Exception as e:
        print(f"\n❌ Lecture transcription failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_lecture_transcription.py <input_file>")
        print("Example: python test_lecture_transcription.py my_lecture.mp4")
        sys.exit(1)

    input_file = sys.argv[1]
    test_lecture_transcription(input_file)

if __name__ == "__main__":
    main()
