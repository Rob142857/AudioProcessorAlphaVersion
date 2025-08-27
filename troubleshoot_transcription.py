#!/usr/bin/env python3
"""
Troubleshooting script for transcription issues.
This script tests different transcription approaches to help identify why transcripts are too short.
"""

import os
import sys
import argparse
from pathlib import Path

def run_troubleshooting_test(input_file, output_dir=None):
    """Run multiple transcription tests to compare results."""

    if not os.path.isfile(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return

    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    print("🔧 TRANSCRIPTION TROUBLESHOOTING")
    print(f"📁 Input: {os.path.basename(input_file)}")
    print(f"📂 Output: {output_dir}")
    print()

    # Import required functions
    try:
        from transcribe import transcribe_file, transcribe_file_no_vad
    except ImportError as e:
        print(f"❌ Error importing transcription functions: {e}")
        return

    # Test 1: Original VAD-based transcription
    print("🧪 Test 1: Original VAD-based transcription")
    try:
        result1 = transcribe_file(input_file, model_name="medium", output_dir=output_dir)
        print(f"✅ VAD transcription completed: {os.path.basename(result1)}")

        # Read and analyze the result
        with open(result1, 'r', encoding='utf-8') as f:
            text1 = f.read()
        print(f"📊 VAD result: {len(text1)} characters, {len(text1.split())} words")
        print()

    except Exception as e:
        print(f"❌ VAD transcription failed: {e}")
        text1 = ""
        print()

    # Test 2: No VAD transcription
    print("🧪 Test 2: No VAD transcription (entire file as one)")
    try:
        result2 = transcribe_file_no_vad(input_file, model_name="medium", output_dir=output_dir)
        print(f"✅ No-VAD transcription completed: {os.path.basename(result2)}")

        # Read and analyze the result
        with open(result2, 'r', encoding='utf-8') as f:
            text2 = f.read()
        print(f"📊 No-VAD result: {len(text2)} characters, {len(text2.split())} words")
        print()

    except Exception as e:
        print(f"❌ No-VAD transcription failed: {e}")
        text2 = ""
        print()

    # Test 3: Large model with VAD
    print("🧪 Test 3: Large model with VAD")
    try:
        result3 = transcribe_file(input_file, model_name="large", output_dir=output_dir)
        print(f"✅ Large model transcription completed: {os.path.basename(result3)}")

        # Read and analyze the result
        with open(result3, 'r', encoding='utf-8') as f:
            text3 = f.read()
        print(f"📊 Large model result: {len(text3)} characters, {len(text3.split())} words")
        print()

    except Exception as e:
        print(f"❌ Large model transcription failed: {e}")
        text3 = ""
        print()

    # Test 4: Large model without VAD
    print("🧪 Test 4: Large model without VAD")
    try:
        result4 = transcribe_file_no_vad(input_file, model_name="large", output_dir=output_dir)
        print(f"✅ Large model (no VAD) transcription completed: {os.path.basename(result4)}")

        # Read and analyze the result
        with open(result4, 'r', encoding='utf-8') as f:
            text4 = f.read()
        print(f"📊 Large model (no VAD) result: {len(text4)} characters, {len(text4.split())} words")
        print()

    except Exception as e:
        print(f"❌ Large model (no VAD) transcription failed: {e}")
        text4 = ""
        print()

    # Summary
    print("📋 SUMMARY")
    print("=" * 50)

    results = [
        ("VAD + Medium", text1),
        ("No VAD + Medium", text2),
        ("VAD + Large", text3),
        ("No VAD + Large", text4)
    ]

    for name, text in results:
        if text:
            char_count = len(text)
            word_count = len(text.split())
            print("25")
        else:
            print("25")

    # Find the best result
    valid_results = [(name, text) for name, text in results if text]
    if valid_results:
        best_name, best_text = max(valid_results, key=lambda x: len(x[1]))
        print()
        print(f"🎯 Best result: {best_name} ({len(best_text)} characters)")
        print("💡 Recommendation: Use this approach for your transcriptions")
    print()
    print("🔍 TROUBLESHOOTING TIPS:")
    print("   • If No VAD versions are much longer, VAD is too optimised")
    print("   • If Large model versions are much longer, use Large model")
    print("   • Check the actual content of files to see what was captured")
    print("   • Look at the log files for segment information")

def main():
    parser = argparse.ArgumentParser(description="Troubleshoot transcription issues")
    parser.add_argument("input", help="Input audio/video file to test")
    parser.add_argument("--output-dir", help="Output directory (default: Downloads)")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    run_troubleshooting_test(args.input, args.output_dir)

if __name__ == "__main__":
    main()
