#!/usr/bin/env python3
"""
Test script for dataset-optimized transcription.

This script demonstrates how to use the new dataset-based GPU pipeline optimization
that addresses the PyTorch suggestion: "you seem to be using pipelines in the gpu
in order to maximise efficiency please use a dataset"

Usage:
    python test_dataset_optimization.py --input "path/to/audio.mp4"

Environment Variables:
    TRANSCRIBE_USE_DATASET=1    # Enable dataset optimization
    TRANSCRIBE_MAX_PERF=1       # Enable maximum performance mode
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test dataset-optimized transcription")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--dataset", action="store_true", help="Enable dataset optimization")
    parser.add_argument("--max-perf", action="store_true", help="Enable maximum performance mode")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set environment variables for dataset optimization
    if args.dataset:
        os.environ["TRANSCRIBE_USE_DATASET"] = "1"
        print("🎯 Dataset optimization enabled")

    if args.max_perf:
        os.environ["TRANSCRIBE_MAX_PERF"] = "1"
        print("🚀 Maximum performance mode enabled")

    # Import and run transcription
    try:
        from transcribe_optimised import transcribe_file_simple_auto

        print(f"🔄 Starting transcription of: {os.path.basename(args.input)}")

        result_path = transcribe_file_simple_auto(args.input)

        if result_path:
            print(f"✅ Transcription completed successfully: {result_path}")
            return 0
        else:
            print("❌ Transcription failed")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())