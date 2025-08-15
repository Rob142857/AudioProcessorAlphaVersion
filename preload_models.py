#!/usr/bin/env python3
"""
Preload Whisper models to avoid first-run download delays.
This script downloads and caches the medium model during installation.
"""

import os
import sys
import logging

def preload_medium_model():
    """Download and cache the Whisper medium model."""
    try:
        print("Preloading Whisper medium model...")
        import whisper
        
        # Load the medium model (this will download and cache it)
        model = whisper.load_model("medium")
        print(f"✓ Successfully preloaded Whisper medium model")
        
        # Get cache directory
        import torch
        cache_dir = torch.hub.get_dir()
        print(f"  Model cache directory: {cache_dir}")
        
        # Optionally preload large model too if user wants it
        if len(sys.argv) > 1 and sys.argv[1] == "--include-large":
            print("Also preloading Whisper large model...")
            large_model = whisper.load_model("large")
            print(f"✓ Successfully preloaded Whisper large model")
        
        return True
        
    except ImportError:
        print("❌ Error: whisper package not installed. Install requirements.txt first.")
        return False
    except Exception as e:
        print(f"❌ Error preloading model: {e}")
        return False

if __name__ == "__main__":
    success = preload_medium_model()
    if not success:
        sys.exit(1)
    print("Model preloading complete!")
