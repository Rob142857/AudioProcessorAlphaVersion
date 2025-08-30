#!/usr/bin/env python3
"""
Preload Whisper models to avoid first-run download delays.
This script downloads and caches the large model during installation.
"""

import os
import sys
import logging

def preload_large_model():
    """Download and cache the Whisper large model."""
    try:
        print("📥 Downloading Whisper Large model (3GB)...")
        print("   This may take several minutes depending on your internet speed...")
        import whisper
        
        # Load the large model (this will download and cache it)
        print("   Loading Large model...")
        model = whisper.load_model("large")
        print("✅ Successfully downloaded and cached Whisper Large model")
        
        # Get cache directory
        import torch
        cache_dir = torch.hub.get_dir()
        print(f"   Model cache location: {cache_dir}")
        
        return True
        
    except ImportError:
        print("❌ Error: whisper package not installed. Install requirements.txt first.")
        return False
    except Exception as e:
        print(f"❌ Error downloading Large model: {e}")
        return False

if __name__ == "__main__":
    success = preload_large_model()
    if not success:
        sys.exit(1)
    print("🎉 Large model download complete! Ready for transcription.")
