#!/usr/bin/env python3
"""
Preload Whisper models to avoid first-run download delays.
This script downloads and caches the preferred turbo model during installation.
"""

import os
import sys
import logging

def preload_large_model():
    """Download and cache the preferred Whisper model (large-v3-turbo if available)."""
    try:
        print("Preloading Whisper model (preferring large-v3-turbo)...")
        import whisper
        
        # Prefer turbo; fall back gracefully
        try:
            avail = set(whisper.available_models())
        except Exception:
            avail = set()
        model_name = "large-v3-turbo" if "large-v3-turbo" in avail else ("large-v3" if "large-v3" in avail else "large")

        # Load the model (this will download and cache it)
        model = whisper.load_model(model_name)
        print(f"✓ Successfully preloaded Whisper model: {model_name}")
        
        # Get cache directory
        import torch
        cache_dir = torch.hub.get_dir()
        print(f"  Model cache directory: {cache_dir}")
        
        return True
        
    except ImportError:
        print("❌ Error: whisper package not installed. Install requirements.txt first.")
        return False
    except Exception as e:
        print(f"❌ Error preloading model: {e}")
        return False

if __name__ == "__main__":
    success = preload_large_model()
    if not success:
        sys.exit(1)
    print("Model preloading complete!")
