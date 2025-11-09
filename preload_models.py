#!/usr/bin/env python3
"""
Preload Whisper models to avoid first-run download delays.
This script downloads and caches large-v3-turbo, large-v3 and also the base medium model (for the GUI alias 'turbo-medium-v3').

Notes:
 - 'turbo-medium-v3' is treated as an alias that internally loads the 'medium' model (official Whisper model).
 - If in the future an actual 'turbo-medium-v3' variant is published and appears in whisper.available_models(), it will be preloaded directly.
"""

import os
import sys
import logging

def preload_whisper_models():
    """Download and cache large/medium Whisper models including alias support for 'turbo-medium-v3'."""
    try:
        import whisper
        import torch
        
        # Get available models
        try:
            avail = set(whisper.available_models())
        except Exception:
            avail = set()
        
        # Get cache directory
        cache_dir = torch.hub.get_dir()
        print(f"Model cache directory: {cache_dir}\n")
        
        # Models to preload (in priority order)
        models_to_load = []
        # Prefer turbo large first
        if "large-v3-turbo" in avail:
            models_to_load.append("large-v3-turbo")
        # Then accuracy-focused large
        if "large-v3" in avail:
            models_to_load.append("large-v3")
        # Alias: turbo-medium-v3 -> medium (if the alias itself isn't a real model yet)
        if "turbo-medium-v3" in avail:
            models_to_load.append("turbo-medium-v3")
        elif "medium" in avail:
            models_to_load.append("medium")
        # Legacy large fallback
        if not models_to_load and "large" in avail:
            models_to_load.append("large")
        
        if not models_to_load:
            print("⚠ Warning: No large models found in available models list")
            return False
        
        print(f"Preloading {len(models_to_load)} Whisper model(s)...")
        if "medium" in models_to_load:
            print("🔁 'turbo-medium-v3' alias will use cached 'medium' model.")
        
        # Load each model
        success_count = 0
        for model_name in models_to_load:
            try:
                print(f"\n[{success_count + 1}/{len(models_to_load)}] Loading {model_name}...")
                # Map alias if needed
                load_name = model_name
                if model_name == "turbo-medium-v3" and "turbo-medium-v3" not in avail:
                    load_name = "medium"
                model = whisper.load_model(load_name)
                print(f"✓ Successfully preloaded: {model_name}")
                del model  # Free memory between loads
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to preload {model_name}: {e}")
        
        if success_count > 0:
            print(f"\n✓ Successfully preloaded {success_count}/{len(models_to_load)} model(s)")
            return True
        else:
            print("\n❌ Failed to preload any models")
            return False
        
    except ImportError:
        print("❌ Error: whisper package not installed. Install requirements.txt first.")
        return False
    except Exception as e:
        print(f"❌ Error during model preloading: {e}")
        return False

if __name__ == "__main__":
    success = preload_whisper_models()
    if not success:
        sys.exit(1)
    print("\nModel preloading complete!")
