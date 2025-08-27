#!/usr/bin/env python3
"""
Test Model Loading Script

This script tests whether the correct Whisper model is loaded based on the request.
Useful for debugging model selection issues.

Usage:
    python test_model_loading.py [model_name]
"""

import sys
import torch
import whisper

def test_model_loading(requested_model="large"):
    """Test loading a specific Whisper model and verify it's correct."""
    print(f"Testing Whisper model loading for: {requested_model}")
    print("=" * 50)

    try:
        # Load the requested model
        print(f"Loading model '{requested_model}'...")
        model = whisper.load_model(requested_model)

        # Get model information
        model_size = sum(p.numel() for p in model.parameters())
        print(f"Model size: {model_size:,} parameters")

        # Estimate model type based on size
        if model_size > 700000000:  # Large model has ~1.5B parameters
            detected_model = "large"
        elif model_size > 350000000:  # Medium model has ~770M parameters
            detected_model = "medium"
        elif model_size > 70000000:  # Small model has ~244M parameters
            detected_model = "small"
        elif model_size > 35000000:  # Base model has ~74M parameters
            detected_model = "base"
        else:
            detected_model = "tiny"

        print(f"Detected model type: {detected_model}")
        print(f"Requested model: {requested_model}")

        # Check if correct model was loaded
        if detected_model == requested_model:
            print("✅ SUCCESS: Correct model loaded!")
            return True
        else:
            print(f"❌ ISSUE: Requested '{requested_model}' but got '{detected_model}'")
            print("   This indicates a cached model is being used instead of the requested one.")
            print("   To fix: Run 'python clear_whisper_cache.py' to clear the cache.")
            return False

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        model_to_test = sys.argv[1]
    else:
        model_to_test = "large"  # Default to large since that's what the user was trying to use

    success = test_model_loading(model_to_test)

    if not success:
        print("\n🔧 Troubleshooting steps:")
        print("1. Run: python clear_whisper_cache.py")
        print("2. Run this test again: python test_model_loading.py large")
        print("3. If still failing, check your PyTorch installation")

if __name__ == "__main__":
    main()
