#!/usr/bin/env python3
"""
Clear Whisper Model Cache Script

This script clears the Whisper model cache to force fresh downloads.
Useful when you want to ensure the correct model is loaded instead of using cached versions.

Usage:
    python clear_whisper_cache.py
"""

import os
import shutil
import sys
import torch
from pathlib import Path

def clear_whisper_cache():
    """Clear Whisper model cache from torch.hub directory."""
    print("🔧 Clearing Whisper model cache...")

    # Get torch.hub cache directory
    hub_cache = Path(torch.hub.get_dir())

    # Whisper models are typically cached in torch/hub/checkpoints
    checkpoints_dir = hub_cache / "checkpoints"

    if not checkpoints_dir.exists():
        print("ℹ️  No Whisper cache directory found.")
        return False

    # Look for Whisper model files
    whisper_files = []
    for file_path in checkpoints_dir.glob("*"):
        if file_path.is_file():
            filename = file_path.name.lower()
            # Whisper model files typically contain "whisper" or have .pt extension
            if "whisper" in filename or filename.endswith(".pt"):
                whisper_files.append(file_path)

    if not whisper_files:
        print("ℹ️  No Whisper model files found in cache.")
        return False

    print(f"📁 Found {len(whisper_files)} Whisper model files:")
    for file_path in whisper_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(".1f")

    # Ask for confirmation
    print("\n⚠️  This will delete the cached Whisper models.")
    print("   Next transcription will download fresh models (may take longer).")
    response = input("   Continue? (y/N): ").strip().lower()

    if response not in ['y', 'yes']:
        print("❌ Cache clearing cancelled.")
        return False

    # Delete the files
    deleted_count = 0
    for file_path in whisper_files:
        try:
            file_path.unlink()
            print(f"🗑️  Deleted: {file_path.name}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Failed to delete {file_path.name}: {e}")

    print(f"\n✅ Successfully deleted {deleted_count} cached model files.")
    print("   Next transcription will download fresh models.")
    return True

def main():
    """Main function."""
    print("Whisper Model Cache Cleaner")
    print("=" * 40)

    try:
        success = clear_whisper_cache()
        if success:
            print("\n🎉 Cache cleared successfully!")
            print("   Run your transcription again to download fresh models.")
        else:
            print("\nℹ️  No action taken.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
