#!/usr/bin/env python3
"""Quick CUDA transcription test"""

from transcribe import transcribe_file
import sys
import time

# Create a small test audio if needed
test_file = r"C:\Windows\Media\tada.wav"  # Windows system sound

print("🎮 CUDA Transcription Test")
print("=" * 40)
print(f"Test file: {test_file}")

if len(sys.argv) > 1:
    test_file = sys.argv[1]

try:
    start_time = time.time()
    result = transcribe_file(
        test_file, 
        model_name="base",
        device_preference="cuda",  # Force CUDA
        output_dir="."
    )
    end_time = time.time()
    
    print(f"✅ CUDA Transcription completed!")
    print(f"⚡ Time: {end_time - start_time:.2f} seconds")
    print(f"📝 Output: {result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Make sure the audio file exists and CUDA is working")

print("\n💡 Tips:")
print("  - In GUI: Set device to 'cuda' or 'auto'")
print("  - Monitor GPU usage in Task Manager → Performance → GPU")
print("  - CUDA should be 2-5x faster than CPU for larger files")
