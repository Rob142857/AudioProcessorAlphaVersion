#!/usr/bin/env python3
"""Quick CUDA test for Whisper with GTX 1070 Ti"""

import torch
import whisper

def test_cuda_whisper():
    print("=== CUDA Whisper Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print("\nTesting Whisper model loading...")
        try:
            # Load small model to test quickly
            model = whisper.load_model("base", device="cuda")
            print("✓ Whisper model loaded successfully on CUDA!")
            print(f"Model device: {next(model.parameters()).device}")
            
            # Test a simple tensor operation on GPU
            test_tensor = torch.randn(100, 100, device="cuda")
            result = torch.mm(test_tensor, test_tensor)
            print(f"✓ GPU tensor test successful: {result.shape}")
            
        except Exception as e:
            print(f"✗ Error loading Whisper on CUDA: {e}")
            print("  Trying CPU fallback...")
            model = whisper.load_model("base", device="cpu")
            print("✓ Whisper loaded on CPU as fallback")
    else:
        print("✗ CUDA not available")

if __name__ == "__main__":
    test_cuda_whisper()
