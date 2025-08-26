#!/usr/bin/env python3
"""Simple CUDA vs CPU performance test"""

import torch
import whisper
import time

def test_cuda_performance():
    print("=== CUDA Performance Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Test tensor operations on GPU vs CPU
    print("\n🧪 Testing tensor operations...")
    
    # GPU test
    torch.cuda.synchronize()
    start = time.time()
    gpu_tensor = torch.randn(1000, 1000, device="cuda")
    gpu_result = torch.mm(gpu_tensor, gpu_tensor)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # CPU test  
    start = time.time()
    cpu_tensor = torch.randn(1000, 1000, device="cpu")
    cpu_result = torch.mm(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start
    
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"GPU speedup: {cpu_time/gpu_time:.1f}x faster")
    
    print("\n🎤 For Whisper transcription:")
    print("  - Use device='cuda' for GPU acceleration")
    print("  - Use device='cpu' for CPU-only")
    print("  - Use device='auto' to automatically choose GPU")

if __name__ == "__main__":
    test_cuda_performance()
