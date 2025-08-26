#!/usr/bin/env python3
"""Monitor GPU usage during transcription"""

import psutil
import time
import subprocess
import sys

def monitor_gpu():
    print("🎮 GPU Monitoring (GTX 1070 Ti)")
    print("=" * 40)
    
    try:
        # Try nvidia-smi if available
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"GPU Status: {gpu_info}")
        else:
            print("nvidia-smi not available, use Task Manager to monitor GPU usage")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("📊 Monitor GPU usage in Task Manager:")
        print("   1. Open Task Manager (Ctrl+Shift+Esc)")
        print("   2. Go to Performance tab")
        print("   3. Select 'GPU 1' (your GTX 1070 Ti)")
        print("   4. Watch for activity during transcription")

if __name__ == "__main__":
    monitor_gpu()
