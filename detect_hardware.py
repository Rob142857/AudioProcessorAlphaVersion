#!/usr/bin/env python3
"""
Hardware Detection Script for AudioProcessor
Detects GPU hardware and recommends optimal PyTorch installation
"""

import subprocess
import sys
import platform

def run_command(cmd):
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except:
        return False, '', ''

def detect_nvidia_gpu():
    """Check for NVIDIA GPU using nvidia-smi"""
    success, stdout, stderr = run_command('nvidia-smi --query-gpu=name --format=csv,noheader,nounits')
    if success and stdout.strip():
        gpu_name = stdout.strip().split('\n')[0]
        return True, gpu_name
    return False, None

def detect_amd_gpu():
    """Check for AMD GPU using wmic"""
    success, stdout, stderr = run_command('wmic path win32_VideoController get name')
    if success:
        for line in stdout.split('\n'):
            line = line.strip()
            if 'amd' in line.lower() or 'radeon' in line.lower():
                return True, line
    return False, None

def detect_intel_gpu():
    """Check for Intel GPU using wmic"""
    success, stdout, stderr = run_command('wmic path win32_VideoController get name')
    if success:
        for line in stdout.split('\n'):
            line = line.strip()
            if 'intel' in line.lower():
                return True, line
    return False, None

def get_pytorch_install_command(build_type):
    """Get the appropriate PyTorch installation command"""
    if build_type == 'cuda':
        return 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
    elif build_type == 'directml':
        return 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && python -m pip install torch-directml'
    else:  # cpu
        return 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'

def main():
    """Main hardware detection and recommendation function"""
    print("🔍 AudioProcessor Hardware Detection")
    print("=" * 40)
    print()

    print("System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print()

    print("🔍 Scanning for GPUs...")
    print()

    # Check for NVIDIA GPU first (best performance)
    nvidia_detected, nvidia_name = detect_nvidia_gpu()
    if nvidia_detected:
        print(f"✅ NVIDIA GPU detected: {nvidia_name}")
        print("🚀 Recommended: CUDA PyTorch for maximum performance")
        print()
        print("Installation command:")
        print(f"  {get_pytorch_install_command('cuda')}")
        return 'cuda'

    # Check for AMD GPU
    amd_detected, amd_name = detect_amd_gpu()
    if amd_detected:
        print(f"✅ AMD GPU detected: {amd_name}")
        print("🚀 Recommended: DirectML PyTorch for GPU acceleration")
        print()
        print("Installation commands:")
        for cmd in get_pytorch_install_command('directml').split(' && '):
            print(f"  {cmd}")
        return 'directml'

    # Check for Intel GPU
    intel_detected, intel_name = detect_intel_gpu()
    if intel_detected:
        print(f"✅ Intel GPU detected: {intel_name}")
        print("🚀 Recommended: DirectML PyTorch for GPU acceleration")
        print()
        print("Installation commands:")
        for cmd in get_pytorch_install_command('directml').split(' && '):
            print(f"  {cmd}")
        return 'directml'

    # No GPU detected
    print("💻 No compatible GPU detected")
    print("📦 Recommended: CPU-only PyTorch")
    print()
    print("Installation command:")
    print(f"  {get_pytorch_install_command('cpu')}")
    print()
    print("ℹ️  CPU-only mode will work but may be slower for large files")
    return 'cpu'

if __name__ == "__main__":
    main()
