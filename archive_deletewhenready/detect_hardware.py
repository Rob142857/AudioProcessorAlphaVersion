#!/usr/bin/env python3
"""
Hardware Detection and PyTorch Installation Script for AudioProcessor
Detects GPU hardware and installs optimal PyTorch configuration
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

def install_pytorch(build_type, description):
    """Install PyTorch with the specified build type"""
    print(f'📦 Installing {description}...')
    if build_type == 'cuda':
        cmd = 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
    elif build_type == 'directml':
        # First install CPU PyTorch, then DirectML
        cmd_cpu = 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
        success, _, _ = run_command(cmd_cpu)
        if success:
            cmd = 'python -m pip install torch-directml'
        else:
            return False
    else:  # cpu
        cmd = 'python -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
    
    success, stdout, stderr = run_command(cmd)
    if success:
        print(f'✅ {description} installed successfully!')
        return True
    else:
        print(f'❌ {description} installation failed')
        if stderr:
            print(f'Error: {stderr}')
        return False

def main():
    """Main hardware detection and installation function"""
    print("🔍 AudioProcessor Hardware Detection & PyTorch Installation")
    print("=" * 60)
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
        if install_pytorch('cuda', 'CUDA PyTorch for NVIDIA GPU'):
            print('🚀 CUDA acceleration ready!')
            return True

    # Check for AMD GPU
    amd_detected, amd_name = detect_amd_gpu()
    if amd_detected:
        print(f"✅ AMD GPU detected: {amd_name}")
        print("🚀 Recommended: DirectML PyTorch for GPU acceleration")
        print()
        if install_pytorch('directml', 'DirectML PyTorch for AMD GPU'):
            print('🚀 DirectML acceleration ready!')
            return True

    # Check for Intel GPU
    intel_detected, intel_name = detect_intel_gpu()
    if intel_detected:
        print(f"✅ Intel GPU detected: {intel_name}")
        print("🚀 Recommended: DirectML PyTorch for GPU acceleration")
        print()
        if install_pytorch('directml', 'DirectML PyTorch for Intel GPU'):
            print('🚀 DirectML acceleration ready!')
            return True

    # Fallback to CPU-only
    print("💻 No compatible GPU detected")
    print("📦 Installing CPU-only PyTorch (universal compatibility)")
    print()
    if install_pytorch('cpu', 'CPU-only PyTorch'):
        print('✅ CPU PyTorch ready!')
        print('ℹ️  CPU-only mode will work but may be slower for large files')
        return True
    else:
        print('❌ All PyTorch installation attempts failed!')
        print('ℹ️  You can manually install PyTorch later with:')
        print('   pip install torch --index-url https://download.pytorch.org/whl/cpu')
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
