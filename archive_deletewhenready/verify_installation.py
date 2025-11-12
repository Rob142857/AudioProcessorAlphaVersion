#!/usr/bin/env python3
"""
Installation Verification Script for AudioProcessor
Tests that all components are properly installed and working
"""

import sys
import subprocess
import importlib.util

def run_command(cmd, description=""):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        success = result.returncode == 0
        if description:
            status = "✅" if success else "❌"
            print(f"{status} {description}")
        return success, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        if description:
            print(f"❌ {description} - Error: {str(e)}")
        return False, '', str(e)

def test_python_version():
    """Test Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_package_import(package_name, description=""):
    """Test if a package can be imported"""
    try:
        importlib.import_module(package_name)
        if description:
            print(f"✅ {description}")
        return True
    except ImportError as e:
        if description:
            print(f"❌ {description} - Import failed: {str(e)}")
        return False

def test_pytorch_gpu():
    """Test PyTorch GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA GPU available: {device_name} ({device_count} device{'s' if device_count != 1 else ''})")
            return True, "cuda"
        elif hasattr(torch, 'directml') and torch.directml.is_available():
            print("✅ DirectML GPU available")
            return True, "directml"
        else:
            print("⚠️  PyTorch CPU-only mode")
            return True, "cpu"
    except ImportError:
        print("❌ PyTorch not installed")
        return False, None

def test_whisper_model():
    """Test Whisper model loading"""
    try:
        import whisper
        # Try to load the large model (should be cached from preload)
        model = whisper.load_model("large")
        print("✅ Whisper large model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Whisper model loading failed: {str(e)}")
        return False

def test_ffmpeg():
    """Test FFmpeg availability"""
    success, stdout, stderr = run_command("ffmpeg -version", "")
    if success:
        # Extract version from first line
        first_line = stdout.split('\n')[0] if stdout else ""
        print(f"✅ FFmpeg available: {first_line}")
        return True
    else:
        print("❌ FFmpeg not found in PATH")
        return False

def main():
    """Main verification function"""
    print("🔍 AudioProcessor Installation Verification")
    print("=" * 50)
    print()

    tests_passed = 0
    total_tests = 0

    # Test Python version
    total_tests += 1
    if test_python_version():
        tests_passed += 1

    print()

    # Test core packages
    packages_to_test = [
        ("torch", "PyTorch"),
        ("whisper", "OpenAI Whisper"),
        ("moviepy", "MoviePy"),
        ("docx", "python-docx"),
        ("psutil", "psutil"),
        ("tqdm", "tqdm")
    ]

    for package, description in packages_to_test:
        total_tests += 1
        if test_package_import(package, description):
            tests_passed += 1

    print()

    # Test optional packages
    optional_packages = [
        ("webrtcvad", "WebRTC VAD (Voice Activity Detection)"),
        ("deepmultilingualpunctuation", "Deep Multilingual Punctuation")
    ]

    for package, description in optional_packages:
        if test_package_import(package, description):
            tests_passed += 1
        total_tests += 1

    print()

    # Test PyTorch GPU
    total_tests += 1
    gpu_success, gpu_type = test_pytorch_gpu()
    if gpu_success:
        tests_passed += 1

    print()

    # Test FFmpeg
    total_tests += 1
    if test_ffmpeg():
        tests_passed += 1

    print()

    # Test Whisper model (only if PyTorch is working)
    if gpu_success:
        total_tests += 1
        if test_whisper_model():
            tests_passed += 1

    print()
    print("=" * 50)

    # Summary
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! AudioProcessor is ready to use.")
        if gpu_type == "cuda":
            print("🚀 CUDA acceleration is available for maximum performance.")
        elif gpu_type == "directml":
            print("🚀 DirectML acceleration is available for GPU performance.")
        else:
            print("💻 CPU-only mode is working (consider GPU for better performance).")
    elif tests_passed >= total_tests - 2:  # Allow for 1-2 optional failures
        print("✅ Core functionality is working. Some optional features may be missing.")
        print("ℹ️  The application should still work, but with reduced features.")
    else:
        print("❌ Critical components are missing. Please run the installer again.")
        print("   Try: install.bat (full) or install_simple.bat (minimal)")

    print()
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
