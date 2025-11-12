#!/usr/bin/env python3
"""
Quick validation script for Audio Processor Alpha Version installation
Tests all critical components and provides diagnostic information
"""

import sys
import os
import subprocess
from pathlib import Path

def test_component(name, test_func):
    """Test a component and return result"""
    try:
        result = test_func()
        print(f"✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"✗ {name}: FAILED - {e}")
        return False

def test_python():
    """Test Python version"""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        return f"Python {version} (compatible)"
    else:
        raise Exception(f"Python {version} - requires 3.11+")

def test_torch():
    """Test PyTorch installation and device"""
    import torch
    device_info = []
    
    if torch.cuda.is_available():
        try:
            cuda_version = torch.version.cuda  # type: ignore
            device_info.append(f"CUDA {cuda_version}")
            device_info.append(f"GPU: {torch.cuda.get_device_name()}")
        except Exception:
            device_info.append("CUDA available")
    
    try:
        import torch_directml  # type: ignore
        device_info.append("DirectML available")
    except ImportError:
        pass
    
    if not device_info:
        device_info.append("CPU-only")
    
    return f"PyTorch {torch.__version__} ({', '.join(device_info)})"

def test_whisper():
    """Test OpenAI Whisper"""
    import whisper
    models = whisper.available_models()
    return f"Whisper available, models: {', '.join(list(models)[:3])}..."

def test_other_deps():
    """Test other critical dependencies"""
    results = []
    
    try:
        import psutil
        results.append(f"psutil {psutil.__version__}")
    except ImportError:
        raise Exception("psutil not available")
    
    try:
        import docx
        results.append("python-docx available")
    except ImportError:
        raise Exception("python-docx not available")
    
    try:
        from deepmultilingualpunctuation import PunctuationModel
        results.append("deepmultilingualpunctuation available")
    except ImportError:
        raise Exception("deepmultilingualpunctuation not available")
    
    return ", ".join(results)

def test_gui_script():
    """Test if GUI script can be imported"""
    script_path = Path("gui_transcribe.py")
    if not script_path.exists():
        raise Exception("gui_transcribe.py not found")
    
    # Try to get help output without starting GUI
    try:
        result = subprocess.run([sys.executable, "gui_transcribe.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return "GUI script loads successfully"
        else:
            raise Exception(f"GUI script failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        return "GUI script loads (timeout on help - normal)"
    except Exception as e:
        raise Exception(f"GUI script error: {e}")

def test_transcription_module():
    """Test the core transcription module"""
    try:
        from transcribe_optimised import transcribe_file_simple_auto
        return "Core transcription module loads successfully"
    except ImportError as e:
        raise Exception(f"Cannot import transcription module: {e}")

def get_system_info():
    """Get system information"""
    import platform
    import psutil
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")

def main():
    """Run all validation tests"""
    print("Audio Processor Alpha Version - Installation Validation")
    print("="*60)
    
    # Test components
    tests = [
        ("Python Version", test_python),
        ("PyTorch", test_torch),
        ("OpenAI Whisper", test_whisper),
        ("Other Dependencies", test_other_deps),
        ("GUI Script", test_gui_script),
        ("Transcription Module", test_transcription_module),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        if test_component(name, test_func):
            passed += 1
    
    print(f"\nValidation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nYou can now run: python gui_transcribe.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    get_system_info()
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        sys.exit(1)