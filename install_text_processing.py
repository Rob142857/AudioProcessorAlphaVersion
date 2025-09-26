#!/usr/bin/env python3
"""
Install Enhanced Text Processing Libraries for AudioProcessor Alpha Version

This script installs the additional libraries needed for ultra-enhanced text processing
with parallel processing capabilities.

Usage:
    python install_text_processing.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def install_package(package_name, description=None):
    """Install a Python package via pip."""
    if description is None:
        description = f"Installing {package_name}"
    
    cmd = f'python -m pip install "{package_name}" --upgrade'
    return run_command(cmd, description)

def download_spacy_model():
    """Download the spaCy English model."""
    cmd = "python -m spacy download en_core_web_sm"
    return run_command(cmd, "Downloading spaCy English model")

def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = [
            ('punkt', 'Punkt tokenizer'),
            ('averaged_perceptron_tagger', 'POS tagger'),
            ('maxent_ne_chunker', 'Named entity chunker'),
            ('words', 'Word corpus'),
            ('stopwords', 'Stopwords corpus'),
        ]
        
        for data_name, description in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
                print(f"✅ NLTK {description} already available")
            except LookupError:
                print(f"📥 Downloading NLTK {description}...")
                nltk.download(data_name, quiet=True)
                print(f"✅ NLTK {description} downloaded")
        
        return True
    except Exception as e:
        print(f"❌ NLTK data download failed: {e}")
        return False

def main():
    """Main installation function."""
    print("🚀 Installing Enhanced Text Processing Libraries")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # List of packages to install
    packages = [
        ("spacy>=3.7.0,<4.0.0", "Installing spaCy for advanced NLP"),
        ("nltk>=3.8.0,<4.0.0", "Installing NLTK for text processing"),
        ("textstat>=0.7.0,<1.0.0", "Installing textstat for readability analysis"),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    # Install packages
    for package, description in packages:
        if install_package(package, description):
            success_count += 1
        else:
            print(f"⚠️  Continuing with remaining installations...")
    
    print("\n" + "=" * 60)
    print(f"📊 Package installation results: {success_count}/{total_count} successful")
    
    # Download spaCy model if spaCy was installed successfully
    if success_count > 0:
        print("\n🔄 Downloading language models and data...")
        
        spacy_success = download_spacy_model()
        nltk_success = download_nltk_data()
        
        model_success = sum([spacy_success, nltk_success])
        print(f"📊 Model download results: {model_success}/2 successful")
    
    # Test the installation
    print("\n🧪 Testing installation...")
    try:
        # Test ultra text processor import
        sys.path.insert(0, str(Path(__file__).parent))
        from text_processor_ultra import create_ultra_processor
        
        # Create a test processor
        processor = create_ultra_processor(max_workers=2)
        
        # Test with simple text
        test_text = "hello world this is a test"
        result = processor.process_text_ultra(test_text, passes=3)
        
        if result and len(result) > 0:
            print("✅ Ultra text processor test successful")
            print(f"   Input: {repr(test_text)}")
            print(f"   Output: {repr(result)}")
        else:
            print("⚠️  Ultra text processor test returned empty result")
            
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print("⚠️  Some features may not work correctly")
    
    print("\n" + "=" * 60)
    
    if success_count == total_count:
        print("🎉 Enhanced text processing installation completed successfully!")
        print("✨ You can now use ultra-enhanced text processing with parallel processing")
        print("\nFeatures enabled:")
        print("   • Multi-pass text processing (up to 6 passes)")
        print("   • Parallel processing across multiple CPU cores")
        print("   • Advanced sentence and paragraph segmentation")
        print("   • Context-aware capitalization and punctuation")
        print("   • Readability assessment and optimization")
        print("   • Semantic paragraph grouping")
        return True
    else:
        print("⚠️  Installation completed with some issues")
        print("📋 Some advanced features may not be available")
        print("💡 Try running this script again or install packages manually")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)