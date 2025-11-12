#!/usr/bin/env python3
"""Simple syntax check for transcribe_optimised.py"""

import sys
import os
import py_compile

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Try to compile the module
    py_compile.compile('transcribe_optimised.py', doraise=True)
    print("✅ Syntax check passed - no compilation errors")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Import/compilation error: {e}")
    sys.exit(1)