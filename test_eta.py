#!/usr/bin/env python3
"""Test the ETA calculator functionality."""

from eta_calculator import ETACalculator
import time

def test_eta_calculator():
    """Test ETA calculations with simulated progress."""
    print("🧪 Testing ETA Calculator...")
    
    # Simulate a 100-segment transcription job that takes 1 hour of audio
    calc = ETACalculator(100, 3600)  # 100 segments, 3600 seconds (1 hour)
    
    print("\nSimulating transcription progress:")
    
    test_points = [5, 15, 30, 50, 70, 85, 95, 100]
    
    for completed in test_points:
        time.sleep(0.2)  # Simulate some processing time
        
        eta_str, speed_mult = calc.update_progress(completed)
        summary = calc.get_progress_summary(completed, 4)  # 4 active threads
        
        print(f"  {completed:3d}% complete:")
        print(f"    ETA: {eta_str}")
        print(f"    Speed: {speed_mult:.1f}x realtime")
        print(f"    Summary: {summary}")
        print()

if __name__ == "__main__":
    test_eta_calculator()
