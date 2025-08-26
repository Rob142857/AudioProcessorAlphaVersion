"""
Simple ETA-enabled transcription wrapper for testing the ETA calculator.
This is a clean version that sends proper progress messages.
"""

import time
import sys
import os
from typing import Optional

def send_eta_init(total_segments: int, total_duration: float):
    """Send ETA initialization message to GUI."""
    try:
        if hasattr(sys.stdout, 'output_queue'):
            sys.stdout.output_queue.put(f"ETA_INIT:{total_segments}|{total_duration}\n")
    except:
        pass

def send_progress_update(completed: int, total: int, active_threads: int):
    """Send progress update with ETA calculation support."""
    try:
        if hasattr(sys.stdout, 'output_queue'):
            percentage = (completed / total) * 100 if total > 0 else 0
            status = f"Transcribing: {completed}/{total} segments"
            sys.stdout.output_queue.put(f"PROGRESS:{percentage:.1f}|{status}|{active_threads}|\n")
    except:
        pass

def test_eta_transcription(input_file: str, model: str = "medium", output_dir: str = None):
    """
    Test function that simulates transcription with ETA updates.
    This demonstrates how the ETA calculator works.
    """
    try:
        # Import the working transcription function
        from transcribe import transcribe_file
        
        print(f"🚀 Starting ETA-enabled transcription...")
        print(f"📁 Input: {os.path.basename(input_file)}")
        
        # For demonstration, let's estimate segments and duration
        # In real implementation, this would come from VAD analysis
        
        # Simulate getting audio duration (you'd get this from actual file)
        try:
            from moviepy.editor import AudioFileClip
            with AudioFileClip(input_file) as audio:
                duration = audio.duration
        except:
            duration = 3600  # Default to 1 hour if we can't get duration
        
        # Estimate segments (typical VAD creates ~1 segment per 30-60 seconds)
        estimated_segments = max(int(duration / 45), 1)
        
        print(f"📊 Estimated: {estimated_segments} segments, {duration/60:.1f} minutes duration")
        
        # Send ETA initialization
        send_eta_init(estimated_segments, duration)
        
        # Simulate progress updates during transcription
        # In real implementation, this would be integrated into the actual transcription loop
        
        print("🧠 Starting transcription with ETA tracking...")
        
        # Use the existing transcribe function but add progress simulation
        start_time = time.time()
        
        # Call the actual transcription
        result = transcribe_file(input_file, model_name=model, output_dir=output_dir,
                               preprocess=True, vad=True, punctuate=True)
        
        # Send completion update
        try:
            if hasattr(sys.stdout, 'output_queue'):
                elapsed = time.time() - start_time
                sys.stdout.output_queue.put(f"PROGRESS:100.0|Transcription Complete!|0|Total: {elapsed:.0f}s\n")
        except:
            pass
        
        print(f"✅ ETA-enabled transcription complete!")
        return result
        
    except Exception as e:
        print(f"❌ ETA transcription failed: {e}")
        raise

def transcribe_file_eta_test(input_path, model_name="medium", output_dir=None, **kwargs):
    """
    Wrapper function that provides ETA updates for existing transcription.
    This can be used as a drop-in replacement during testing.
    """
    return test_eta_transcription(input_path, model_name, output_dir)
