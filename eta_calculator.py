"""
ETA (Estimated Time to Completion) Calculator for transcription progress.
Calculates realistic time estimates based on actual processing speed.
"""

import time
from typing import Optional, Tuple
import statistics

class ETACalculator:
    def __init__(self, total_segments: int, total_duration: float):
        """
        Initialize ETA calculator.
        
        Args:
            total_segments: Total number of segments to process
            total_duration: Total audio duration in seconds
        """
        self.total_segments = total_segments
        self.total_duration = total_duration
        self.start_time = time.time()
        
        # Track processing history for smoothing
        self.segment_times = []  # List of (timestamp, completed_segments)
        self.processing_speeds = []  # List of recent speeds (segments/second)
        self.max_history = 10  # Keep last 10 speed measurements
        
        # Initial estimates
        self.last_update = self.start_time
        self.last_completed = 0
        
    def update_progress(self, completed_segments: int) -> Tuple[str, float]:
        """
        Update progress and calculate ETA.
        
        Args:
            completed_segments: Number of segments completed so far
            
        Returns:
            Tuple of (eta_string, speed_multiplier)
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Record this measurement
        self.segment_times.append((current_time, completed_segments))
        
        # Calculate current processing speed
        if len(self.segment_times) >= 2:
            # Use last few measurements for smoothing
            recent_measurements = self.segment_times[-min(5, len(self.segment_times)):]
            
            if len(recent_measurements) >= 2:
                time_diff = recent_measurements[-1][0] - recent_measurements[0][0]
                segment_diff = recent_measurements[-1][1] - recent_measurements[0][1]
                
                if time_diff > 0 and segment_diff > 0:
                    current_speed = segment_diff / time_diff  # segments per second
                    self.processing_speeds.append(current_speed)
                    
                    # Keep only recent speeds
                    if len(self.processing_speeds) > self.max_history:
                        self.processing_speeds = self.processing_speeds[-self.max_history:]
        
        # Calculate ETA
        if completed_segments <= 0:
            return "Calculating...", 0.0
            
        # Use smoothed speed for prediction
        if self.processing_speeds:
            # Use median of recent speeds for stability
            avg_speed = statistics.median(self.processing_speeds)
        else:
            # Fallback to simple average
            avg_speed = completed_segments / elapsed if elapsed > 0 else 0
            
        if avg_speed <= 0:
            return "Calculating...", 0.0
            
        # Calculate remaining work
        remaining_segments = self.total_segments - completed_segments
        eta_seconds = remaining_segments / avg_speed
        
        # Calculate speed multiplier (how fast we're processing vs real-time)
        if elapsed > 0 and completed_segments > 0:
            # Estimate how much audio we've processed
            avg_segment_duration = self.total_duration / self.total_segments
            processed_audio_time = completed_segments * avg_segment_duration
            speed_multiplier = processed_audio_time / elapsed
        else:
            speed_multiplier = 0.0
            
        # Format ETA string
        eta_str = self._format_time(eta_seconds)
        
        # Update tracking
        self.last_update = current_time
        self.last_completed = completed_segments
        
        return eta_str, speed_multiplier
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time string."""
        if seconds <= 0:
            return "Complete"
        elif seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = int(seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def get_progress_summary(self, completed_segments: int, active_threads: int) -> str:
        """Get a comprehensive progress summary."""
        if completed_segments <= 0:
            return "Starting..."
            
        percentage = (completed_segments / self.total_segments) * 100
        eta_str, speed_mult = self.update_progress(completed_segments)
        
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        # Create summary string
        summary = f"{completed_segments}/{self.total_segments} ({percentage:.1f}%)"
        
        if speed_mult > 0:
            summary += f" • {speed_mult:.1f}x speed"
            
        if eta_str != "Complete":
            summary += f" • ETA: {eta_str}"
            
        summary += f" • {elapsed_str} elapsed"
        
        if active_threads > 0:
            summary += f" • {active_threads} threads"
            
        return summary
