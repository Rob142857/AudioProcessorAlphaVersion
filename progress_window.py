"""
Compact Always-On-Top Progress Window for Transcription
Shows current file, progress bar, and live text output
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time


class TranscriptionProgressWindow:
    """Small always-on-top window showing transcription progress."""
    
    def __init__(self):
        self.root = None
        self.is_running = False
        self.message_queue = queue.Queue()
        self._thread = None
        
        # State
        self.current_file = ""
        self.progress = 0
        self.current_text = ""
        self.status = "Idle"
        
    def start(self):
        """Start the progress window in a separate thread."""
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_window, daemon=True)
        self._thread.start()
        # Give window time to initialize
        time.sleep(0.2)
        
    def stop(self):
        """Stop and close the progress window."""
        self.is_running = False
        if self.root:
            try:
                self.root.quit()
            except:
                pass
                
    def set_file(self, filename: str):
        """Set the current file being processed."""
        self.message_queue.put(('file', filename))
        
    def set_progress(self, percent: float):
        """Set progress bar value (0-100)."""
        self.message_queue.put(('progress', percent))
        
    def set_status(self, status: str):
        """Set status text (e.g., 'Transcribing...', 'Processing...')."""
        self.message_queue.put(('status', status))
        
    def set_text(self, text: str):
        """Set the current text being output (shows last ~80 chars)."""
        self.message_queue.put(('text', text))
        
    def set_file_progress(self, current: int, total: int):
        """Set batch progress (e.g., file 3 of 10)."""
        self.message_queue.put(('batch', (current, total)))
        
    def _run_window(self):
        """Run the tkinter window (called in separate thread)."""
        self.root = tk.Tk()
        self.root.title("Transcription Progress")
        
        # Always on top, compact size
        self.root.attributes('-topmost', True)
        self.root.geometry("500x120")
        self.root.resizable(True, False)
        self.root.minsize(400, 120)
        
        # Dark theme colors
        bg_color = "#2b2b2b"
        fg_color = "#e0e0e0"
        accent_color = "#4a9eff"
        
        self.root.configure(bg=bg_color)
        
        # Main frame with padding
        main_frame = tk.Frame(self.root, bg=bg_color, padx=10, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Row 1: File name and batch progress
        file_frame = tk.Frame(main_frame, bg=bg_color)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(file_frame, text="📄", bg=bg_color, fg=fg_color, font=("Segoe UI Emoji", 10)).pack(side=tk.LEFT)
        
        self.file_label = tk.Label(
            file_frame, 
            text="No file selected", 
            bg=bg_color, 
            fg=fg_color,
            font=("Segoe UI", 9),
            anchor="w"
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.batch_label = tk.Label(
            file_frame,
            text="",
            bg=bg_color,
            fg="#888888",
            font=("Segoe UI", 9)
        )
        self.batch_label.pack(side=tk.RIGHT)
        
        # Row 2: Status and progress bar
        progress_frame = tk.Frame(main_frame, bg=bg_color)
        progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.status_label = tk.Label(
            progress_frame,
            text="Idle",
            bg=bg_color,
            fg=accent_color,
            font=("Segoe UI", 9, "bold"),
            width=15,
            anchor="w"
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Style for progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor='#3c3c3c',
            background=accent_color,
            lightcolor=accent_color,
            darkcolor=accent_color,
            bordercolor='#3c3c3c',
            thickness=18
        )
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            style="Custom.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.percent_label = tk.Label(
            progress_frame,
            text="0%",
            bg=bg_color,
            fg=fg_color,
            font=("Segoe UI", 9),
            width=5
        )
        self.percent_label.pack(side=tk.RIGHT)
        
        # Row 3: Live text output
        text_frame = tk.Frame(main_frame, bg="#1e1e1e", padx=5, pady=3)
        text_frame.pack(fill=tk.X, pady=(0, 0))
        
        tk.Label(text_frame, text="💬", bg="#1e1e1e", fg=fg_color, font=("Segoe UI Emoji", 9)).pack(side=tk.LEFT)
        
        self.text_label = tk.Label(
            text_frame,
            text="Waiting for transcription...",
            bg="#1e1e1e",
            fg="#aaaaaa",
            font=("Consolas", 9),
            anchor="w"
        )
        self.text_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Handle close button
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start update loop
        self._update_loop()
        
        try:
            self.root.mainloop()
        except:
            pass
            
    def _on_close(self):
        """Handle window close - just hide it."""
        self.root.withdraw()
        
    def _update_loop(self):
        """Process message queue and update UI."""
        if not self.is_running:
            return
            
        try:
            while True:
                try:
                    msg_type, value = self.message_queue.get_nowait()
                    
                    if msg_type == 'file':
                        # Truncate long filenames
                        display_name = value
                        if len(display_name) > 50:
                            display_name = "..." + display_name[-47:]
                        self.file_label.config(text=display_name)
                        self.current_file = value
                        
                    elif msg_type == 'progress':
                        self.progress_bar['value'] = value
                        self.percent_label.config(text=f"{int(value)}%")
                        
                    elif msg_type == 'status':
                        self.status_label.config(text=value)
                        
                    elif msg_type == 'text':
                        # Show last ~70 characters
                        display_text = value.strip()
                        if len(display_text) > 70:
                            display_text = display_text[-70:]
                        self.text_label.config(text=display_text)
                        
                    elif msg_type == 'batch':
                        current, total = value
                        if total > 1:
                            self.batch_label.config(text=f"[{current}/{total}]")
                        else:
                            self.batch_label.config(text="")
                            
                except queue.Empty:
                    break
                    
        except Exception as e:
            pass
            
        # Schedule next update
        if self.is_running and self.root:
            self.root.after(100, self._update_loop)


# Global instance
_progress_window = None


def get_progress_window() -> TranscriptionProgressWindow:
    """Get or create the global progress window instance."""
    global _progress_window
    if _progress_window is None:
        _progress_window = TranscriptionProgressWindow()
    return _progress_window


def show_progress_window():
    """Show the progress window."""
    window = get_progress_window()
    window.start()
    return window


def hide_progress_window():
    """Hide/stop the progress window."""
    global _progress_window
    if _progress_window:
        _progress_window.stop()
        _progress_window = None


# Test code
if __name__ == "__main__":
    import time
    
    window = show_progress_window()
    time.sleep(0.5)
    
    window.set_file("C:\\Users\\Test\\Documents\\0726 Movement Exercise.mp4")
    window.set_status("Transcribing...")
    window.set_file_progress(1, 5)
    
    # Simulate progress
    for i in range(101):
        window.set_progress(i)
        if i % 10 == 0:
            window.set_text(f"This is sample transcription text at {i}% progress...")
        time.sleep(0.05)
    
    window.set_status("Complete!")
    time.sleep(2)
    hide_progress_window()
