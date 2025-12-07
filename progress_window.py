"""
Compact Always-On-Top Progress Window for Transcription
Shows current file, progress bar, and live text output

This version uses a Toplevel window that must be created from the main GUI thread.
"""

import tkinter as tk
from tkinter import ttk
import queue


class TranscriptionProgressWindow:
    """Small always-on-top window showing transcription progress.
    
    This is a Toplevel window that must be created and managed from the main GUI thread.
    Use the module-level functions for thread-safe updates.
    """
    
    def __init__(self, parent_root):
        """Create the progress window as a Toplevel of the parent root."""
        self.parent_root = parent_root
        self.message_queue = queue.Queue()
        self.is_visible = False
        
        # State
        self.current_file = ""
        self.progress = 0
        self.current_text = ""
        self.status = "Idle"
        
        # Create the toplevel window
        self.window = tk.Toplevel(parent_root)
        self.window.title("Transcription Progress")
        
        # Always on top, compact size
        self.window.attributes('-topmost', True)
        self.window.geometry("500x120")
        self.window.resizable(True, False)
        self.window.minsize(400, 120)
        
        # Dark theme colors
        bg_color = "#2b2b2b"
        fg_color = "#e0e0e0"
        accent_color = "#4a9eff"
        
        self.window.configure(bg=bg_color)
        
        # Main frame with padding
        main_frame = tk.Frame(self.window, bg=bg_color, padx=10, pady=8)
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
        
        # Handle close button - just hide
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        
        # Start hidden
        self.window.withdraw()
        
        # Start update loop
        self._schedule_update()
    
    def _schedule_update(self):
        """Schedule the next queue check."""
        if self.parent_root:
            try:
                self.parent_root.after(100, self._process_queue)
            except:
                pass
    
    def _process_queue(self):
        """Process pending messages from the queue."""
        try:
            while True:
                try:
                    msg_type, value = self.message_queue.get_nowait()
                    
                    if msg_type == 'file':
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
                            
                    elif msg_type == 'show':
                        self.window.deiconify()
                        self.window.lift()
                        self.is_visible = True
                        
                    elif msg_type == 'hide':
                        self.window.withdraw()
                        self.is_visible = False
                        
                except queue.Empty:
                    break
        except Exception:
            pass
        
        # Schedule next update
        self._schedule_update()
    
    def show(self):
        """Show the window (thread-safe via queue)."""
        self.message_queue.put(('show', None))
    
    def hide(self):
        """Hide the window (thread-safe via queue)."""
        self.message_queue.put(('hide', None))
        
    def set_file(self, filename: str):
        """Set the current file being processed (thread-safe)."""
        self.message_queue.put(('file', filename))
        
    def set_progress(self, percent: float):
        """Set progress bar value 0-100 (thread-safe)."""
        self.message_queue.put(('progress', percent))
        
    def set_status(self, status: str):
        """Set status text (thread-safe)."""
        self.message_queue.put(('status', status))
        
    def set_text(self, text: str):
        """Set the current text being output (thread-safe)."""
        self.message_queue.put(('text', text))
        
    def set_file_progress(self, current: int, total: int):
        """Set batch progress e.g. file 3 of 10 (thread-safe)."""
        self.message_queue.put(('batch', (current, total)))


# Global instance - will be set by the main GUI
_progress_window = None
_parent_root = None


def init_progress_window(parent_root):
    """Initialize the progress window with the parent tkinter root.
    
    Must be called from the main GUI thread before using other functions.
    """
    global _progress_window, _parent_root
    _parent_root = parent_root
    _progress_window = TranscriptionProgressWindow(parent_root)
    return _progress_window


def get_progress_window():
    """Get the progress window instance (may be None if not initialized)."""
    return _progress_window


def show_progress_window():
    """Show the progress window."""
    if _progress_window:
        _progress_window.show()
    return _progress_window


def hide_progress_window():
    """Hide the progress window."""
    if _progress_window:
        _progress_window.hide()
