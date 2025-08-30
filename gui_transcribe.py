"""Simplified GUI for Speech-to-Text transcription with large model only.

This GUI uses a single configuration:
- Large model only
- Auto device detection (CUDA > DirectML > CPU)
- No VAD segmentation
- No preprocessing
- Maximum threads based on RAM
- AI guardrails disabled
"""
import argparse
import os
import threading
import queue
import sys

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    tk = None

# Default Downloads folder with proper Windows path handling
DEFAULT_DOWNLOADS = os.path.normpath(os.path.join(os.path.expanduser("~"), "Downloads"))

def run_transcription(input_file: str, outdir: str, output_queue: queue.Queue):
    """Run simplified transcription with large model and auto device detection."""
    try:
        # Import and run the simplified transcription
        from transcribe_aggressive import transcribe_file_simple_auto
        
        output_queue.put(f"🚀 Starting SIMPLIFIED transcription for: {os.path.basename(input_file)}\n")
        output_queue.put("Using Large model with auto device detection\n")
        output_queue.put("No VAD, no preprocessing, maximum threads\n")
        
        out_txt = transcribe_file_simple_auto(input_file, output_dir=outdir)
        
        if out_txt and os.path.exists(out_txt):
            output_queue.put(f"✅ Transcription complete! Files saved to Downloads folder.\n")
            output_queue.put(f"  📄 Text file: {os.path.basename(out_txt)}\n")
            output_queue.put(f"  📄 Word document: {os.path.basename(out_txt.replace('.txt', '.docx'))}\n")
        else:
            output_queue.put("❌ Transcription failed or no output generated.\n")
            
    except Exception as e:
        output_queue.put(f"✗ Transcription failed: {e}\n")


class QueueWriter:
    """A file-like writer that puts text into a queue."""
    def __init__(self, q: queue.Queue):
        self.q = q
        self.output_queue = q  # Make queue accessible for progress updates

    def write(self, msg):
        if msg:
            # Don't use print() here as it causes infinite recursion with redirected stdout
            self.q.put(str(msg))

    def flush(self):
        pass


def launch_gui(default_outdir: str = None):
    if tk is None:
        print("Tkinter is not available in this Python build.")
        return

    root = tk.Tk()
    root.title("🎙️ Audio Transcription Pro")
    root.geometry("800x650")
    root.resizable(True, True)
    root.configure(bg='#1e1e1e')  # Dark background

    # Configure grid weights for proper resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Modern styling
    style = ttk.Style()
    style.configure('Modern.TFrame', background='#1e1e1e')
    style.configure('Modern.TLabel', background='#1e1e1e', foreground='#ffffff', font=('Segoe UI', 10))
    style.configure('Modern.TButton', font=('Segoe UI', 10, 'bold'), padding=10)
    style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#007acc', background='#1e1e1e')
    style.configure('Card.TLabelframe', background='#2d2d30', foreground='#ffffff', borderwidth=1, relief='solid')
    style.configure('Card.TLabelframe.Label', background='#2d2d30', foreground='#007acc', font=('Segoe UI', 11, 'bold'))

    mainframe = ttk.Frame(root, style='Modern.TFrame', padding="20 20 20 20")
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    mainframe.columnconfigure(1, weight=1)
    mainframe.rowconfigure(4, weight=1)

    # Title with icon
    title_frame = ttk.Frame(mainframe, style='Modern.TFrame')
    title_frame.grid(column=0, row=0, columnspan=3, pady=(0, 20), sticky=(tk.W, tk.E))
    title_frame.columnconfigure(1, weight=1)

    title_label = ttk.Label(title_frame, text="🎙️ Audio Transcription Pro", style='Title.TLabel')
    title_label.grid(column=0, row=0, sticky=tk.W)

    subtitle_label = ttk.Label(title_frame, text="Professional speech-to-text transcription with AI", 
                              font=('Segoe UI', 9), foreground='#cccccc', background='#1e1e1e')
    subtitle_label.grid(column=0, row=1, sticky=tk.W, pady=(5, 0))

    # Input file selection with modern card design
    input_frame = ttk.LabelFrame(mainframe, text=" 📁 File Selection ", style='Card.TLabelframe', padding="15 15 15 15")
    input_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
    input_frame.columnconfigure(1, weight=1)

    ttk.Label(input_frame, text="Audio/Video File:", style='Modern.TLabel').grid(column=0, row=0, sticky=tk.W, pady=(0, 8))

    input_var = tk.StringVar()
    input_entry = ttk.Entry(input_frame, textvariable=input_var, font=('Segoe UI', 10))
    input_entry.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10), padx=(0, 10))

    def browse_input():
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]
        p = filedialog.askopenfilename(title="Select audio/video file", filetypes=filetypes)
        if p:
            input_var.set(p)
            # Update status when file is selected
            status_label.config(text=f"📁 Selected: {os.path.basename(p)}", foreground='#4ade80')

    browse_btn = ttk.Button(input_frame, text="📂 Browse", command=browse_input, style='Modern.TButton')
    browse_btn.grid(column=2, row=1, sticky=tk.E)

    # Status label
    status_label = ttk.Label(input_frame, text="No file selected", font=('Segoe UI', 9), 
                            foreground='#888888', background='#2d2d30')
    status_label.grid(column=0, row=2, columnspan=3, sticky=tk.W, pady=(5, 0))

    # Output info
    downloads_dir = default_outdir if default_outdir else DEFAULT_DOWNLOADS
    output_frame = ttk.LabelFrame(mainframe, text=" 📍 Output Location ", style='Card.TLabelframe', padding="15 10 15 10")
    output_frame.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

    output_label = ttk.Label(output_frame, text=f"💾 {downloads_dir}", style='Modern.TLabel', 
                            font=('Segoe UI', 9), foreground='#cccccc')
    output_label.grid(column=0, row=0, sticky=tk.W)

    # Settings info with modern card
    settings_frame = ttk.LabelFrame(mainframe, text=" ⚙️ Configuration ", style='Card.TLabelframe', padding="15 10 15 10")
    settings_frame.grid(column=0, row=3, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

    settings_text = """🎯 Large AI Model (Best Quality)
🖥️  Auto Device Detection (CUDA → DirectML → CPU)
🚫 No VAD Segmentation (Faster for continuous speech)
🚫 No Audio Preprocessing (Direct processing)
🧵  Maximum CPU Threads (RAM-optimized)
🛡️  AI Guardrails Disabled (Captures all audio)"""

    settings_label = tk.Text(settings_frame, height=6, wrap=tk.WORD, font=('Consolas', 9), 
                            bg='#2d2d30', fg='#ffffff', borderwidth=0, highlightthickness=0)
    settings_label.insert(tk.END, settings_text)
    settings_label.config(state=tk.DISABLED)
    settings_label.grid(column=0, row=0, sticky=(tk.W, tk.E))

    # Progress and Status section
    progress_frame = ttk.LabelFrame(mainframe, text=" 📊 Progress & Status ", style='Card.TLabelframe', padding="15 15 15 15")
    progress_frame.grid(column=0, row=4, columnspan=3, sticky='ew', pady=(10, 5))
    progress_frame.columnconfigure(1, weight=1)

    # Progress bar with modern styling
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100, 
                                  length=400, style='Horizontal.TProgressbar')
    progress_bar.grid(column=0, row=0, columnspan=3, sticky='ew', pady=(0, 10))

    # Status labels
    progress_label = ttk.Label(progress_frame, text="Ready to start transcription", 
                              style='Modern.TLabel', font=('Segoe UI', 10, 'bold'))
    progress_label.grid(column=0, row=1, sticky='w')

    time_label = ttk.Label(progress_frame, text="", style='Modern.TLabel', foreground='#007acc')
    time_label.grid(column=2, row=1, sticky='e', padx=(10, 0))

    # Log area with modern design
    log_frame = ttk.LabelFrame(mainframe, text=" 📝 Activity Log ", style='Card.TLabelframe', padding="10 10 10 10")
    log_frame.grid(column=0, row=5, columnspan=3, sticky='nsew', pady=(0, 15))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)

    log_text = tk.Text(log_frame, wrap=tk.WORD, height=8, font=('Consolas', 9), 
                      bg='#1e1e1e', fg='#ffffff', borderwidth=0, highlightthickness=0,
                      insertbackground='#ffffff')
    log_text.grid(column=0, row=0, sticky='nsew')

    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_scroll.grid(column=1, row=0, sticky='ns')
    log_text['yscrollcommand'] = log_scroll.set
    log_text.configure(state='disabled')

    # Add initial welcome message
    log_text.configure(state='normal')
    log_text.insert('end', "🎯 Welcome to Audio Transcription Pro!\n", "title")
    log_text.insert('end', "Select an audio or video file and click 'Start Transcription' to begin.\n\n")
    log_text.insert('end', "💡 Tip: For best results with continuous speech, this tool processes the entire file at once.\n\n")
    log_text.configure(state='disabled')

    # Tag configuration for colored text
    log_text.tag_configure("title", foreground="#007acc", font=('Consolas', 9, 'bold'))
    log_text.tag_configure("success", foreground="#4ade80")
    log_text.tag_configure("error", foreground="#ef4444")
    log_text.tag_configure("info", foreground="#60a5fa")

    q = queue.Queue()

    def poll_queue():
        try:
            while True:
                msg = q.get_nowait()
                # Don't use print() here to avoid recursion with redirected stdout
                
                # Handle special progress messages
                if msg.startswith("PROGRESS:"):
                    _, progress_data = msg.split(":", 1)
                    try:
                        parts = progress_data.split("|")
                        if len(parts) >= 3:
                            percentage = float(parts[0])
                            status_text = parts[1]
                            thread_count = int(parts[2])
                            
                            # Update progress bar
                            progress_var.set(percentage)
                            progress_label.configure(text=status_text)
                            # Thread count removed from UI
                            
                            # Optional elapsed time
                            if len(parts) >= 4:
                                time_label.configure(text=parts[3])
                    except (ValueError, IndexError):
                        pass  # Ignore malformed progress messages
                        
                # Smart progress detection from log messages
                elif "segments" in msg.lower() and ("/" in msg or "complete" in msg.lower()):
                    # Try to extract progress from messages like "50/200 segments"
                    try:
                        import re
                        match = re.search(r'(\d+)[/\s]+(\d+)\s+segments?', msg, re.IGNORECASE)
                        if match:
                            completed = int(match.group(1))
                            total = int(match.group(2))
                            percentage = (completed / total) * 100
                            
                            progress_var.set(percentage)
                            progress_label.configure(text=f"Processing: {completed}/{total} segments ({percentage:.1f}%)")
                            
                    except (ValueError, ImportError):
                        pass
                        
                # Detect phase changes for status updates
                elif any(phase in msg for phase in ["Loading", "Preprocessing", "Extracting", "Transcribing", "Complete", "Model", "Detected loaded model"]):
                    # Update status based on current phase
                    if "Loading Whisper model" in msg:
                        progress_label.configure(text="Loading AI models...")
                        progress_var.set(5)
                    elif "Detected loaded model" in msg:
                        progress_label.configure(text="AI models loaded successfully")
                        progress_var.set(8)
                    elif "WARNING: Requested model" in msg and "but loaded" in msg:
                        progress_label.configure(text="⚠️ Wrong model loaded - use cache clear button")
                        progress_var.set(5)
                    elif "Preprocessing" in msg:
                        progress_label.configure(text="Preprocessing audio...")
                        progress_var.set(10)
                    elif "Extracting" in msg:
                        progress_label.configure(text="Extracting audio segments...")
                        progress_var.set(20)
                    elif "Transcribing" in msg or "Starting optimised" in msg:
                        progress_label.configure(text="Transcribing audio...")
                        progress_var.set(30)
                    elif "Complete" in msg and "successfully" in msg:
                        progress_var.set(100)
                        progress_label.configure(text="Transcription completed!")
                        # System status removed from UI
                        
                # Detect thread activity from webrtcvad warnings (indicates parallel processing)
                elif "webrtcvad" in msg and "warning" in msg.lower():
                    # Count concurrent processes by counting warnings
                    import threading
                    active = threading.active_count()
                    # Worker count removed from UI
                
                # Regular log message
                formatted_msg = format_log_message(msg)
                log_text.configure(state='normal')
                log_text.insert('end', formatted_msg)
                log_text.see('end')
                log_text.configure(state='disabled')
        except queue.Empty:
            pass
        except Exception as e:
            # Debug: print to stderr so it doesn't interfere with stdout redirection
            import sys
            print(f"DEBUG: poll_queue exception: {e}", file=sys.__stderr__)
        root.after(200, poll_queue)

    def format_log_message(msg):
        """Format log messages for better readability in the GUI."""
        if not msg.strip():
            return msg
        
        # Add proper spacing and structure to various message types
        formatted = msg
        
        # Header messages (starts with emoji or title)
        if any(msg.startswith(prefix) for prefix in ['🚀', '🤖', '🖥️', '📊', '🧠', '🖥️', '✂️', '🎵', '🗣️', '⚡']):
            formatted = f"\n{msg}\n"
        
        # Config/system info (contains ":")
        elif any(word in msg for word in ['Config:', 'RAM:', 'GPU:', 'CPU:', 'Workers:', 'System:']):
            formatted = f"  {msg}"
            
        # Progress indicators (contains numbers or percentages)
        elif any(char in msg for char in ['%']) or 'segments' in msg or 'Speed:' in msg:
            formatted = f"    {msg}"
            
        # Error messages
        elif any(prefix in msg for prefix in ['❌', '✗', '⚠️', 'Failed', 'Error']):
            formatted = f"\n❌ {msg}\n"
            
        # Success messages  
        elif any(prefix in msg for prefix in ['✓', '✅']) or 'complete' in msg.lower():
            formatted = f"\n✅ {msg}"
            
        # File paths and outputs (contains → or file extensions)
        elif '→' in msg or any(ext in msg for ext in ['.txt', '.docx', '.mp3', '.wav']):
            formatted = f"    {msg}"
            
        # Model loading messages
        elif any(word in msg for word in ['Model', 'Loaded', 'Loading']):
            formatted = f"      {msg}"
            
        # Time/duration info
        elif any(word in msg for word in ['Duration:', 'Time:', 'seconds', 'minutes']):
            formatted = f"    {msg}"
            
        return formatted

    def start_transcription_thread():
        # Debug: print to stderr so it doesn't interfere with stdout redirection
        import sys
        print("DEBUG: start_transcription_thread called", file=sys.__stderr__)
        
        inp = input_var.get().strip()
        print(f"DEBUG: Input file path: '{inp}'", file=sys.__stderr__)
        print(f"DEBUG: Input file exists: {os.path.isfile(inp) if inp else 'N/A'}", file=sys.__stderr__)
        
        if not inp or not os.path.isfile(inp):
            error_msg = f"Please select a valid audio or video file.\nPath: '{inp}'\nExists: {os.path.isfile(inp) if inp else False}"
            print(f"DEBUG: Validation failed: {error_msg}", file=sys.__stderr__)
            messagebox.showerror("Input Required", error_msg)
            return

        print("DEBUG: Validation passed, proceeding with transcription", file=sys.__stderr__)

        # Clear log and show starting message
        log_text.configure(state='normal')
        log_text.delete(1.0, 'end')
        log_text.insert('end', "Starting transcription...\n")
        log_text.see('end')
        log_text.configure(state='disabled')

        def worker():
            import sys
            try:
                # Send initial status message
                q.put("🔄 Initializing transcription process...\n")
                
                # Redirect prints from transcribe to the queue
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = QueueWriter(q)
                sys.stderr = QueueWriter(q)
                try:
                    run_transcription(inp, downloads_dir, q)
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
                
                # Send completion message
                q.put("✅ Transcription process finished!\n")
                
            except Exception as e:
                import traceback
                error_msg = f"Worker error: {e}\n{traceback.format_exc()}"
                q.put(error_msg)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # Run button
    run_btn = ttk.Button(mainframe, text="Start Transcription", command=start_transcription_thread)
    run_btn.grid(column=1, row=6, pady=(10, 0))

    poll_queue()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Simplified GUI for speech-to-text transcription")
    parser.add_argument("--input", help="input audio/video file (if provided, runs headless)")
    parser.add_argument("--outdir", help="output folder (defaults to Downloads)")
    args = parser.parse_args()

    outdir = args.outdir or DEFAULT_DOWNLOADS

    if args.input:
        # Run headless
        q = queue.Queue()
        
        def runner():
            run_transcription(args.input, outdir, q)
        
        t = threading.Thread(target=runner)
        t.start()
        
        # Print live output
        while t.is_alive() or not q.empty():
            try:
                msg = q.get(timeout=0.2)
                print(msg, end="")
            except queue.Empty:
                pass
        return

    # Launch GUI
    launch_gui(default_outdir=outdir)


if __name__ == "__main__":
    main()
