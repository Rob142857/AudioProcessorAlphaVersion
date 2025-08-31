"""Speech to Text Transcription Tool v1.0Beta

A clean, professional GUI for converting audio and video files to text using AI.
- Large AI model for professional quality
- Auto device detection (CUDA > DirectML > CPU)
- No VAD segmentation for continuous speech
- Maximum CPU threads based on RAM
- Clean, modern interface
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
    """Run transcription with large model and auto device detection."""
    try:
        # Import and run the transcription
        from transcribe_optimised import transcribe_file_simple_auto

        output_queue.put("Starting transcription for: {}\n".format(os.path.basename(input_file)))
        output_queue.put("Using Large model with auto device detection\n")
        output_queue.put("No VAD, no preprocessing, maximum threads\n")

        out_txt = transcribe_file_simple_auto(input_file, output_dir=outdir)

        if out_txt and os.path.exists(out_txt):
            output_queue.put("Transcription complete! Files saved to Downloads folder.\n")
            output_queue.put("  Text file: {}\n".format(os.path.basename(out_txt)))
            output_queue.put("  Word document: {}\n".format(os.path.basename(out_txt.replace('.txt', '.docx'))))
        else:
            output_queue.put("Transcription failed or no output generated.\n")

    except Exception as e:
        output_queue.put("Transcription failed: {}\n".format(e))


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
    root.title("Speech to Text Transcription Tool v1.0Beta")
    root.geometry("800x750")
    root.minsize(700, 650)  # Set minimum window size
    root.resizable(True, True)
    root.configure(bg='#f8f9fa')  # Clean light background

    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    # Configure grid weights for proper resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Clean modern styling
    style = ttk.Style()

    # Configure overall theme
    style.configure('Clean.TFrame', background='#f8f9fa')
    style.configure('Clean.TLabel', background='#f8f9fa', foreground='#2c3e50', font=('Segoe UI', 11))

    # Title styling
    style.configure('Title.TLabel',
                   font=('Segoe UI', 20, 'bold'),
                   foreground='#1a365d',
                   background='#f8f9fa')
    style.configure('Subtitle.TLabel',
                   font=('Segoe UI', 10),
                   foreground='#64748b',
                   background='#f8f9fa')

    # Section headers
    style.configure('Section.TLabel',
                   font=('Segoe UI', 13, 'bold'),
                   foreground='#374151',
                   background='#f8f9fa')

    mainframe = ttk.Frame(root, style='Clean.TFrame', padding="30 30 30 30")
    mainframe.grid(column=0, row=0, sticky="nsew")
    mainframe.columnconfigure(1, weight=1)
    mainframe.rowconfigure(6, weight=1)  # Log area expands
    mainframe.rowconfigure(7, minsize=60)  # Start button row has minimum height

    # Clean title section
    title_frame = ttk.Frame(mainframe, style='Clean.TFrame')
    title_frame.grid(column=0, row=0, columnspan=3, pady=(0, 30), sticky="ew")
    title_frame.columnconfigure(1, weight=1)

    title_label = ttk.Label(title_frame, text="Speech to Text Transcription Tool", style='Title.TLabel')
    title_label.grid(column=0, row=0, sticky="w")

    subtitle_label = ttk.Label(title_frame,
                              text="v1.0Beta - Professional AI-powered transcription",
                              style='Subtitle.TLabel')
    subtitle_label.grid(column=0, row=1, sticky="w", pady=(8, 0))

    # Clean file selection section
    file_section_label = ttk.Label(mainframe, text="File Selection", style='Section.TLabel')
    file_section_label.grid(column=0, row=1, columnspan=3, sticky="w", pady=(0, 15))

    # File input area with subtle background
    file_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
    file_frame.grid(column=0, row=2, columnspan=3, sticky="ew", pady=(0, 25))
    file_frame.columnconfigure(1, weight=1)

    ttk.Label(file_frame, text="Audio/Video File:", background='white', foreground='#374151',
             font=('Segoe UI', 10, 'bold')).grid(column=0, row=0, sticky="w", padx=20, pady=(20, 10))

    input_var = tk.StringVar()
    input_entry = tk.Entry(file_frame, textvariable=input_var, font=('Segoe UI', 10),
                          relief='flat', borderwidth=1, bg='white', fg='#2c3e50',
                          insertbackground='#2c3e50')
    input_entry.grid(column=0, row=1, columnspan=2, sticky="ew", padx=20, pady=(0, 20))

    def browse_input():
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]
        p = filedialog.askopenfilename(title="Select audio/video file", filetypes=filetypes)
        if p:
            input_var.set(p)
            status_label.config(text="Selected: {}".format(os.path.basename(p)), foreground='#059669')

    browse_btn = tk.Button(file_frame, text="Browse", command=browse_input,
                          font=('Segoe UI', 10, 'bold'), bg='#007acc', fg='white',
                          relief='flat', borderwidth=0, padx=20, pady=8,
                          activebackground='#0056b3', activeforeground='white')
    browse_btn.grid(column=2, row=1, sticky="e", padx=20)

    # Status label
    status_label = ttk.Label(file_frame, text="No file selected",
                            font=('Segoe UI', 9), foreground='#6b7280', background='white')
    status_label.grid(column=0, row=2, columnspan=3, sticky="w", padx=20, pady=(0, 20))

    downloads_dir = default_outdir if default_outdir else DEFAULT_DOWNLOADS

    # Clean settings section
    settings_section_label = ttk.Label(mainframe, text="Settings", style='Section.TLabel')
    settings_section_label.grid(column=0, row=3, columnspan=3, sticky="w", pady=(0, 15))

    # Settings area with subtle background
    settings_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
    settings_frame.grid(column=0, row=4, columnspan=3, sticky="ew", pady=(0, 25))

    settings_text = """Large AI Model (Professional Quality)
Auto Device Detection (CUDA -> DirectML -> CPU)
Direct Audio Processing (No Segmentation)
Maximum CPU Threads (RAM-Optimized)
Output Directory: {}
Full Audio Capture (All Content Preserved)""".format(downloads_dir)

    settings_label = tk.Text(settings_frame, height=6, wrap=tk.WORD, font=('Segoe UI', 10),
                            bg='white', fg='#374151', borderwidth=0, highlightthickness=0,
                            padx=20, pady=20)
    settings_label.insert(tk.END, settings_text)
    settings_label.config(state=tk.DISABLED)
    settings_label.grid(column=0, row=0, sticky="ew")

    # Log area with modern design
    log_section_label = ttk.Label(mainframe, text="Activity Log", style='Section.TLabel')
    log_section_label.grid(column=0, row=5, columnspan=3, sticky="w", pady=(0, 15))

    log_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
    log_frame.grid(column=0, row=6, columnspan=3, sticky="nsew", pady=(0, 25))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)

    log_text = tk.Text(log_frame, wrap=tk.WORD, height=15, font=('Segoe UI', 9),
                      bg='white', fg='#2c3e50', borderwidth=0, highlightthickness=0,
                      insertbackground='#2c3e50')
    log_text.grid(column=0, row=0, sticky="nsew")

    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_scroll.grid(column=1, row=0, sticky="ns")
    log_text['yscrollcommand'] = log_scroll.set
    log_text.configure(state='disabled')

    # Add initial welcome message
    log_text.configure(state='normal')
    log_text.insert('end', "Welcome to Speech to Text Transcription Tool v1.0Beta!\n", "title")
    log_text.insert('end', "Select an audio or video file and click 'Start Transcription' to begin.\n\n")
    log_text.insert('end', "Tip: For best results with continuous speech, this tool processes the entire file at once.\n\n")
    log_text.configure(state='disabled')

    # Tag configuration for colored text
    log_text.tag_configure("title", foreground="#007acc", font=('Segoe UI', 9, 'bold'))
    log_text.tag_configure("success", foreground="#4ade80")
    log_text.tag_configure("error", foreground="#ef4444")
    log_text.tag_configure("info", foreground="#60a5fa")

    q = queue.Queue()

    def poll_queue():
        try:
            while True:
                msg = q.get_nowait()
                # Don't use print() here to avoid recursion with redirected stdout

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
            print("DEBUG: poll_queue exception: {}".format(e), file=sys.__stderr__)
        root.after(200, poll_queue)

    def format_log_message(msg):
        """Format log messages for better readability in the GUI."""
        if not msg.strip():
            return msg

        # Add proper spacing and structure to various message types
        formatted = msg

        # Header messages (starts with emoji or title)
        if any(msg.startswith(prefix) for prefix in ['Starting', 'Using', 'Transcription']):
            formatted = "{}\n".format(msg)

        # Config/system info (contains ":")
        elif any(word in msg for word in ['Config:', 'RAM:', 'GPU:', 'CPU:', 'Workers:', 'System:']):
            formatted = "  {}".format(msg)

        # Progress indicators (contains numbers or percentages)
        elif any(char in msg for char in ['%']) or 'segments' in msg or 'Speed:' in msg:
            formatted = "    {}".format(msg)

        # Error messages
        elif any(prefix in msg for prefix in ['Transcription failed', 'Error']):
            formatted = "\nError: {}\n".format(msg)

        # Success messages
        elif any(prefix in msg for prefix in ['complete', 'Complete']) or 'saved' in msg.lower():
            formatted = "\nSuccess: {}".format(msg)

        # File paths and outputs (contains file extensions)
        elif any(ext in msg for ext in ['.txt', '.docx', '.mp3', '.wav']):
            formatted = "    {}".format(msg)

        # Model loading messages
        elif any(word in msg for word in ['Model', 'Loaded', 'Loading']):
            formatted = "      {}".format(msg)

        # Time/duration info
        elif any(word in msg for word in ['Duration:', 'Time:', 'seconds', 'minutes']):
            formatted = "    {}".format(msg)

        return formatted

    def start_transcription_thread():
        # Debug: print to stderr so it doesn't interfere with stdout redirection
        import sys
        print("DEBUG: start_transcription_thread called", file=sys.__stderr__)

        inp = input_var.get().strip()
        print("DEBUG: Input file path: '{}'".format(inp), file=sys.__stderr__)
        print("DEBUG: Input file exists: {}".format(os.path.isfile(inp) if inp else 'N/A'), file=sys.__stderr__)

        if not inp or not os.path.isfile(inp):
            error_msg = "Please select a valid audio or video file.\nPath: '{}'\nExists: {}".format(inp, os.path.isfile(inp) if inp else False)
            print("DEBUG: Validation failed: {}".format(error_msg), file=sys.__stderr__)
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
                q.put("Initializing transcription process...\n")

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
                q.put("Transcription process finished!\n")

            except Exception as e:
                import traceback
                error_msg = "Worker error: {}\n{}".format(e, traceback.format_exc())
                q.put(error_msg)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # Run button
    run_btn = tk.Button(mainframe, text="Start Transcription", command=start_transcription_thread,
                       font=('Segoe UI', 12, 'bold'), bg='#007acc', fg='white',
                       relief='flat', borderwidth=0, padx=30, pady=12,
                       activebackground='#0056b3', activeforeground='white',
                       cursor='hand2')
    run_btn.grid(column=1, row=7, pady=(10, 0))

    poll_queue()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Speech to Text Transcription Tool v1.0Beta")
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
