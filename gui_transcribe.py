"""Optimized GUI for Speech-to-Text transcription with best quality settings.

This GUI uses optimized settings for best transcription quality:
- Preprocessing always enabled
- VAD segmentation always enabled  
- Punctuation restoration always enabled
- Optimized Whisper parameters to capture all speech
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

def run_transcription(input_file: str, outdir: str, options: dict, output_queue: queue.Queue):
    """Call transcribe.transcribe_file with optimized settings."""
    try:
        from transcribe import transcribe_file
    except Exception as e:
        output_queue.put(f"Failed to import transcribe module: {e}\n")
        return

    try:
        output_queue.put(f"Starting optimized transcription for: {input_file}\n")
        output_queue.put("Using optimized settings: Preprocessing ✓ VAD Segmentation ✓ Punctuation ✓\n")
        
        # Always use optimized settings for best quality
        out_txt = transcribe_file(input_file,
                                  model_name=options.get("model", "medium"),
                                  keep_temp=options.get("keep_temp", False),
                                  device_preference=options.get("device", "auto"),
                                  output_dir=outdir)
        output_queue.put(f"✓ Transcription complete! Files saved to Downloads folder.\n")
        output_queue.put(f"  → Text file: {os.path.basename(out_txt)}\n")
        output_queue.put(f"  → Word document: {os.path.basename(out_txt.replace('.txt', '.docx'))}\n")
    except Exception as e:
        output_queue.put(f"✗ Transcription failed: {e}\n")


class QueueWriter:
    """A file-like writer that puts text into a queue."""
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, msg):
        if msg:
            self.q.put(str(msg))

    def flush(self):
        pass


def launch_gui(default_outdir: str = None):
    if tk is None:
        print("Tkinter is not available in this Python build.")
        return

    root = tk.Tk()
    root.title("Speech-to-Text Transcription")
    root.geometry("900x550")
    root.resizable(True, True)

    # Configure grid weights for proper resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    mainframe = ttk.Frame(root, padding="15 15 15 15")
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    mainframe.columnconfigure(1, weight=1)
    mainframe.rowconfigure(3, weight=1)

    # Title
    title_label = ttk.Label(mainframe, text="Speech-to-Text Transcription", font=('TkDefaultFont', 16, 'bold'))
    title_label.grid(column=0, row=0, columnspan=4, pady=(0, 15))

    # Input file selection
    input_var = tk.StringVar()
    ttk.Label(mainframe, text="Select audio or video file:", font=('TkDefaultFont', 10)).grid(column=0, row=1, sticky=tk.W)
    input_entry = ttk.Entry(mainframe, textvariable=input_var, width=70, font=('TkDefaultFont', 10))
    input_entry.grid(column=1, row=1, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10))

    def browse_input():
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]
        p = filedialog.askopenfilename(title="Select audio/video file", filetypes=filetypes)
        if p:
            input_var.set(p)

    browse_btn = ttk.Button(mainframe, text="Browse", command=browse_input)
    browse_btn.grid(column=3, row=1, padx=(10, 0))

    # Output info (fixed to Downloads)
    downloads_dir = default_outdir if default_outdir else DEFAULT_DOWNLOADS
    output_label = ttk.Label(mainframe, text=f"Output location: {downloads_dir}", font=('TkDefaultFont', 9), foreground='#666666')
    output_label.grid(column=0, row=2, columnspan=4, sticky=tk.W, pady=(5, 15))

    # Settings frame
    settings_frame = ttk.LabelFrame(mainframe, text="Settings", padding="10 10 10 10")
    settings_frame.grid(column=0, row=3, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 15))
    settings_frame.columnconfigure(1, weight=1)

    # Model selection
    model_var = tk.StringVar(value="medium")
    ttk.Label(settings_frame, text="AI Model:").grid(column=0, row=0, sticky=tk.W, padx=(0, 10))
    model_combo = ttk.Combobox(settings_frame, textvariable=model_var, values=("medium", "large"), 
                              state="readonly", width=20)
    model_combo.grid(column=1, row=0, sticky=tk.W)
    
    # Model info
    model_info = ttk.Label(settings_frame, text="Medium: Good quality, faster (~1.4GB) • Large: Best quality, slower (~2.9GB)", 
                          font=('TkDefaultFont', 8), foreground='#666666')
    model_info.grid(column=0, row=1, columnspan=4, sticky=tk.W, pady=(5, 10))

    # Processing selection  
    device_var = tk.StringVar(value="auto")
    ttk.Label(settings_frame, text="Processing:").grid(column=0, row=2, sticky=tk.W, padx=(0, 10))
    device_combo = ttk.Combobox(settings_frame, textvariable=device_var, values=("auto", "cpu", "cuda"), 
                               state="readonly", width=20)
    device_combo.grid(column=1, row=2, sticky=tk.W)

    # Processing info
    processing_info = ttk.Label(settings_frame, text="Auto: Best available • CPU: Compatible with all systems • CUDA: NVIDIA GPU acceleration", 
                               font=('TkDefaultFont', 8), foreground='#666666')
    processing_info.grid(column=0, row=3, columnspan=4, sticky=tk.W, pady=(5, 10))

    # Keep temp files option
    keep_temp_var = tk.BooleanVar(value=False)
    temp_check = ttk.Checkbutton(settings_frame, text="Keep temporary files (for debugging)", variable=keep_temp_var)
    temp_check.grid(column=0, row=4, columnspan=2, sticky=tk.W, pady=(5, 0))

    # Quality info
    quality_frame = ttk.LabelFrame(mainframe, text="Quality Settings (Always Enabled)", padding="10 10 10 10")
    quality_frame.grid(column=0, row=4, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 15))
    
    quality_text = "✓ Audio Preprocessing (noise reduction, normalization)\n✓ Voice Activity Detection (smart segmentation)\n✓ Punctuation Restoration (proper sentences)\n✓ Optimized AI Parameters (captures all speech, ignores music filtering)"
    quality_label = ttk.Label(quality_frame, text=quality_text, font=('TkDefaultFont', 9), foreground='#006600')
    quality_label.grid(column=0, row=0, sticky=tk.W)

    # Log area
    log_frame = ttk.LabelFrame(mainframe, text="Progress Log", padding="5 5 5 5")
    log_frame.grid(column=0, row=5, columnspan=4, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(0, 15))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    
    log_text = tk.Text(log_frame, wrap=tk.WORD, height=12, font=('Consolas', 9))
    log_text.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
    
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_scroll.grid(column=1, row=0, sticky=(tk.N, tk.S))
    log_text['yscrollcommand'] = log_scroll.set
    log_text.configure(state=tk.DISABLED)

    # Add initial welcome message
    log_text.configure(state=tk.NORMAL)
    log_text.insert(tk.END, "Welcome to Speech-to-Text Transcription!\n")
    log_text.insert(tk.END, "Select a file and click 'Start Transcription' to begin.\n")
    log_text.insert(tk.END, "Output files will be saved to your Downloads folder.\n\n")
    log_text.configure(state=tk.DISABLED)

    q = queue.Queue()

    def poll_queue():
        try:
            while True:
                msg = q.get_nowait()
                log_text.configure(state=tk.NORMAL)
                log_text.insert(tk.END, msg)
                log_text.see(tk.END)
                log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        root.after(200, poll_queue)

    def start_transcription_thread():
        inp = input_var.get().strip()
        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Input Required", "Please select a valid audio or video file.")
            return

        # Clear log
        log_text.configure(state=tk.NORMAL)
        log_text.delete(1.0, tk.END)
        log_text.configure(state=tk.DISABLED)

        options = {
            "model": model_var.get(),
            "device": device_var.get(),
            "keep_temp": keep_temp_var.get(),
        }

        def worker():
            try:
                # Redirect prints from transcribe to the queue
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = QueueWriter(q)
                sys.stderr = QueueWriter(q)
                try:
                    run_transcription(inp, downloads_dir, options, q)
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
            except Exception as e:
                q.put(f"Worker error: {e}\n")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # Run button
    run_btn = ttk.Button(mainframe, text="Start Transcription", command=start_transcription_thread)
    run_btn.grid(column=1, row=6, columnspan=2, pady=(0, 10))

    poll_queue()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="GUI for optimized speech-to-text transcription")
    parser.add_argument("--input", help="input audio/video file (if provided, runs headless)")
    parser.add_argument("--outdir", help="output folder (defaults to Downloads)")
    parser.add_argument("--model", default="medium", help="whisper model: medium or large")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    outdir = args.outdir or DEFAULT_DOWNLOADS

    if args.input:
        # Run headless
        q = queue.Queue()
        options = {
            "model": args.model,
            "device": args.device,
            "keep_temp": args.keep_temp,
        }
        
        def runner():
            run_transcription(args.input, outdir, options, q)
        
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
