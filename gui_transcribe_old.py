"""Simple GUI and CLI wrapper for `transcribe.transcribe_file`.

Usage:
  - GUI mode (default): python gui_transcribe.py
  - CLI mode: python gui_transcribe.py --input "C:\\path\\file.mp4" --outdir "C:\\path\\dest"

The GUI defaults output folder to the user's Downloads folder.
"""
import argparse
import os
import threading
import queue
import sys
from pathlib import Path

# Defer importing heavy modules (whisper/torch) until we actually run a transcription.
# This allows --help and launching the GUI quickly without loading models.

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    tk = None

# Use os.path.normpath to ensure proper Windows backslashes
DEFAULT_DOWNLOADS = os.path.normpath(os.path.join(os.path.expanduser("~"), "Downloads"))

# Small helper to run transcription (imports transcribe lazily)
def run_transcription(input_file: str, outdir: str, options: dict, output_queue: queue.Queue):
    """Call transcribe.transcribe_file with given args and stream prints to output_queue."""
    try:
        # Import here to avoid heavy imports during GUI startup
        from transcribe import transcribe_file
    except Exception as e:
        output_queue.put(f"Failed to import transcribe module: {e}\n")
        return

    try:
        output_queue.put(f"Starting optimized transcription for: {input_file}\n")
        # Use optimized settings (preprocessing, VAD, punctuation always enabled)
        out_txt = transcribe_file(input_file,
                                  model_name=options.get("model", "medium"),
                                  keep_temp=options.get("keep_temp", False),
                                  bitrate="192k",
                                  device_preference=options.get("device", "auto"),
                                  output_dir=outdir)
        output_queue.put(f"Transcription finished. Files saved to Downloads folder.\n")
    except Exception as e:
        output_queue.put(f"Transcription failed: {e}\n")


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
    root.title("Speech2Text GUI")
    root.geometry("720x520")

    mainframe = ttk.Frame(root, padding="8 8 8 8")
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Input file
    input_var = tk.StringVar()
    ttk.Label(mainframe, text="Input file:").grid(column=0, row=0, sticky=tk.W)
    input_entry = ttk.Entry(mainframe, textvariable=input_var, width=70)
    input_entry.grid(column=1, row=0, columnspan=3, sticky=(tk.W, tk.E))

    def browse_input():
        p = filedialog.askopenfilename(title="Select audio/video file")
        if p:
            input_var.set(p)

    ttk.Button(mainframe, text="Browse...", command=browse_input).grid(column=4, row=0)

    # Output dir - ensure consistent backslash formatting
    display_outdir = os.path.normpath(str(default_outdir)) if default_outdir else DEFAULT_DOWNLOADS
    outdir_var = tk.StringVar(value=display_outdir)
    ttk.Label(mainframe, text="Output folder:").grid(column=0, row=1, sticky=tk.W)
    out_entry = ttk.Entry(mainframe, textvariable=outdir_var, width=70)
    out_entry.grid(column=1, row=1, columnspan=3, sticky=(tk.W, tk.E))

    def browse_outdir():
        p = filedialog.askdirectory(title="Select output folder", initialdir=outdir_var.get())
        if p:
            # Normalize the path to use proper Windows backslashes
            outdir_var.set(os.path.normpath(p))

    ttk.Button(mainframe, text="Browse...", command=browse_outdir).grid(column=4, row=1)

    # Options
    opts_frame = ttk.LabelFrame(mainframe, text="Options")
    opts_frame.grid(column=0, row=2, columnspan=5, pady=8, sticky=(tk.W, tk.E))

    model_var = tk.StringVar(value="medium")
    ttk.Label(opts_frame, text="Model:").grid(column=0, row=0, sticky=tk.W)
    model_combo = ttk.Combobox(opts_frame, textvariable=model_var, values=("base", "small", "medium", "large"), state="readonly", width=10)
    model_combo.grid(column=1, row=0, sticky=tk.W)

    preprocess_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts_frame, text="Preprocess (denoise/normalize)", variable=preprocess_var).grid(column=2, row=0, sticky=tk.W, padx=8)

    vad_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts_frame, text="Use VAD segmentation", variable=vad_var).grid(column=0, row=1, sticky=tk.W, pady=4)

    punctuate_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts_frame, text="Restore punctuation", variable=punctuate_var).grid(column=1, row=1, sticky=tk.W, pady=4)

    keep_temp_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts_frame, text="Keep temporary files", variable=keep_temp_var).grid(column=0, row=2, sticky=tk.W, pady=4)

    aggressive_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts_frame, text="Aggressive mode (ignore music/copyright)", variable=aggressive_var).grid(column=1, row=2, sticky=tk.W, pady=4)

    bitrate_var = tk.StringVar(value="192k")
    ttk.Label(opts_frame, text="Bitrate:").grid(column=3, row=0, sticky=tk.W, padx=8)
    bitrate_entry = ttk.Entry(opts_frame, textvariable=bitrate_var, width=8)
    bitrate_entry.grid(column=4, row=0, sticky=tk.W)

    device_var = tk.StringVar(value="auto")
    ttk.Label(opts_frame, text="Processing:").grid(column=3, row=1, sticky=tk.W, padx=8)
    device_combo = ttk.Combobox(opts_frame, textvariable=device_var, values=("auto", "cpu", "cuda", "dml"), state="readonly", width=12)
    device_combo.grid(column=4, row=1, sticky=tk.W)

    # Add help text for device options
    help_text = "auto = Best available • cpu = CPU only • cuda = NVIDIA GPU • dml = DirectML (AMD/Intel/NVIDIA)"
    ttk.Label(opts_frame, text=help_text, font=('TkDefaultFont', 7), foreground='#666666').grid(column=0, row=3, columnspan=5, sticky=tk.W, pady=(4,0))

    # Log area
    log_frame = ttk.LabelFrame(mainframe, text="Log")
    log_frame.grid(column=0, row=3, columnspan=5, pady=8, sticky=(tk.N, tk.S, tk.E, tk.W))
    mainframe.rowconfigure(3, weight=1)
    log_text = tk.Text(log_frame, wrap=tk.WORD, height=15)
    log_text.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_scroll.grid(column=1, row=0, sticky=(tk.N, tk.S))
    log_text['yscrollcommand'] = log_scroll.set
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
        outd = outdir_var.get().strip() or DEFAULT_DOWNLOADS
        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Input required", "Please select a valid input audio/video file.")
            return
        if not os.path.isdir(outd):
            try:
                os.makedirs(outd, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Output folder error", f"Failed to create output folder: {e}")
                return

        options = {
            "model": model_var.get(),
            "preprocess": preprocess_var.get(),
            "vad": vad_var.get(),
            "punctuate": punctuate_var.get(),
            "keep_temp": keep_temp_var.get(),
            "bitrate": bitrate_var.get(),
            "device": device_var.get(),
            "aggressive": aggressive_var.get(),
        }

        # No need to change working dir since we pass output_dir explicitly
        def worker():
            try:
                # Redirect prints from transcribe to the queue
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = QueueWriter(q)
                sys.stderr = QueueWriter(q)
                try:
                    run_transcription(inp, outd, options, q)
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
            except Exception as e:
                q.put(f"Worker error: {e}\n")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    run_btn = ttk.Button(mainframe, text="Run", command=start_transcription_thread)
    run_btn.grid(column=2, row=4, pady=8)

    poll_queue()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="GUI/CLI wrapper for transcription")
    parser.add_argument("--input", help="input audio/video file (if provided, runs headless)")
    parser.add_argument("--outdir", help="output folder (defaults to Downloads)")
    parser.add_argument("--model", default="base", help="whisper model to use")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--punctuate", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument("--bitrate", default="192k")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    outdir = args.outdir or DEFAULT_DOWNLOADS

    if args.input:
        # run headless
        q = queue.Queue()
        options = {
            "model": args.model,
            "preprocess": args.preprocess,
            "vad": args.vad,
            "punctuate": args.punctuate,
            "keep_temp": args.keep_temp,
            "bitrate": args.bitrate,
            "device": args.device,
            "aggressive": args.aggressive,
        }
        # run and print queue messages
        def runner():
            run_transcription(args.input, outdir, options, q)
        t = threading.Thread(target=runner)
        t.start()
        # print live
        while t.is_alive() or not q.empty():
            try:
                msg = q.get(timeout=0.2)
                print(msg, end="")
            except queue.Empty:
                pass
        return

    # launch GUI
    launch_gui(default_outdir=outdir)


if __name__ == "__main__":
    main()
