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
import subprocess
from typing import Optional, List

# GUI toolkit
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
TKINTER_AVAILABLE = True

# Supported file extensions for batch processing
SUPPORTED_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma",
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"
)

# Default Downloads folder retained for compatibility with CLI --outdir override
DEFAULT_DOWNLOADS = os.path.normpath(os.path.join(os.path.expanduser("~"), "Downloads"))

def run_transcription(input_file: str, outdir: Optional[str], output_queue: queue.Queue, *, threads_override: Optional[int] = None):
    """Run transcription for a single file with large model and auto device detection.

    Output directory defaults to the source file directory if not provided.
    """
    try:
        # Memory cleanup BEFORE processing each file
        print("🧹 Performing memory cleanup before file processing...")
        try:
            import gc
            import psutil
            import torch
            import sys

            # Force garbage collection
            collected = gc.collect()
            print(f"   Garbage collected: {collected} objects")

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
                print("   GPU cache cleared")

            # Clear any cached modules that might be holding memory (exclude torch)
            modules_to_clear = []
            for module_name in sys.modules:
                if module_name.startswith(('whisper', 'transformers')):
                    modules_to_clear.append(module_name)

            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    print(f"   Cleared cached module: {module_name}")

            # Force another garbage collection
            collected2 = gc.collect()
            print(f"   Additional garbage collected: {collected2} objects")

            # Monitor memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"   Memory usage before processing: {memory_mb:.1f} MB")

            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"   GPU memory before processing: {gpu_memory:.1f} GB")
                except:
                    pass

        except ImportError:
            print("   Memory cleanup skipped (missing dependencies)")
        except Exception as e:
            print(f"   Memory cleanup warning: {e}")
        # Import and run the transcription
        from transcribe_optimised import transcribe_file_simple_auto

        # Determine output dir (default to same folder as source file)
        target_outdir = outdir if outdir else os.path.dirname(input_file)

        output_queue.put("Starting transcription for: {}\n".format(os.path.basename(input_file)))
        output_queue.put("Using Whisper large-v3-turbo (auto device selection)\n")
        output_queue.put("Direct processing, maximum threads\n")

        out_txt = transcribe_file_simple_auto(
            input_file,
            output_dir=target_outdir,
            threads_override=threads_override,
        )

        if out_txt and os.path.exists(out_txt):
            output_queue.put("Transcription complete! Files saved next to the source file.\n")
            output_queue.put("  Text file: {}\n".format(os.path.basename(out_txt)))
            output_queue.put("  Word document: {}\n".format(os.path.basename(out_txt.replace('.txt', '.docx'))))
        else:
            output_queue.put("Transcription failed or no output generated.\n")

    except Exception as e:
        output_queue.put("Transcription failed: {}\n".format(e))
        import traceback
        output_queue.put("Error details: {}\n".format(traceback.format_exc()))

    finally:
        # Memory cleanup AFTER processing each file
        print("🧹 Performing memory cleanup after file processing...")
        try:
            import gc
            import psutil
            import torch
            import sys

            # Force garbage collection
            collected = gc.collect()
            print(f"   Post-processing garbage collected: {collected} objects")

            # Clear GPU cache aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("   GPU cache cleared after processing")

                # Force release of any remaining GPU memory
                try:
                    torch.cuda.empty_cache()
                    if torch.cuda.memory_allocated() > 0:
                        print("   Warning: Some GPU memory still allocated")
                except:
                    pass

            # Clear cached modules again (exclude torch)
            modules_to_clear = []
            for module_name in sys.modules:
                if module_name.startswith(('whisper', 'transformers')):
                    modules_to_clear.append(module_name)

            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    try:
                        del sys.modules[module_name]
                        print(f"   Cleared cached module: {module_name}")
                    except:
                        pass  # Some modules can't be deleted

            # Final garbage collection
            collected_final = gc.collect()
            print(f"   Final garbage collected: {collected_final} objects")

            # Monitor final memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"   Memory usage after processing: {memory_mb:.1f} MB")

            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"   GPU memory after processing: {gpu_memory:.1f} GB")
                except:
                    pass

        except Exception as e:
            print(f"   Post-processing cleanup warning: {e}")


def run_batch_transcription(paths: List[str], outdir_override: Optional[str], output_queue: queue.Queue, *, threads_override: Optional[int] = None):
    """Run batch transcription sequentially over provided file paths.

    Each file is saved next to its source (unless outdir_override is provided).
    Continues processing even if individual files fail.
    """
    total = len(paths)
    output_queue.put("Batch mode: {} eligible files queued.\n".format(total))
    successful = 0
    failed = 0

    for idx, p in enumerate(paths, start=1):
        output_queue.put("\n[{} / {}] Processing: {}\n".format(idx, total, os.path.basename(p)))
        try:
            run_transcription(p, outdir_override, output_queue, threads_override=threads_override)
            successful += 1
        except Exception as e:
            failed += 1
            output_queue.put("Failed to process '{}': {}\n".format(os.path.basename(p), str(e)))
            output_queue.put("Continuing with next file...\n")

    output_queue.put("\nBatch processing complete!\n")
    output_queue.put("Successfully processed: {} files\n".format(successful))
    if failed > 0:
        output_queue.put("Failed: {} files\n".format(failed))


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


def launch_gui(default_outdir: Optional[str] = None, *, default_threads: Optional[int] = None):
    if not TKINTER_AVAILABLE:
        print("Tkinter is not available in this Python build.")
        return

    assert tk is not None and ttk is not None and filedialog is not None and messagebox is not None

    # Memory cleanup summary for the log
    memory_cleanup_info = []
    try:
        import gc, psutil
        collected = gc.collect()
        memory_cleanup_info.append(f"Garbage collector freed {collected} objects")
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_cleanup_info.append(f"Current memory usage: {mem_mb:.1f} MB")
        # Clear whisper/transformers modules
        to_clear = [m for m in list(sys.modules.keys()) if m.startswith(('whisper','transformers'))]
        for m in to_clear:
            sys.modules.pop(m, None)
        gc.collect()
        memory_cleanup_info.append("Module caches cleared (whisper/transformers)")
    except Exception as e:
        memory_cleanup_info.append(f"Memory cleanup warning: {e}")

    try:
        root = tk.Tk()
        root.title("Speech to Text Transcription Tool v1.0Beta")
        # Open large enough to show the Start button; also allow maximizing
        root.geometry("1100x800")
        root.minsize(960, 720)
        root.configure(bg='#f8f9fa')
        # Make the root grid resizable so children can expand
        try:
            root.grid_columnconfigure(0, weight=1)
            root.grid_rowconfigure(0, weight=1)
            # Maximize by default on Windows to ensure button visibility
            root.state('zoomed')
        except Exception:
            pass

        style = ttk.Style()
        style.configure('Clean.TFrame', background='#f8f9fa')
        style.configure('Clean.TLabel', background='#f8f9fa', foreground='#2c3e50', font=('Segoe UI', 11))
        style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'), foreground='#1a365d', background='#f8f9fa')
        style.configure('Subtitle.TLabel', font=('Segoe UI', 10), foreground='#64748b', background='#f8f9fa')
        style.configure('Section.TLabel', font=('Segoe UI', 13, 'bold'), foreground='#374151', background='#f8f9fa')

        mainframe = ttk.Frame(root, style='Clean.TFrame', padding="30 30 30 30")
        mainframe.grid(column=0, row=0, sticky="nsew")
        # Allow horizontal expansion across all columns
        mainframe.columnconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=1)
        mainframe.columnconfigure(2, weight=1)
        mainframe.rowconfigure(6, weight=1)
        mainframe.rowconfigure(7, minsize=60)

        # Title
        title_frame = ttk.Frame(mainframe, style='Clean.TFrame')
        title_frame.grid(column=0, row=0, columnspan=3, pady=(0, 30), sticky="ew")
        ttk.Label(title_frame, text="Speech to Text Transcription Tool", style='Title.TLabel').grid(column=0, row=0, sticky="w")
        ttk.Label(title_frame, text="v1.0Beta - Professional AI-powered transcription", style='Subtitle.TLabel').grid(column=0, row=1, sticky="w", pady=(8, 0))

        # Input & Settings (combined)
        ttk.Label(mainframe, text="Input & Settings", style='Section.TLabel').grid(column=0, row=1, columnspan=3, sticky="w", pady=(0, 15))
        combined_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
        combined_frame.grid(column=0, row=2, columnspan=3, sticky="ew", pady=(0, 25))
        # Column 1 expands; buttons stay compact
        combined_frame.columnconfigure(1, weight=1)

        # Row 0: File/Folder selection line
        ttk.Label(combined_frame, text="Audio/Video File or Folder:", background='white', foreground='#374151', font=('Segoe UI', 10, 'bold')).grid(column=0, row=0, sticky="w", padx=20, pady=(18, 6))
        input_var = tk.StringVar()
        tk.Entry(combined_frame, textvariable=input_var, font=('Segoe UI', 10), relief='flat', borderwidth=1, bg='#f9fafb', fg='#111827', insertbackground='#111827').grid(column=1, row=0, sticky="ew", padx=(12, 6), pady=(18, 6))

        # Define handlers before creating buttons
        def browse_input():
            types = [("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma"),("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),("All files", "*.*")]
            p = filedialog.askopenfilename(title="Select audio/video file", filetypes=types)
            if p:
                input_var.set(p)
                status_label.config(text=f"Selected file: {os.path.basename(p)}", foreground='#059669')

        def browse_folder():
            d = filedialog.askdirectory(title="Select folder for batch processing")
            if d:
                input_var.set(d)
                try:
                    files = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
                    status_label.config(text=f"Selected folder: {os.path.basename(d)} ({len(files)} eligible files)", foreground='#059669')
                except Exception:
                    status_label.config(text=f"Selected folder: {os.path.basename(d)}", foreground='#059669')

        tk.Button(combined_frame, text="Browse File", command=browse_input, font=('Segoe UI', 10, 'bold'), bg='#007acc', fg='white', relief='flat', borderwidth=0, padx=16, pady=6, activebackground='#0056b3', activeforeground='white').grid(column=2, row=0, sticky="e", padx=(6, 6), pady=(18, 6))
        tk.Button(combined_frame, text="Browse Folder", command=browse_folder, font=('Segoe UI', 10, 'bold'), bg='#0ea5e9', fg='white', relief='flat', borderwidth=0, padx=16, pady=6, activebackground='#0284c7', activeforeground='white').grid(column=3, row=0, sticky="e", padx=(6, 20), pady=(18, 6))

        # Row 1: Status line
        status_label = ttk.Label(combined_frame, text="No file or folder selected", font=('Segoe UI', 9), foreground='#6b7280', background='white')
        status_label.grid(column=0, row=1, columnspan=4, sticky="w", padx=20, pady=(0, 8))

        # Thin separator
        ttk.Separator(combined_frame, orient='horizontal').grid(column=0, row=2, columnspan=4, sticky='ew', padx=20, pady=(4, 10))

    # Row 3: Threads override
        tk.Label(combined_frame, text="CPU Threads (optional):", bg='white', fg='#374151', font=('Segoe UI', 10)).grid(column=0, row=3, sticky='w', padx=20, pady=(4, 6))
        threads_var = tk.StringVar(value=str(default_threads) if default_threads and default_threads > 0 else "")
        tk.Entry(combined_frame, textvariable=threads_var, width=10, bg='#f9fafb', fg='#111827', relief='flat').grid(column=1, row=3, sticky='w', padx=(12, 6), pady=(4, 6))
    tk.Label(combined_frame, text="Leave blank for Auto. Or set TRANSCRIBE_THREADS.", bg='white', fg='#6b7280', font=('Segoe UI', 8)).grid(column=2, row=3, columnspan=2, sticky='w', padx=(6, 20), pady=(4, 6))

    # Row 3b: Max performance mode
    max_perf_var = tk.IntVar(value=1)
    tk.Checkbutton(combined_frame, text="Maximise performance (use all cores, high priority, aggressive VRAM)", variable=max_perf_var, bg='white', fg='#374151', selectcolor='white', activebackground='white').grid(column=0, row=4, columnspan=4, sticky='w', padx=20, pady=(0, 8))

        # Row 4: Compact description
        desc = (
            "Whisper large-v3-turbo • Auto device (CUDA/DirectML/CPU) • Direct audio • RAM-optimized threads\n"
            "Outputs saved next to source file(s)."
        )
        ttk.Label(combined_frame, text=desc, background='white', foreground='#374151', font=('Segoe UI', 9), wraplength=920, justify='left').grid(column=0, row=5, columnspan=4, sticky='w', padx=20, pady=(8, 16))

        # Handlers attached to the buttons above

        # Log
        ttk.Label(mainframe, text="Activity Log", style='Section.TLabel').grid(column=0, row=5, columnspan=3, sticky="w", pady=(0, 15))
        log_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
        log_frame.grid(column=0, row=6, columnspan=3, sticky="nsew", pady=(0, 25))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_text = tk.Text(log_frame, wrap=tk.WORD, height=25, font=('Segoe UI', 9), bg='white', fg='#2c3e50', borderwidth=0, highlightthickness=0, insertbackground='#2c3e50')
        log_text.grid(column=0, row=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
        log_scroll.grid(column=1, row=0, sticky="ns")
        log_text['yscrollcommand'] = log_scroll.set
        log_text.configure(state='normal')
        log_text.insert('end', "Welcome to Speech to Text Transcription Tool v1.0Beta!\n", "title")
        log_text.insert('end', "Select an audio or video file and click 'Start Transcription' to begin.\n\n")
        if memory_cleanup_info:
            log_text.insert('end', "System Memory Cleanup Results:\n", "info")
            for info in memory_cleanup_info:
                log_text.insert('end', f"  {info}\n")
            log_text.insert('end', "\n")
        log_text.configure(state='disabled')
        log_text.tag_configure("title", foreground="#007acc", font=('Segoe UI', 9, 'bold'))
        log_text.tag_configure("info", foreground="#60a5fa")

        q = queue.Queue()

        def format_log_message(msg: str) -> str:
            return msg

        def poll_queue():
            try:
                while True:
                    msg = q.get_nowait()
                    log_text.configure(state='normal')
                    log_text.insert('end', format_log_message(msg))
                    log_text.see('end')
                    log_text.configure(state='disabled')
            except queue.Empty:
                pass
            root.after(200, poll_queue)

        def start_transcription_thread():
            inp = input_var.get().strip()
            if not inp or not os.path.exists(inp):
                messagebox.showerror("Input Required", f"Please select a valid file or folder.\nPath: '{inp}'\nExists: {os.path.exists(inp) if inp else False}")
                return

            log_text.configure(state='normal')
            log_text.delete(1.0, 'end')
            log_text.insert('end', "Starting transcription...\n")
            log_text.configure(state='disabled')
            run_btn.configure(state='disabled')

            def worker():
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = QueueWriter(q)
                sys.stderr = QueueWriter(q)
                try:
                    def _parse_int_opt(s: str) -> Optional[int]:
                        try:
                            v = int(s.strip())
                            return v if v > 0 else None
                        except Exception:
                            return None

                    thr = _parse_int_opt(threads_var.get())

                    # Apply max performance env for the worker
                    try:
                        if max_perf_var.get() == 1:
                            os.environ["TRANSCRIBE_MAX_PERF"] = "1"
                        else:
                            os.environ.pop("TRANSCRIBE_MAX_PERF", None)
                    except Exception:
                        pass

                    if os.path.isdir(inp):
                        files = []
                        try:
                            for name in sorted(os.listdir(inp)):
                                full = os.path.join(inp, name)
                                if os.path.isfile(full) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                                    files.append(full)
                        except Exception as e:
                            q.put(f"Failed to list folder '{inp}': {e}\n")
                        if not files:
                            q.put("No supported media files found in the selected folder.\n")
                        else:
                            run_batch_transcription(files, outdir_override=None, output_queue=q, threads_override=thr)
                    else:
                        run_transcription(inp, outdir=None, output_queue=q, threads_override=thr)
                    q.put("Transcription process finished!\n")
                except Exception as e:
                    import traceback
                    q.put(f"Worker error: {e}\n{traceback.format_exc()}")
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
                    root.after(0, lambda: run_btn.configure(state='normal'))

            threading.Thread(target=worker, daemon=True).start()

        run_btn = tk.Button(mainframe, text="Start Transcription", command=start_transcription_thread, font=('Segoe UI', 12, 'bold'), bg='#007acc', fg='white', relief='flat', borderwidth=0, padx=30, pady=12, activebackground='#0056b3', activeforeground='white', cursor='hand2')
        run_btn.grid(column=1, row=7, pady=(10, 0))
        poll_queue()
        root.mainloop()
    except Exception as e:
        import traceback
        error_msg = f"GUI Error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        try:
            if tk and tk.Tk:
                r = tk.Tk(); r.withdraw(); messagebox.showerror("GUI Error", error_msg); r.destroy()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Speech to Text Transcription Tool v1.0Beta")
    parser.add_argument("--input", help="input audio/video file OR folder (if provided, runs headless)")
    parser.add_argument("--outdir", help="optional output folder override; defaults to saving next to source file(s)")
    parser.add_argument("--gui", action="store_true", help="launch GUI mode")
    parser.add_argument("--threads", type=int, help="Override CPU threads for PyTorch/OMP/MKL")
    args = parser.parse_args()

    outdir = args.outdir or None  # default to same-as-source when not specified

    if args.gui or "--gui" in sys.argv:
        # Launch GUI in this process
        launch_gui(default_outdir=outdir, default_threads=args.threads)
        return

    if args.input:
        # Run headless
        q = queue.Queue()

        def runner():
            p = args.input
            if os.path.isdir(p):
                # Batch over folder (non-recursive)
                files = [
                    os.path.join(p, name)
                    for name in sorted(os.listdir(p))
                    if os.path.isfile(os.path.join(p, name)) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS
                ]
                if not files:
                    q.put("No supported media files found in the provided folder.\n")
                else:
                    run_batch_transcription(files, outdir_override=outdir, output_queue=q, threads_override=args.threads)
            else:
                run_transcription(p, outdir, q, threads_override=args.threads)

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

    # Launch GUI in detached process
    try:
        subprocess.run(["cmd", "/c", "start", "python", __file__, "--gui"], 
                      creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        print("GUI launched in background. You can now use the terminal for other commands.")
    except Exception as e:
        print(f"Failed to launch GUI in background: {e}")
        print("Launching GUI in foreground instead...")
        launch_gui(default_outdir=outdir)


if __name__ == "__main__":
    main()
