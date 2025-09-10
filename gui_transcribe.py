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

def run_transcription(input_file: str, outdir: Optional[str], output_queue: queue.Queue, *, threads_override: Optional[int] = None, batch_size_override: Optional[int] = None):
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
        output_queue.put("Using Large model with auto device detection\n")
        output_queue.put("Direct processing, maximum threads\n")

        out_txt = transcribe_file_simple_auto(
            input_file,
            output_dir=target_outdir,
            threads_override=threads_override,
            batch_size_override=batch_size_override,
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


def run_batch_transcription(paths: List[str], outdir_override: Optional[str], output_queue: queue.Queue, *, threads_override: Optional[int] = None, batch_size_override: Optional[int] = None):
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
            run_transcription(p, outdir_override, output_queue, threads_override=threads_override, batch_size_override=batch_size_override)
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


def launch_gui(default_outdir: Optional[str] = None, *, default_threads: Optional[int] = None, default_batch_size: Optional[int] = None):
    if not TKINTER_AVAILABLE:
        print("Tkinter is not available in this Python build.")
        return

    # Help static analyzers: ensure tkinter symbols are non-None beyond this point
    assert tk is not None and ttk is not None and filedialog is not None and messagebox is not None

    # Memory cleanup on GUI launch
    memory_cleanup_info = []
    print("Performing memory cleanup...")
    try:
        import gc
        import psutil
        import os

        # Force garbage collection
        collected = gc.collect()
        print(f"Garbage collector freed {collected} objects")
        memory_cleanup_info.append(f"Garbage collector freed {collected} objects")

        # Get current process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
        memory_cleanup_info.append(f"Current memory usage: {memory_mb:.1f} MB")

        # Clear any cached imports that might be holding memory (exclude torch)
        import sys
        modules_to_clear = []
        for module_name in sys.modules:
            if module_name.startswith(('whisper', 'transformers')):
                modules_to_clear.append(module_name)

        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
                print(f"Cleared cached module: {module_name}")
                memory_cleanup_info.append(f"Cleared cached module: {module_name}")

        # Force another garbage collection after clearing modules
        collected2 = gc.collect()
        print(f"Additional garbage collection freed {collected2} objects")
        memory_cleanup_info.append(f"Additional garbage collection freed {collected2} objects")

        print("Memory cleanup completed. GUI starting...")
        memory_cleanup_info.append("Memory cleanup completed successfully")

    except ImportError:
        print("Memory cleanup skipped (psutil not available)")
        memory_cleanup_info.append("Memory cleanup skipped (psutil not available)")
    except Exception as e:
        print(f"Memory cleanup warning: {e}")
        memory_cleanup_info.append(f"Memory cleanup warning: {e}")

    try:

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

        # Clean file/folder selection section
        file_section_label = ttk.Label(mainframe, text="File Selection", style='Section.TLabel')
        file_section_label.grid(column=0, row=1, columnspan=3, sticky="w", pady=(0, 15))

        # File input area with subtle background
        file_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
        file_frame.grid(column=0, row=2, columnspan=3, sticky="ew", pady=(0, 25))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Audio/Video File or Folder:", background='white', foreground='#374151',
                 font=('Segoe UI', 10, 'bold')).grid(column=0, row=0, sticky="w", padx=20, pady=(20, 10))

        input_var = tk.StringVar()
        input_entry = tk.Entry(file_frame, textvariable=input_var, font=('Segoe UI', 10),
                              relief='flat', borderwidth=1, bg='white', fg='#2c3e50',
                              insertbackground='#2c3e50')
        input_entry.grid(column=0, row=1, columnspan=2, sticky="ew", padx=20, pady=(0, 20))

        def browse_input():
            import os
            filetypes = [
                ("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma"),
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("All files", "*.*")
            ]
            p = filedialog.askopenfilename(title="Select audio/video file", filetypes=filetypes)
            if p:
                input_var.set(p)
                status_label.config(text="Selected file: {}".format(os.path.basename(p)), foreground='#059669')

        def browse_folder():
            import os
            d = filedialog.askdirectory(title="Select folder for batch processing")
            if d:
                input_var.set(d)
                # Count eligible files
                try:
                    files = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
                    status_label.config(
                        text="Selected folder: {} ({} eligible files)".format(os.path.basename(d), len(files)),
                        foreground='#059669'
                    )
                except Exception:
                    status_label.config(text="Selected folder: {}".format(os.path.basename(d)), foreground='#059669')

        browse_btn = tk.Button(file_frame, text="Browse File", command=browse_input,
                              font=('Segoe UI', 10, 'bold'), bg='#007acc', fg='white',
                              relief='flat', borderwidth=0, padx=20, pady=8,
                              activebackground='#0056b3', activeforeground='white')
        browse_btn.grid(column=2, row=1, sticky="e", padx=(10, 10))

        browse_folder_btn = tk.Button(file_frame, text="Browse Folder", command=browse_folder,
                              font=('Segoe UI', 10, 'bold'), bg='#0ea5e9', fg='white',
                              relief='flat', borderwidth=0, padx=20, pady=8,
                              activebackground='#0284c7', activeforeground='white')
        browse_folder_btn.grid(column=2, row=2, sticky="e", padx=(10, 20), pady=(0, 20))

        # Status label
        status_label = ttk.Label(file_frame, text="No file or folder selected",
                                font=('Segoe UI', 9), foreground='#6b7280', background='white')
        status_label.grid(column=0, row=3, columnspan=3, sticky="w", padx=20, pady=(0, 10))

    # Clean settings section
    settings_section_label = ttk.Label(mainframe, text="Settings", style='Section.TLabel')
    settings_section_label.grid(column=0, row=3, columnspan=3, sticky="w", pady=(0, 15))

    # Settings area with subtle background
    settings_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
    settings_frame.grid(column=0, row=4, columnspan=3, sticky="ew", pady=(0, 25))

    settings_text = """Large AI Model (Professional Quality)
Auto Device Detection (CUDA -> DirectML -> CPU)
Direct Audio Processing
Maximum CPU Threads (RAM-Optimized)
Output Directory: Same as source file(s)
Full Audio Capture (All Content Preserved)"""

    settings_label = tk.Text(settings_frame, height=6, wrap=tk.WORD, font=('Segoe UI', 10),
                                bg='white', fg='#374151', borderwidth=0, highlightthickness=0,
                                padx=20, pady=20)
    settings_label.insert(tk.END, settings_text)
    settings_label.config(state=tk.DISABLED)
    settings_label.grid(column=0, row=0, sticky="ew")

    # Advanced performance overrides (optional)
    overrides_frame = tk.Frame(settings_frame, bg='white')
    overrides_frame.grid(column=0, row=1, sticky="ew", padx=20, pady=(0, 20))
    overrides_frame.columnconfigure(1, weight=1)

        tk.Label(overrides_frame, text="CPU Threads (optional):", bg='white', fg='#374151', font=('Segoe UI', 10)).grid(column=0, row=0, sticky='w', pady=(5, 5))
        threads_var = tk.StringVar()
        if default_threads and isinstance(default_threads, int) and default_threads > 0:
            threads_var.set(str(default_threads))
        threads_entry = tk.Entry(overrides_frame, textvariable=threads_var, width=10, bg='#f9fafb', fg='#111827', relief='flat')
        threads_entry.grid(column=1, row=0, sticky='w', padx=(10, 0))

        tk.Label(overrides_frame, text="Batch Size (optional):", bg='white', fg='#374151', font=('Segoe UI', 10)).grid(column=0, row=1, sticky='w', pady=(5, 5))
        batch_var = tk.StringVar()
        if default_batch_size and isinstance(default_batch_size, int) and default_batch_size > 0:
            batch_var.set(str(default_batch_size))
        batch_entry = tk.Entry(overrides_frame, textvariable=batch_var, width=10, bg='#f9fafb', fg='#111827', relief='flat')
        batch_entry.grid(column=1, row=1, sticky='w', padx=(10, 0))

    tip_label = tk.Label(overrides_frame, text="Leave blank for Auto. You can also set env vars TRANSCRIBE_THREADS and TRANSCRIBE_BATCH_SIZE.", bg='white', fg='#6b7280', font=('Segoe UI', 8))
    tip_label.grid(column=0, row=2, columnspan=2, sticky='w', pady=(6, 0))

        # Log area with modern design
        log_section_label = ttk.Label(mainframe, text="Activity Log", style='Section.TLabel')
        log_section_label.grid(column=0, row=5, columnspan=3, sticky="w", pady=(0, 15))

        log_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
        log_frame.grid(column=0, row=6, columnspan=3, sticky="nsew", pady=(0, 25))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        log_text = tk.Text(log_frame, wrap=tk.WORD, height=25, font=('Segoe UI', 9),
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

        # Add memory cleanup information to the log
        if memory_cleanup_info:
            log_text.insert('end', "System Memory Cleanup Results:\n", "info")
            for info in memory_cleanup_info:
                log_text.insert('end', f"  {info}\n")
            log_text.insert('end', "\n")

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
            import os
            print("DEBUG: start_transcription_thread called", file=sys.__stderr__)

            inp = input_var.get().strip()
            print("DEBUG: Input file path: '{}'".format(inp), file=sys.__stderr__)
            print("DEBUG: Path exists: {}".format(os.path.exists(inp) if inp else 'N/A'), file=sys.__stderr__)

            if not inp or not os.path.exists(inp):
                error_msg = "Please select a valid file or folder.\nPath: '{}'\nExists: {}".format(inp, os.path.exists(inp) if inp else False)
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

            # Disable run button during processing
            run_btn.configure(state='disabled')

            def worker():
                import sys
                import os
                try:
                    # Send initial status message
                    q.put("Initializing transcription process...\n")

                    # Redirect prints from transcribe to the queue
                    old_out, old_err = sys.stdout, sys.stderr
                    sys.stdout = QueueWriter(q)
                    sys.stderr = QueueWriter(q)
                    try:
                        # Parse overrides from GUI fields
                        def _parse_int_opt(s: str) -> Optional[int]:
                            try:
                                v = int(s.strip())
                                return v if v > 0 else None
                            except Exception:
                                return None

                        threads_override = _parse_int_opt(threads_var.get())
                        batch_override = _parse_int_opt(batch_var.get())

                        if os.path.isdir(inp):
                            # Build list of eligible files (non-recursive)
                            files = []
                            try:
                                for name in sorted(os.listdir(inp)):
                                    full = os.path.join(inp, name)
                                    if os.path.isfile(full) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                                        files.append(full)
                            except Exception as e:
                                q.put("Failed to list folder '{}': {}\n".format(inp, e))

                            if not files:
                                q.put("No supported media files found in the selected folder.\n")
                            else:
                                run_batch_transcription(
                                    files,
                                    outdir_override=None,
                                    output_queue=q,
                                    threads_override=threads_override,
                                    batch_size_override=batch_override,
                                )
                        else:
                            # Single file mode, save next to source file
                            run_transcription(
                                inp,
                                outdir=None,
                                output_queue=q,
                                threads_override=threads_override,
                                batch_size_override=batch_override,
                            )
                    finally:
                        sys.stdout = old_out
                        sys.stderr = old_err

                    # Send completion message
                    q.put("Transcription process finished!\n")

                except Exception as e:
                    import traceback
                    error_msg = "Worker error: {}\n{}".format(e, traceback.format_exc())
                    q.put(error_msg)
                finally:
                    # Re-enable button when done (from main thread via queue)
                    def enable_btn():
                        run_btn.configure(state='normal')
                    root.after(0, enable_btn)

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

    except Exception as e:
        import traceback
        error_msg = "GUI Error: {}\n{}".format(e, traceback.format_exc())
        print(error_msg)
        # Try to show error in a messagebox if possible
        try:
            if tk and tk.Tk:
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                messagebox.showerror("GUI Error", error_msg)
                root.destroy()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Speech to Text Transcription Tool v1.0Beta")
    parser.add_argument("--input", help="input audio/video file OR folder (if provided, runs headless)")
    parser.add_argument("--outdir", help="optional output folder override; defaults to saving next to source file(s)")
    parser.add_argument("--gui", action="store_true", help="launch GUI mode")
    parser.add_argument("--threads", type=int, help="Override CPU threads for PyTorch/OMP/MKL")
    parser.add_argument("--batch-size", type=int, help="Override inference batch size")
    args = parser.parse_args()

    outdir = args.outdir or None  # default to same-as-source when not specified

    if args.gui or "--gui" in sys.argv:
        # Launch GUI in this process
        launch_gui(default_outdir=outdir, default_threads=args.threads, default_batch_size=getattr(args, "batch_size", None))
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
                    run_batch_transcription(files, outdir_override=outdir, output_queue=q, threads_override=args.threads, batch_size_override=getattr(args, "batch_size", None))
            else:
                run_transcription(p, outdir, q, threads_override=args.threads, batch_size_override=getattr(args, "batch_size", None))

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
