"""
Optimised transcription utilities with safe PyTorch lifecycle management.
This module provides a high-quality single-file transcription path that avoids
re-importing torch and aggressively clearing only model-level caches between runs.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

import os
import sys
import time
import gc
import psutil
import argparse
import multiprocessing
from typing import Any, cast, Optional

# IMPORTANT: Import torch once at module import time. Do NOT delete torch.* from sys.modules.
try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    _torch_import_error = e
else:
    _torch_import_error = None


def _ensure_torch_available():
    if torch is None:
        raise RuntimeError(f"PyTorch is required but failed to import: {_torch_import_error}")


def get_maximum_hardware_config():
    """Detect hardware and return a conservative, stable config dict."""
    _ensure_torch_available()
    torch_api = cast(Any, torch)
    cpu_cores = max(multiprocessing.cpu_count(), 1)
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    # Default: 80% of total RAM
    usable_ram_gb = max(total_ram_gb * 0.8, 1.0)
    # RAM overrides via env: prefer absolute GB then fraction
    try:
        env_ram_gb = float(os.environ.get("TRANSCRIBE_RAM_GB", "") or 0)
    except Exception:
        env_ram_gb = 0.0
    try:
        env_ram_frac = float(os.environ.get("TRANSCRIBE_RAM_FRACTION", "") or 0)
    except Exception:
        env_ram_frac = 0.0
    if env_ram_gb > 0:
        usable_ram_gb = max(1.0, min(total_ram_gb, env_ram_gb))
    elif 0.05 <= env_ram_frac <= 1.0:
        usable_ram_gb = max(1.0, total_ram_gb * env_ram_frac)

    devices = ["cpu"]
    has_cuda = False
    cuda_total_vram_gb = 0.0
    try:
        has_cuda = torch_api.cuda.is_available()
        if has_cuda:
            devices.insert(0, "cuda")
            try:
                cuda_total_vram_gb = torch_api.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                cuda_total_vram_gb = 0.0
    except Exception:
        has_cuda = False

    # DirectML availability (optional)
    dml_available = False
    try:
        import torch_directml  # type: ignore
        _ = torch_directml.device()
        dml_available = True
        if not has_cuda:
            devices.insert(0, "dml")
    except Exception:
        dml_available = False

    # Threads: be more aggressive but still bounded by RAM and practical limits
    if usable_ram_gb >= 16:
        cpu_threads = min(max(cpu_cores, 1), 32)
    elif usable_ram_gb >= 8:
        cpu_threads = min(max(cpu_cores, 1), 24)
    else:
        cpu_threads = min(max(cpu_cores, 1), 16)
    cpu_threads = max(cpu_threads, 4)

    # Environment override for threads
    try:
        env_threads = int(os.environ.get("TRANSCRIBE_THREADS", "") or 0)
    except Exception:
        env_threads = 0
    if env_threads > 0:
        cpu_threads = max(1, min(64, env_threads))

    # For stability, use 1 GPU worker. Loading multiple large models wastes VRAM for a single-file run.
    gpu_workers = 1 if has_cuda or dml_available else 0

    # VRAM overrides via env
    try:
        env_vram_gb = float(os.environ.get("TRANSCRIBE_VRAM_GB", "") or 0)
    except Exception:
        env_vram_gb = 0.0
    try:
        env_vram_frac = float(os.environ.get("TRANSCRIBE_VRAM_FRACTION", "") or 0)
    except Exception:
        env_vram_frac = 0.0
    allowed_vram_gb = cuda_total_vram_gb
    if cuda_total_vram_gb > 0:
        if env_vram_gb > 0:
            allowed_vram_gb = max(0.5, min(cuda_total_vram_gb, env_vram_gb))
        elif 0.05 <= env_vram_frac <= 1.0:
            allowed_vram_gb = max(0.5, cuda_total_vram_gb * env_vram_frac)

    cfg = {
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "total_ram_gb": total_ram_gb,
        "usable_ram_gb": usable_ram_gb,
        "devices": devices,
        "gpu_workers": gpu_workers,
        "total_workers": max(cpu_threads, gpu_workers),
        "dml_available": dml_available,
        "cuda_total_vram_gb": cuda_total_vram_gb,
        "allowed_vram_gb": allowed_vram_gb,
    }
    return cfg


def adjust_workers_for_model(config, model_name):
    """Optionally tweak worker counts based on model size. Keep it conservative."""
    cfg = dict(config)
    name = (model_name or "large").lower()
    # Larger models -> fewer CPU threads to reduce contention
    if name in ("large", "medium"):
        cfg["cpu_threads"] = min(cfg.get("cpu_threads", 8), 12)
    cfg["total_workers"] = max(cfg.get("cpu_threads", 1), cfg.get("gpu_workers", 0))
    return cfg


def force_gpu_memory_cleanup():
    """Clear GPU caches and model-related module caches without touching torch modules."""
    try:
        _ensure_torch_available()
        torch_api = cast(Any, torch)
        if torch_api.cuda.is_available():
            torch_api.cuda.empty_cache()
            try:
                torch_api.cuda.synchronize()
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️  GPU cache clear warning: {e}")

    # Clear whisper/transformers modules only (keeps torch intact)
    to_clear = [
        name for name in list(sys.modules.keys())
        if name.startswith(("whisper", "transformers"))
    ]
    for name in to_clear:
        try:
            del sys.modules[name]
        except Exception:
            pass

    gc.collect()


def transcribe_file_simple_auto(input_path, output_dir=None, *, threads_override: Optional[int] = None, batch_size_override: Optional[int] = None):
    """
    High-quality, simplified single-file transcription on best available device.
    - Device selection: CUDA > DirectML > CPU
    - No VAD; transcribe the entire file
    - Robust DOCX save with fallback
    - Safe cleanup that avoids torch re-import problems
    Returns path to the .txt file.
    """
    _ensure_torch_available()
    torch_api = cast(Any, torch)

    # Lazy imports (after torch is imported) to avoid docstring errors
    import whisper  # type: ignore
    from docx import Document  # type: ignore
    from deepmultilingualpunctuation import PunctuationModel  # type: ignore
    from transcribe import (
        get_media_duration, split_into_paragraphs, format_duration,
    )

    start_time = time.time()
    print("🚀 MAXIMUM PERFORMANCE AUTO-DETECTED TRANSCRIPTION")
    print(f"📁 Input: {os.path.basename(input_path)}")

    config = get_maximum_hardware_config()
    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    duration = get_media_duration(input_path)
    if duration:
        print(f"⏱️  Duration: {format_duration(duration)}")

    # Pre-run cleanup
    force_gpu_memory_cleanup()

    # Choose device and load one model only
    device_name = "CPU"
    model: Any = None
    chosen_device = "cpu"

    try:
        if "cuda" in config["devices"] and torch_api.cuda.is_available():
            chosen_device = "cuda"
            device_name = f"CUDA GPU ({torch_api.cuda.get_device_name(0)})"
            print("🎯 Device: CUDA GPU")
            # Apply CUDA per-process memory fraction if an allowed VRAM cap is set
            try:
                total_vram = float(config.get("cuda_total_vram_gb") or 0.0)
                allowed_vram = float(config.get("allowed_vram_gb") or 0.0)
                if total_vram > 0 and 0.5 <= allowed_vram < total_vram:
                    frac = max(0.05, min(0.95, allowed_vram / total_vram))
                    torch_api.cuda.set_per_process_memory_fraction(frac, device=0)
                    print(f"🧩 Limiting CUDA allocator to ~{frac*100:.0f}% of VRAM ({allowed_vram:.1f}GB)")
            except Exception as e:
                print(f"⚠️  Could not set CUDA memory fraction: {e}")
            model = whisper.load_model("large", device="cuda")
        elif config.get("dml_available", False):
            try:
                import torch_directml  # type: ignore
                dml_device = torch_directml.device()
                chosen_device = dml_device
                device_name = "DirectML GPU"
                print("🎯 Device: DirectML GPU")
                model = whisper.load_model("large", device=dml_device)
            except Exception as e:
                print(f"⚠️  DirectML unavailable, falling back to CPU: {e}")
                model = None
        if model is None:
            chosen_device = "cpu"
            device_name = f"CPU ({multiprocessing.cpu_count()} cores)"
            print(f"🎯 Device: {device_name}")
            model = whisper.load_model("large", device="cpu")
    except Exception as load_e:
        print(f"❌ Model load failed on preferred device: {load_e}")
        print("🔄 Falling back to CPU...")
        chosen_device = "cpu"
        device_name = f"CPU ({multiprocessing.cpu_count()} cores)"
        model = whisper.load_model("large", device="cpu")

    # Apply explicit threads override if provided
    if isinstance(threads_override, int) and threads_override > 0:
        config["cpu_threads"] = max(1, min(64, threads_override))

    # Set CPU threads (& interop)
    torch_api.set_num_threads(config["cpu_threads"])
    try:
        torch_api.set_num_interop_threads(max(2, min(8, config["cpu_threads"] // 4)))
    except Exception:
        pass
    # Also hint MKL/OMP to use similar thread counts
    try:
        os.environ.setdefault("MKL_NUM_THREADS", str(config["cpu_threads"]))
        os.environ.setdefault("OMP_NUM_THREADS", str(config["cpu_threads"]))
    except Exception:
        pass
    print(f"🧵 PyTorch threads set to: {config['cpu_threads']} (interop≈{max(2, min(8, config['cpu_threads'] // 4))})")

    # Choose batch size based on device resources
    def _choose_batch_size(dev: str) -> int:
        if dev == "cuda":
            vram = float(config.get("allowed_vram_gb") or (config.get("cuda_total_vram_gb") or 0.0))
            if vram >= 12:
                return 48
            if vram >= 8:
                return 32
            if vram >= 6:
                return 24
            return 16
        if dev == "cpu":
            return max(4, min(12, config["cpu_threads"] // 2))
        # DirectML or others
        return 16
    batch_size = _choose_batch_size("cuda" if chosen_device == "cuda" else ("cpu" if chosen_device == "cpu" else "dml"))
    # Allow environment override for batch size
    try:
        env_bs = int(os.environ.get("TRANSCRIBE_BATCH_SIZE", "") or 0)
    except Exception:
        env_bs = 0
    if env_bs > 0:
        batch_size = max(1, min(128, env_bs))
    # Explicit parameter override wins last
    if isinstance(batch_size_override, int) and batch_size_override > 0:
        batch_size = max(1, min(128, batch_size_override))
    print(f"📦 Inference batch size: {batch_size}")

    # Run transcription in a watchdog thread
    import threading
    transcription_complete = False
    transcription_result = None
    transcription_error = None

    def _run_transcribe():
        nonlocal transcription_complete, transcription_result, transcription_error
        try:
            print("🔄 Starting Whisper transcription process...")
            if model is None:
                raise RuntimeError("Whisper model is not loaded")
            # Prefer a light VAD filter when supported to avoid music-only hallucinations
            try:
                result = model.transcribe(
                    input_path,
                    language=None,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-2.0,
                    no_speech_threshold=0.3,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    verbose=True,
                    batch_size=batch_size,
                    vad_filter=True,  # available in newer openai-whisper builds
                    vad_parameters={
                        "vad_onset": 0.6,   # be conservative about what counts as speech
                        "vad_offset": 0.4,
                    },
                )
            except TypeError:
                # Older whisper without vad_filter support
                result = model.transcribe(
                    input_path,
                    language=None,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-2.0,
                    no_speech_threshold=0.3,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    verbose=True,
                    batch_size=batch_size,
                )
            transcription_result = result
            print("✅ Whisper transcription completed successfully")
        except Exception as e:
            transcription_error = e
        finally:
            transcription_complete = True

    transcribe_thread = threading.Thread(target=_run_transcribe, daemon=True)
    transcribe_thread.start()

    start_watch = time.time()
    timeout_minutes = 60
    print(f"⏱️  Monitoring transcription progress (timeout: {timeout_minutes} minutes)...")
    while not transcription_complete and (time.time() - start_watch) < (timeout_minutes * 60):
        time.sleep(5)
        if torch_api.cuda.is_available() and chosen_device == "cuda":
            try:
                used = torch_api.cuda.memory_allocated() / (1024 ** 3)
                total = torch_api.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                pct = (used / total) * 100
                elapsed = time.time() - start_watch
                print(f"📊 Progress: {elapsed:.0f}s elapsed | GPU Memory: {used:.1f}/{total:.1f}GB ({pct:.1f}%)")
                if pct > 95:
                    torch_api.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️  GPU monitoring error: {e}")

    # Timeout -> CPU fallback
    if not transcription_complete:
        print(f"⏰ Transcription timed out after {timeout_minutes} minutes; falling back to CPU...")
        if torch_api.cuda.is_available():
            try:
                torch_api.cuda.empty_cache()
                torch_api.cuda.synchronize()
            except Exception:
                pass
        try:
            model = whisper.load_model("large", device="cpu")
            torch_api.set_num_threads(config["cpu_threads"])
            transcription_result = model.transcribe(
                input_path,
                language=None,
                compression_ratio_threshold=2.4,
                logprob_threshold=-2.0,
                no_speech_threshold=0.3,
                condition_on_previous_text=False,
                temperature=0.0,
                verbose=True,
                batch_size=max(4, min(12, config["cpu_threads"] // 2)),
            )
            transcription_error = None
        except Exception as cpu_e:
            raise Exception(f"Both GPU and CPU transcription timed out/failed: {cpu_e}")

    if transcription_error:
        raise Exception(f"Transcription failed: {transcription_error}")

    # Extract text (with artifact suppression around music)
    result = transcription_result
    full_text = ""
    removed_segments = []
    kept_count = 0
    if isinstance(result, dict):
        segments = result.get("segments")
        if isinstance(segments, list) and segments:
            def _is_suspicious_music_artifact(seg_text: str) -> bool:
                t = (seg_text or "").lower()
                if "©" in seg_text or "(c)" in t or "copyright" in t:
                    return True
                markers = [
                    "bf-watch", "watch tv", "bfwatch", "all rights reserved",
                    "www.", "http://", "https://",
                ]
                return any(m in t for m in markers)

            cleaned_parts = []
            for seg in segments:
                seg_text = str(seg.get("text", "")).strip()
                avg_logprob = seg.get("avg_logprob", 0.0)
                no_speech_prob = seg.get("no_speech_prob", 0.0)

                suspicious = _is_suspicious_music_artifact(seg_text)
                low_confidence = (isinstance(avg_logprob, (int, float)) and avg_logprob < -0.5) or \
                                 (isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.6)

                if seg_text and not (suspicious and low_confidence):
                    cleaned_parts.append(seg_text)
                    kept_count += 1
                else:
                    if seg_text:
                        removed_segments.append({
                            "text": seg_text[:120],
                            "avg_logprob": avg_logprob,
                            "no_speech_prob": no_speech_prob,
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                        })
            full_text = (" ".join(cleaned_parts)).strip()
        else:
            text_result = result.get("text", "")
            full_text = text_result.strip() if isinstance(text_result, str) else str(text_result).strip()
    elif isinstance(result, list) and result:
        full_text = str(result[0]).strip()
    else:
        full_text = ""

    if removed_segments:
        print(f"🧽 Artifact filter removed {len(removed_segments)} low-confidence watermark-like segment(s) during music.")
        # Show a brief preview for diagnostics
        sample = removed_segments[0]
        print(f"   e.g., '{sample['text']}' (avg_logprob={sample['avg_logprob']}, no_speech_prob={sample['no_speech_prob']})")

    if not full_text:
        print("⚠️  Warning: No transcription text generated")
        full_text = "[No speech detected or transcription failed]"

    print(f"⚡ Hardware utilised: {device_name}")

    # Post-processing
    try:
        pm = PunctuationModel()
        full_text = pm.restore_punctuation(full_text)
        print("✅ Punctuation restoration completed")
    except Exception as e:
        print(f"⚠️  Punctuation restoration failed: {e}")

    try:
        formatted = split_into_paragraphs(full_text, max_length=500)
        if isinstance(formatted, list):
            formatted_text = "\n\n".join(formatted)
        else:
            formatted_text = full_text
        print("✅ Text formatting completed")
    except Exception as e:
        print(f"⚠️  Text formatting failed: {e}")
        formatted_text = full_text

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    docx_path = os.path.join(output_dir, f"{base_name}.docx")

    # Save TXT
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"✅ Text file saved: {txt_path}")
    except Exception as e:
        print(f"❌ Failed to save text file: {e}")
        txt_path = None

    # Save DOCX with fallback
    try:
        doc = Document()
        doc.add_heading(f'Maximum Performance Auto Transcription: {base_name}', 0)
        if duration:
            doc.add_paragraph(f'Duration: {format_duration(duration)}')
        elapsed_total = time.time() - start_time
        doc.add_paragraph(f'Processing time: {format_duration(elapsed_total)}')
        doc.add_paragraph('Model: Large (Maximum Performance Auto-detected)')
        doc.add_paragraph(f'Hardware: {device_name}')
        doc.add_paragraph('')
        for para in formatted_text.split("\n\n"):
            if para.strip():
                doc.add_paragraph(para.strip())
        doc.save(docx_path)
        print(f"✅ Word document saved: {docx_path}")
    except Exception as e:
        print(f"❌ Failed to create Word document: {e}")
        try:
            doc = Document()
            doc.add_heading(f'Transcription: {base_name}', 0)
            doc.add_paragraph(formatted_text[:5000])
            doc.save(docx_path)
            print(f"✅ Basic Word document saved: {docx_path}")
        except Exception as e2:
            print(f"❌ Failed to save even basic Word document: {e2}")
            docx_path = None

    # Final stats
    elapsed = time.time() - start_time
    print("\n🎉 TRANSCRIPTION COMPLETE!")
    print(f"📄 Text file: {txt_path}")
    print(f"📄 Word document: {docx_path}")
    print(f"⏱️  Total time: {format_duration(elapsed)}")

    # Cleanup caches (do not touch torch modules)
    force_gpu_memory_cleanup()

    # Print memory status after cleanup
    mem = psutil.virtual_memory()
    if torch_api is not None and torch_api.cuda.is_available():
        try:
            gpu_after = torch_api.cuda.memory_allocated() / (1024 ** 3)
            print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available, GPU {gpu_after:.1f}GB used")
        except Exception:
            print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available")
    else:
        print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available")

    return txt_path


def transcribe_file_optimised(input_path, model_name="medium", output_dir=None, force_optimised=True, *, threads_override: Optional[int] = None, batch_size_override: Optional[int] = None):
    """Compatibility wrapper. Uses the simple auto path."""
    return transcribe_file_simple_auto(input_path, output_dir=output_dir, threads_override=threads_override, batch_size_override=batch_size_override)


def main():
    parser = argparse.ArgumentParser(description="Simplified auto-detected transcription")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--output-dir", help="Output directory (default: Downloads)")
    parser.add_argument("--threads", type=int, help="Override CPU threads for PyTorch/OMP/MKL")
    parser.add_argument("--batch-size", type=int, help="Override inference batch size")
    parser.add_argument("--ram-gb", type=float, help="Override usable RAM in GB (default: auto)")
    parser.add_argument("--ram-fraction", type=float, help="Override usable RAM as fraction (0.0-1.0, default: auto)")
    parser.add_argument("--vram-gb", type=float, help="Override usable VRAM in GB (default: auto)")
    parser.add_argument("--vram-fraction", type=float, help="Override usable VRAM as fraction (0.0-1.0, default: auto)")
    parser.add_argument("--ram-gb", type=float, help="Cap usable system RAM in GB for planning (env TRANSCRIBE_RAM_GB)")
    parser.add_argument("--ram-frac", type=float, help="Cap usable system RAM as fraction 0-1 (env TRANSCRIBE_RAM_FRACTION)")
    parser.add_argument("--vram-gb", type=float, help="Cap usable CUDA VRAM in GB for planning (env TRANSCRIBE_VRAM_GB)")
    parser.add_argument("--vram-frac", type=float, help="Cap usable CUDA VRAM as fraction 0-1 (env TRANSCRIBE_VRAM_FRACTION)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Apply env overrides for RAM/VRAM if provided
    try:
        if getattr(args, "ram_gb", None):
            os.environ["TRANSCRIBE_RAM_GB"] = str(max(1.0, float(args.ram_gb)))
        if getattr(args, "ram_frac", None):
            os.environ["TRANSCRIBE_RAM_FRACTION"] = str(max(0.05, min(1.0, float(args.ram_frac))))
        if getattr(args, "vram_gb", None):
            os.environ["TRANSCRIBE_VRAM_GB"] = str(max(0.5, float(args.vram_gb)))
        if getattr(args, "vram_frac", None):
            os.environ["TRANSCRIBE_VRAM_FRACTION"] = str(max(0.05, min(1.0, float(args.vram_frac))))
    except Exception:
        pass

    # Set RAM/VRAM overrides via env if provided
    if getattr(args, "ram_gb", None) is not None:
        os.environ["TRANSCRIBE_RAM_GB"] = str(args.ram_gb)
    if getattr(args, "ram_fraction", None) is not None:
        os.environ["TRANSCRIBE_RAM_FRACTION"] = str(args.ram_fraction)
    if getattr(args, "vram_gb", None) is not None:
        os.environ["TRANSCRIBE_VRAM_GB"] = str(args.vram_gb)
    if getattr(args, "vram_fraction", None) is not None:
        os.environ["TRANSCRIBE_VRAM_FRACTION"] = str(args.vram_fraction)

    try:
        transcribe_file_simple_auto(
            args.input,
            output_dir=args.output_dir,
            threads_override=args.threads,
            batch_size_override=getattr(args, "batch_size", None),
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())