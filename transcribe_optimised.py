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
from typing import Any, cast, Optional, Dict

# IMPORTANT: Import torch once at module import time. Do NOT delete torch.* from sys.modules.
try:
    import torch  # type: ignore
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    _torch_available = True
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = type(None)  # type: ignore
    DataLoader = type(None)  # type: ignore
    np = None  # type: ignore
    _torch_import_error = e
    _torch_available = False


def _ensure_torch_available():
    if torch is None:
        raise RuntimeError(f"PyTorch is required but failed to import: {_torch_import_error}")


class AudioTranscriptionDataset(Dataset):
    """
    PyTorch Dataset for efficient audio transcription with GPU pipeline optimization.

    This dataset enables:
    - Batch processing of audio segments
    - Efficient GPU memory usage
    - Parallel data loading and preprocessing
    - Better utilization of GPU pipelines
    """

    def __init__(self, audio_path: str, segment_length: int = 30, overlap: int = 5):
        """
        Initialize the dataset.

        Args:
            audio_path: Path to the audio file
            segment_length: Length of each audio segment in seconds
            overlap: Overlap between segments in seconds
        """
        self.audio_path = audio_path
        self.segment_length = segment_length
        self.overlap = overlap
        self.segments = []

        # Load audio and create segments
        self._load_and_segment_audio()

    def _load_and_segment_audio(self):
        """Load audio file and create overlapping segments for efficient processing."""
        try:
            import whisper
            from whisper.audio import load_audio

            # Load the audio file
            audio = load_audio(self.audio_path)
            sample_rate = whisper.audio.SAMPLE_RATE
            total_samples = len(audio)

            # Calculate segment parameters
            segment_samples = self.segment_length * sample_rate
            overlap_samples = self.overlap * sample_rate
            step_samples = segment_samples - overlap_samples

            # Create overlapping segments
            start_sample = 0

            while start_sample < total_samples:
                end_sample = min(start_sample + segment_samples, total_samples)
                segment_audio = audio[start_sample:end_sample]

                # Pad short segments if needed
                if len(segment_audio) < segment_samples and np is not None:
                    padding = np.zeros(segment_samples - len(segment_audio))
                    segment_audio = np.concatenate([segment_audio, padding])

                self.segments.append({
                    'audio': segment_audio,
                    'start_time': start_sample / sample_rate,
                    'end_time': end_sample / sample_rate,
                    'segment_id': len(self.segments)
                })

                start_sample += step_samples

            print(f"📊 Created {len(self.segments)} audio segments for efficient GPU processing")

        except Exception as e:
            print(f"⚠️  Failed to create audio segments: {e}")
            # Fallback: treat entire file as single segment
            self.segments = [{
                'audio': [],
                'start_time': 0.0,
                'end_time': 0.0,
                'segment_id': 0
            }]

    def __len__(self) -> int:
        """Return the number of segments."""
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a segment by index."""
        return self.segments[idx]


def create_efficient_dataloader(audio_path: str, batch_size: int = 4, num_workers: int = 2) -> DataLoader:
    """
    Create an efficient DataLoader for audio transcription.

    Args:
        audio_path: Path to the audio file
        batch_size: Number of segments to process in parallel
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader configured for efficient GPU processing
    """
    dataset = AudioTranscriptionDataset(audio_path)

    # Configure DataLoader for GPU efficiency
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Maintain temporal order
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if torch and torch.cuda else False,  # Faster GPU transfer
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
        persistent_workers=num_workers > 0  # Keep workers alive
    )

    return dataloader


def transcribe_with_dataset_optimization(input_path: str, output_dir=None, threads_override: Optional[int] = None):
    """
    Transcribe audio using dataset-based GPU pipeline optimization.

    This function implements the GPU efficiency improvements suggested by PyTorch:
    - Uses PyTorch Dataset for batch processing
    - Leverages DataLoader for optimized data loading
    - Implements overlapping segments for better context
    - Utilizes GPU pipelines for maximum efficiency
    """
    _ensure_torch_available()
    torch_api = cast(Any, torch)

    # Lazy imports
    import whisper
    from docx import Document
    from deepmultilingualpunctuation import PunctuationModel
    from transcribe import (
        get_media_duration, split_into_paragraphs, format_duration,
    )

    start_time = time.time()
    print("🚀 DATASET-OPTIMIZED TRANSCRIPTION WITH GPU PIPELINE EFFICIENCY")
    print(f"📁 Input: {os.path.basename(input_path)}")

    # Check file size to determine if dataset optimization is beneficial
    try:
        file_size = os.path.getsize(input_path)
        if file_size < 50 * 1024 * 1024:  # Less than 50MB
            print("📊 File size < 50MB - falling back to standard processing for optimal performance")
            return transcribe_file_simple_auto(input_path, output_dir, threads_override=threads_override)
    except Exception:
        pass

    # Get hardware config
    max_perf = os.environ.get("TRANSCRIBE_MAX_PERF", "").strip() in ("1", "true", "True")
    config = get_maximum_hardware_config(max_perf=max_perf)

    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    duration = get_media_duration(input_path)
    if duration:
        print(f"⏱️  Duration: {format_duration(duration)}")

    # Pre-run cleanup
    force_gpu_memory_cleanup()

    # Load model
    device_name = "CPU"
    model = None
    chosen_device = "cpu"
    selected_model_name = "large-v3-turbo"

    try:
        avail = set(whisper.available_models())
        turbo_available = ("large-v3-turbo" in avail)
        if not turbo_available:
            for cand in ("large-v3", "large"):
                if cand in avail:
                    selected_model_name = cand
                    break
        print(f"🧩 Whisper turbo available: {turbo_available}")
        print(f"🗂️  Selecting model: {selected_model_name}")
    except Exception as e:
        print(f"⚠️  Could not query whisper.available_models(): {e}")

    # Load model on best available device
    try:
        if "cuda" in config["devices"] and torch_api.cuda.is_available():
            chosen_device = "cuda"
            device_name = f"CUDA GPU ({torch_api.cuda.get_device_name(0)})"
            print("🎯 Device: CUDA GPU with dataset optimization")

            # Enable GPU optimizations
            if hasattr(torch_api.backends, "cudnn"):
                torch_api.backends.cudnn.benchmark = True
            if hasattr(torch_api.backends, "cuda") and hasattr(torch_api.backends.cuda, "matmul"):
                try:
                    torch_api.backends.cuda.matmul.allow_tf32 = True
                except Exception:
                    pass
            try:
                torch_api.set_float32_matmul_precision("high")
            except Exception:
                pass

            model = whisper.load_model(selected_model_name, device="cuda")
        else:
            chosen_device = "cpu"
            device_name = f"CPU ({multiprocessing.cpu_count()} cores)"
            print(f"🎯 Device: {device_name}")
            model = whisper.load_model(selected_model_name, device="cpu")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        raise

    # Set CPU threads
    if isinstance(threads_override, int) and threads_override > 0:
        config["cpu_threads"] = max(1, min(64, threads_override))

    torch_api.set_num_threads(config["cpu_threads"])
    try:
        interop = max(2, min(16, config["cpu_threads"] // 4))
        torch_api.set_num_interop_threads(interop)
    except Exception:
        pass

    print(f"🧵 PyTorch threads set to: {config['cpu_threads']}")

    # Create dataset and dataloader for efficient processing
    try:
        batch_size = 4 if chosen_device == "cuda" else 1
        num_workers = min(2, config["cpu_threads"] // 2) if config["cpu_threads"] > 2 else 0

        dataloader = create_efficient_dataloader(
            input_path,
            batch_size=batch_size,
            num_workers=num_workers
        )

        print(f"📊 Dataset created with {len(dataloader.dataset)} segments, batch_size={batch_size}, workers={num_workers}")

    except Exception as e:
        print(f"⚠️  Dataset creation failed: {e} - falling back to standard processing")
        return transcribe_file_simple_auto(input_path, output_dir, threads_override=threads_override)

    # Process segments with dataset optimization
    all_segments = []
    segment_count = 0

    print("🔄 Processing audio segments with GPU pipeline optimization...")

    for batch in dataloader:
        try:
            for segment_data in batch:
                segment_audio = segment_data['audio'].numpy() if hasattr(segment_data['audio'], 'numpy') else segment_data['audio']
                start_time_seg = segment_data['start_time']
                end_time_seg = segment_data['end_time']

                # Transcribe this segment
                result = model.transcribe(
                    segment_audio,
                    language="en",  # Optimized for English language
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-2.0,
                    no_speech_threshold=0.3,
                    condition_on_previous_text=True,  # Enable context from previous segments
                    temperature=0.0,
                    verbose=False,  # Reduce verbosity for batch processing
                    suppress_tokens="-1",  # Disable token suppression for guardrail removal
                )

                if isinstance(result, dict) and "segments" in result:
                    for seg in result["segments"]:
                        # Adjust timestamps to global timeline
                        seg_copy = dict(seg)
                        seg_copy["start"] = start_time_seg + seg.get("start", 0)
                        seg_copy["end"] = start_time_seg + seg.get("end", 0)
                        all_segments.append(seg_copy)

                segment_count += 1
                if segment_count % 10 == 0:
                    print(f"📊 Processed {segment_count} segments...")

        except Exception as e:
            print(f"⚠️  Batch processing error: {e} - continuing with next batch")

    print(f"✅ Dataset processing complete: {len(all_segments)} total segments")

    # Combine results
    full_text = ""
    if all_segments:
        # Sort segments by start time
        all_segments.sort(key=lambda x: x.get("start", 0))

        # Extract and combine text
        texts = []
        for seg in all_segments:
            text = seg.get("text", "").strip()
            if text:
                texts.append(text)

        full_text = " ".join(texts).strip()

    if not full_text:
        print("⚠️  Warning: No transcription text generated")
        full_text = "[No speech detected or transcription failed]"

    print(f"⚡ Hardware utilised: {device_name} (Dataset Optimized)")

    # Post-processing (same as original)
    try:
        pm = PunctuationModel()
        t0 = time.time()
        full_text = pm.restore_punctuation(full_text)
        t1 = time.time()
        full_text = pm.restore_punctuation(full_text)
        t2 = time.time()
        print(f"✅ Punctuation restoration completed (passes: 2 | {t1 - t0:.1f}s + {t2 - t1:.1f}s)")
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

    # Save files (same as original)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    docx_path = os.path.join(output_dir, f"{base_name}.docx")

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"✅ Text file saved: {txt_path}")
    except Exception as e:
        print(f"❌ Failed to save text file: {e}")
        txt_path = None

    try:
        doc = Document()
        doc.add_heading(f'Transcription: {base_name}', 0)

        elapsed_total = time.time() - start_time
        speed_factor = None
        try:
            if duration and duration > 0:
                speed_factor = max(0.01, float(duration) / float(elapsed_total))
        except Exception:
            speed_factor = None

        if duration:
            doc.add_paragraph(f'Duration: {format_duration(duration)}')
        if speed_factor is not None:
            doc.add_paragraph(f'Speed: {speed_factor:.2f}× realtime')
        if selected_model_name:
            doc.add_paragraph(f'Model: {selected_model_name} (Dataset Optimized)')
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
    print("\n🎉 DATASET-OPTIMIZED TRANSCRIPTION COMPLETE!")
    print(f"📄 Text file: {txt_path}")
    print(f"📄 Word document: {docx_path}")
    print(f"⏱️  Total time: {format_duration(elapsed)}")

    # Cleanup
    force_gpu_memory_cleanup()

    import psutil as _ps
    mem = _ps.virtual_memory()
    if torch_api.cuda.is_available():
        try:
            gpu_after = torch_api.cuda.memory_allocated() / (1024 ** 3)
            print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available, GPU {gpu_after:.1f}GB used")
        except Exception:
            print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available")
    else:
        print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available")

    return txt_path


def get_maximum_hardware_config(max_perf: bool = False):
    """Detect hardware and return a conservative, stable config dict."""
    _ensure_torch_available()
    torch_api = cast(Any, torch)
    cpu_cores = max(multiprocessing.cpu_count(), 1)
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024 ** 3)
    available_ram_gb = vm.available / (1024 ** 3)
    # Default: plan to use 98% of currently available RAM (ULTRA AGGRESSIVE)
    usable_ram_gb = max(available_ram_gb * 0.98, 1.0)
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
        # Absolute cap (do not exceed physical total)
        usable_ram_gb = max(1.0, min(total_ram_gb, env_ram_gb))
    elif 0.05 <= env_ram_frac <= 1.0:
        # Fraction of currently available RAM
        usable_ram_gb = max(1.0, available_ram_gb * env_ram_frac)

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

    # Threads: target ~90% by default; in max_perf mode, use 100% of logical cores
    import math
    if max_perf:
        cpu_threads = max(1, min(64, cpu_cores))  # Use ALL cores in max perf mode
    else:
        cpu_threads = max(1, min(64, math.ceil(cpu_cores * 0.95)))  # Increased from 90% to 95%

    # Environment override for threads
    try:
        env_threads = int(os.environ.get("TRANSCRIBE_THREADS", "") or 0)
    except Exception:
        env_threads = 0
    if env_threads > 0:
        cpu_threads = max(1, min(64, env_threads))

    # For ULTRA aggressive utilization, use more GPU workers for maximum utilization
    if has_cuda:
        try:
            gpu_count = torch_api.cuda.device_count()
            # Use more workers to take advantage of additional GPU shared memory
            gpu_workers = min(gpu_count * 3, 8)  # Increased from 6 to 8 max, 3x GPU count
            print(f"🎯 GPU Workers: {gpu_workers} (ULTRA AGGRESSIVE - utilizing {15}GB+ shared memory)")
        except Exception:
            gpu_workers = 3  # Increased fallback from 2 to 3
    else:
        gpu_workers = 0

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
    "available_ram_gb": available_ram_gb,
        "usable_ram_gb": usable_ram_gb,
        "devices": devices,
        "gpu_workers": gpu_workers,
        "total_workers": max(cpu_threads, gpu_workers),
        "dml_available": dml_available,
        "cuda_total_vram_gb": cuda_total_vram_gb,
        "allowed_vram_gb": allowed_vram_gb,
        "max_perf": bool(max_perf),
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


def identify_speakers_from_segments(segments, silence_threshold=1.5):
    """
    Basic speaker identification using timing analysis.
    Assigns speaker labels (Speaker 1, Speaker 2, etc.) based on pauses between segments.

    Args:
        segments: List of Whisper segments with start/end times and text
        silence_threshold: Minimum silence duration (seconds) to consider a speaker change

    Returns:
        List of segments with speaker labels added
    """
    if not segments or not isinstance(segments, list):
        return segments

    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x.get("start", 0))

    labeled_segments = []
    current_speaker = 1
    last_end_time = 0

    for i, segment in enumerate(sorted_segments):
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()

        # Skip empty segments
        if not text:
            continue

        # Check if there's a significant pause indicating speaker change
        time_gap = start_time - last_end_time

        if time_gap >= silence_threshold and i > 0:
            current_speaker += 1

        # Create labeled segment
        labeled_segment = dict(segment)
        labeled_segment["speaker"] = f"Speaker {current_speaker}"
        labeled_segments.append(labeled_segment)

        last_end_time = end_time

    return labeled_segments


def format_text_with_speakers(segments, include_timestamps=False):
    """
    Format transcription text with speaker labels.

    Args:
        segments: List of segments with speaker labels
        include_timestamps: Whether to include timestamps in output

    Returns:
        Formatted text string with speaker labels
    """
    if not segments:
        return ""

    formatted_parts = []
    current_speaker = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "").strip()

        if not text:
            continue

        # Add speaker label if it changed
        if speaker != current_speaker:
            if include_timestamps and segment.get("start") is not None:
                start_time = segment["start"]
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                formatted_parts.append(f"\n{speaker} {timestamp}: {text}")
            else:
                formatted_parts.append(f"\n{speaker}: {text}")
            current_speaker = speaker
        else:
            # Continue with same speaker
            formatted_parts.append(text)

    return " ".join(formatted_parts).strip()


def transcribe_file_simple_auto(input_path, output_dir=None, *, threads_override: Optional[int] = None):
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
    try:
        from text_processor_enhanced import create_enhanced_processor
        _enhanced_processor_available = True
    except ImportError:
        _enhanced_processor_available = False
        create_enhanced_processor = None  # Define it to avoid unbound variable error
    from transcribe import (
        get_media_duration, split_into_paragraphs, format_duration,
    )

    # Speaker identification imports
    try:
        import webrtcvad
        _vad_available = True
    except ImportError:
        _vad_available = False

    start_time = time.time()
    print("🚀 MAXIMUM PERFORMANCE AUTO-DETECTED TRANSCRIPTION")
    print(f"📁 Input: {os.path.basename(input_path)}")

    # Check if speaker identification should be enabled
    enable_speakers = False
    try:
        # Speaker identification is disabled - comment out to re-enable if needed
        # enable_speakers = os.environ.get("TRANSCRIBE_SPEAKER_ID", "").strip() in ("1", "true", "True")
        if enable_speakers:
            print("� Speaker identification enabled")
    except Exception:
        enable_speakers = False

    # Check if dataset optimization should be used
    use_dataset = False
    try:
        use_dataset = os.environ.get("TRANSCRIBE_USE_DATASET", "").strip() in ("1", "true", "True")
        if use_dataset:
            print("🎯 Dataset optimization enabled for GPU pipeline efficiency")
    except Exception:
        use_dataset = False

    # Check if aggressive segmentation should be used
    use_aggressive_segmentation = False
    try:
        use_aggressive_segmentation = os.environ.get("TRANSCRIBE_AGGRESSIVE_SEGMENTATION", "").strip() in ("1", "true", "True")
        if use_aggressive_segmentation:
            print("🎯 Aggressive segmentation enabled for maximum CPU utilization")
    except Exception:
        use_aggressive_segmentation = False
    if use_dataset:
        try:
            file_size = os.path.getsize(input_path)
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                print("📊 Large file detected - using dataset optimization")
                return transcribe_with_dataset_optimization(input_path, output_dir, threads_override)
            else:
                print("📊 File size < 50MB - using standard processing")
        except Exception as e:
            print(f"⚠️  Dataset check failed: {e} - using standard processing")

    # Decide max performance mode from env - DEFAULT TO MAX PERF FOR BETTER CPU UTILIZATION
    max_perf = True  # Default to maximum performance for better CPU utilization
    try:
        env_max_perf = os.environ.get("TRANSCRIBE_MAX_PERF", "").strip()
        if env_max_perf in ("0", "false", "False"):
            max_perf = False
    except Exception:
        max_perf = True
    config = get_maximum_hardware_config(max_perf=max_perf)
    # Report planning
    try:
        if config.get('max_perf'):
            print(f"🧠 Planning: ULTRA MAX PERF -> CPU threads {config['cpu_threads']} of {config['cpu_cores']} (ULTRA AGGRESSIVE)")
        else:
            print(f"🧠 Planning: CPU threads ≈95% cores -> {config['cpu_threads']} of {config['cpu_cores']} (OPTIMIZED)")
        print(f"💾 RAM plan: using up to ~{config['usable_ram_gb']:.1f} GB (98% of {config['available_ram_gb']:.1f} GB available - ULTRA AGGRESSIVE)")
        if float(config.get('allowed_vram_gb') or 0) > 0:
            print(f"🎛️ VRAM cap: ~{float(config['allowed_vram_gb']):.1f} GB of total {float(config.get('cuda_total_vram_gb') or 0):.1f} GB (99% utilization)")
            print(f"🔗 GPU Shared Memory: Utilizing additional 15GB+ for parallel processing")
    except Exception:
        pass
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
    selected_model_name = "large-v3-turbo"  # default preference

    # Prefer turbo; if it's not listed as available, fall back to the next best
    try:
        import whisper  # type: ignore
        avail = set(whisper.available_models())
        turbo_available = ("large-v3-turbo" in avail)
        if not turbo_available:
            for cand in ("large-v3", "large"):
                if cand in avail:
                    selected_model_name = cand
                    break
        print(f"🧩 Whisper turbo available: {turbo_available}")
        print(f"🗂️  Selecting model: {selected_model_name}")
    except Exception as e:
        print(f"⚠️  Could not query whisper.available_models(): {e}. Proceeding with '{selected_model_name}'.")

    try:
        # Elevate process priority on Windows for max perf
        if config.get('max_perf'):
            try:
                import psutil
                p = psutil.Process(os.getpid())
                if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                    print("🚀 Process priority set to HIGH")
            except Exception as e:
                print(f"⚠️  Could not raise process priority: {e}")

        if "cuda" in config["devices"] and torch_api.cuda.is_available():
            chosen_device = "cuda"
            device_name = f"CUDA GPU ({torch_api.cuda.get_device_name(0)})"
            print("🎯 Device: CUDA GPU")
            # Apply CUDA per-process memory fraction if an allowed VRAM cap is set
            try:
                total_vram = float(config.get("cuda_total_vram_gb") or 0.0)
                allowed_vram = float(config.get("allowed_vram_gb") or 0.0)
                if total_vram > 0:
                    if 0.5 <= allowed_vram < total_vram:
                        frac = max(0.05, min(0.95, allowed_vram / total_vram))
                        torch_api.cuda.set_per_process_memory_fraction(frac, device=0)
                        print(f"🧩 Limiting CUDA allocator to ~{frac*100:.0f}% of VRAM ({allowed_vram:.1f}GB)")
                    elif config.get('max_perf'):
                        # Default to ULTRA aggressive allocator in max perf mode - USE MAX VRAM
                        try:
                            torch_api.cuda.set_per_process_memory_fraction(0.99, device=0)  # Increased from 0.98 to 0.99
                            print("🧩 Allowing CUDA allocator to use ~99% of VRAM (ULTRA AGGRESSIVE - MAX PERF)")
                        except Exception:
                            pass
            except Exception as e:
                print(f"⚠️  Could not set CUDA memory fraction: {e}")
            # Enable ULTRA performance knobs for maximum GPU utilization
            try:
                if hasattr(torch_api.backends, "cudnn"):
                    torch_api.backends.cudnn.benchmark = True
                    # Enable deterministic mode for reproducibility but max performance
                    torch_api.backends.cudnn.deterministic = False
                # TF32 can speed up matmul on Ampere+; harmless elsewhere
                if hasattr(torch_api.backends, "cuda") and hasattr(torch_api.backends.cuda, "matmul"):
                    try:
                        torch_api.backends.cuda.matmul.allow_tf32 = True
                        # Enable cuBLAS optimizations
                        torch_api.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                    except Exception:
                        pass
                try:
                    torch_api.set_float32_matmul_precision("high")
                except Exception:
                    pass
                # Enable GPU memory pooling for better shared memory utilization
                try:
                    torch_api.cuda.set_per_process_memory_fraction(0.99, device=0)
                    print("🧩 ULTRA GPU Memory Pooling: Enabled for 15GB+ shared memory utilization")
                except Exception:
                    pass
            except Exception:
                pass
            model = whisper.load_model(selected_model_name, device="cuda")
        elif config.get("dml_available", False):
            try:
                import torch_directml  # type: ignore
                dml_device = torch_directml.device()
                chosen_device = dml_device
                device_name = "DirectML GPU"
                print("🎯 Device: DirectML GPU")
                model = whisper.load_model(selected_model_name, device=dml_device)
            except Exception as e:
                print(f"⚠️  DirectML unavailable, falling back to CPU: {e}")
                model = None
        if model is None:
            chosen_device = "cpu"
            device_name = f"CPU ({multiprocessing.cpu_count()} cores)"
            print(f"🎯 Device: {device_name}")
            model = whisper.load_model(selected_model_name, device="cpu")
    except Exception as load_e:
        print(f"❌ Model load failed on preferred device: {load_e}")
        print("🔄 Falling back to CPU...")
        chosen_device = "cpu"
        device_name = f"CPU ({multiprocessing.cpu_count()} cores)"
        try:
            import whisper  # type: ignore
            model = whisper.load_model(selected_model_name, device="cpu")
        except Exception:
            # Last resort
            model = whisper.load_model("large", device="cpu")

    # Apply explicit threads override if provided
    if isinstance(threads_override, int) and threads_override > 0:
        config["cpu_threads"] = max(1, min(64, threads_override))

    # Set CPU threads (& interop)
    torch_api.set_num_threads(config["cpu_threads"])
    try:
        interop = max(2, min(16, (config["cpu_threads"] // 2) if config.get('max_perf') else (config["cpu_threads"] // 4)))
        torch_api.set_num_interop_threads(interop)
    except Exception:
        pass
    # Also hint MKL/OMP to use similar thread counts
    try:
        os.environ.setdefault("MKL_NUM_THREADS", str(config["cpu_threads"]))
        os.environ.setdefault("OMP_NUM_THREADS", str(config["cpu_threads"]))
    except Exception:
        pass
    print(f"🧵 PyTorch threads set to: {config['cpu_threads']} (interop≈{max(2, min(8, config['cpu_threads'] // 4))})")

    # Note: batch size is not passed to Whisper to ensure broad compatibility across versions

    # Optional: NVML for GPU utilisation logging
    nvml = None
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        nvml = pynvml
    except Exception:
        nvml = None

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
            # Call transcribe without batch_size or vad_filter for broad compatibility
            result = model.transcribe(
                input_path,
                language="en",  # Optimized for English language
                compression_ratio_threshold=2.4,
                logprob_threshold=-2.0,
                no_speech_threshold=0.3,
                condition_on_previous_text=False,
                temperature=0.0,
                verbose=True,
                suppress_tokens="-1",  # Disable token suppression for guardrail removal
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

    # Initialize CPU/RAM monitoring
    import psutil
    process = psutil.Process(os.getpid())

    while not transcription_complete and (time.time() - start_watch) < (timeout_minutes * 60):
        time.sleep(5)
        if torch_api.cuda.is_available() and chosen_device == "cuda":
            try:
                used = torch_api.cuda.memory_allocated() / (1024 ** 3)
                total = torch_api.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                pct = (used / total) * 100
                elapsed = time.time() - start_watch

                # Get CPU and RAM usage
                cpu_percent = process.cpu_percent(interval=None)
                ram_used = process.memory_info().rss / (1024 ** 3)  # GB
                ram_total = psutil.virtual_memory().total / (1024 ** 3)  # GB
                ram_percent = (ram_used / ram_total) * 100

                # Format time: use h:m:s after 800 seconds
                if elapsed >= 800:
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = f"{elapsed:.0f}s"

                util_txt = ""
                if nvml is not None:
                    try:
                        h = nvml.nvmlDeviceGetHandleByIndex(0)
                        util = nvml.nvmlDeviceGetUtilizationRates(h)
                        util_txt = f" | GPU Util: {util.gpu}% | Mem Util: {util.memory}%"
                    except Exception:
                        util_txt = ""

                print(f"📊 Progress: {time_str} | GPU Mem: {used:.1f}/{total:.1f}GB ({pct:.1f}%) | CPU: {cpu_percent:.1f}% | RAM: {ram_used:.1f}/{ram_total:.1f}GB ({ram_percent:.1f}%){util_txt}")

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
            model = whisper.load_model(selected_model_name, device="cpu")
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
            )
            transcription_error = None
        except Exception as cpu_e:
            raise Exception(f"Both GPU and CPU transcription timed out/failed: {cpu_e}")

    if transcription_error:
        raise Exception(f"Transcription failed: {transcription_error}")

    # Extract text (with artifact suppression around music)
    result = transcription_result
    full_text = ""
    segments_with_speakers = []
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
            cleaned_segments = []
            for seg in segments:
                seg_text = str(seg.get("text", "")).strip()
                avg_logprob = seg.get("avg_logprob", 0.0)
                no_speech_prob = seg.get("no_speech_prob", 0.0)

                suspicious = _is_suspicious_music_artifact(seg_text)
                low_confidence = (isinstance(avg_logprob, (int, float)) and avg_logprob < -0.5) or \
                                 (isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.6)

                if seg_text and not (suspicious and low_confidence):
                    cleaned_parts.append(seg_text)
                    cleaned_segments.append(seg)
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

            # Apply speaker identification to cleaned segments
            try:
                segments_with_speakers = identify_speakers_from_segments(cleaned_segments)
                speaker_text = format_text_with_speakers(segments_with_speakers)
                if speaker_text:
                    full_text = speaker_text
                    print("✅ Speaker identification completed")
                else:
                    print("⚠️  Speaker identification failed, using standard text")
            except Exception as e:
                print(f"⚠️  Speaker identification failed: {e}")
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

    # Post-processing with enhanced text processing
    try:
        if _enhanced_processor_available and create_enhanced_processor is not None:
            # Use enhanced processor with spaCy and custom rules
            processor = create_enhanced_processor(use_spacy=True, use_transformers=False)
            t0 = time.time()
            full_text = processor.restore_punctuation(full_text)
            t1 = time.time()
            print(f"✅ Enhanced punctuation restoration completed ({t1 - t0:.1f}s)")
        else:
            # Fallback to basic punctuation model
            from deepmultilingualpunctuation import PunctuationModel
            pm = PunctuationModel()
            t0 = time.time()
            full_text = pm.restore_punctuation(full_text)
            t1 = time.time()
            # Second pass for improved sentence boundaries
            full_text = pm.restore_punctuation(full_text)
            t2 = time.time()
            print(f"✅ Basic punctuation restoration completed (passes: 2 | {t1 - t0:.1f}s + {t2 - t1:.1f}s)")
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
        doc.add_heading(f'Transcription: {base_name}', 0)
        # Badge: duration, speed, language, model only
        elapsed_total = time.time() - start_time
        # Compute realtime speed factor if duration available
        speed_factor = None
        try:
            if duration and duration > 0:
                speed_factor = max(0.01, float(duration) / float(elapsed_total))
        except Exception:
            speed_factor = None
        # Language from whisper result (ISO code)
        detected_lang = None
        if isinstance(transcription_result, dict):
            detected_lang = transcription_result.get("language")
        # Write compact metadata badge
        if duration:
            doc.add_paragraph(f'Duration: {format_duration(duration)}')
        if speed_factor is not None:
            doc.add_paragraph(f'Speed: {speed_factor:.2f}× realtime')
        if detected_lang:
            doc.add_paragraph(f'Language: {detected_lang}')
        if selected_model_name:
            doc.add_paragraph(f'Model: {selected_model_name}')
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
    import psutil as _ps
    mem = _ps.virtual_memory()
    if torch_api is not None and torch_api.cuda.is_available():
        try:
            gpu_after = torch_api.cuda.memory_allocated() / (1024 ** 3)
            print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available, GPU {gpu_after:.1f}GB used")
        except Exception:
            print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available")
    else:
        print(f"📊 Memory after cleanup: RAM {mem.available / (1024**3):.1f}GB available")

    return txt_path


def transcribe_file_optimised(input_path, model_name="medium", output_dir=None, force_optimised=True, *, threads_override: Optional[int] = None):
    """Compatibility wrapper. Uses the simple auto path."""
    return transcribe_file_simple_auto(input_path, output_dir=output_dir, threads_override=threads_override)


def main():
    parser = argparse.ArgumentParser(description="Simplified auto-detected transcription")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--output-dir", help="Output directory (default: Downloads)")
    parser.add_argument("--threads", type=int, help="Override CPU threads for PyTorch/OMP/MKL")
    parser.add_argument("--ram-gb", type=float, help="Cap usable system RAM in GB (env TRANSCRIBE_RAM_GB)")
    parser.add_argument("--ram-frac", "--ram-fraction", dest="ram_fraction", type=float, help="Cap usable system RAM as fraction 0-1 (env TRANSCRIBE_RAM_FRACTION)")
    parser.add_argument("--vram-gb", type=float, help="Cap usable CUDA VRAM in GB (env TRANSCRIBE_VRAM_GB)")
    parser.add_argument("--vram-frac", "--vram-fraction", dest="vram_fraction", type=float, help="Cap usable CUDA VRAM as fraction 0-1 (env TRANSCRIBE_VRAM_FRACTION)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Apply env overrides for RAM/VRAM if provided
    try:
        if getattr(args, "ram_gb", None) is not None:
            os.environ["TRANSCRIBE_RAM_GB"] = str(max(1.0, float(args.ram_gb)))
        if getattr(args, "ram_fraction", None) is not None:
            os.environ["TRANSCRIBE_RAM_FRACTION"] = str(max(0.05, min(1.0, float(args.ram_fraction))))
        if getattr(args, "vram_gb", None) is not None:
            os.environ["TRANSCRIBE_VRAM_GB"] = str(max(0.5, float(args.vram_gb)))
        if getattr(args, "vram_fraction", None) is not None:
            os.environ["TRANSCRIBE_VRAM_FRACTION"] = str(max(0.05, min(1.0, float(args.vram_fraction))))
    except Exception:
        pass

    try:
        transcribe_file_simple_auto(
            args.input,
            output_dir=args.output_dir,
            threads_override=args.threads,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())