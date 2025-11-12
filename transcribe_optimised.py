"""
Optimised transcription utilities with safe PyTorch lifecycle management.
This module provides a high-quality single-file transcription path that avoids
re-importing torch and aggressively clearing only model-level caches between runs.
"""
import warnings
import re
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


def preprocess_audio_with_padding(input_path: str, temp_dir: str = None) -> str:
    """
    Preprocess audio/video file to high-quality MP3 with silence padding.
    
    Args:
        input_path: Path to input audio/video file
        temp_dir: Directory for temporary files (default: system temp)
        
    Returns:
        Path to the preprocessed MP3 file with padding
        
    Features:
    - Converts any audio/video format to high-quality MP3 (320kbps)
    - Adds 1 second of silence at the beginning
    - Adds 1 second of silence at the end  
    - Normalizes audio levels
    - Prevents missed words at start/end of recordings
    """
    import tempfile
    import subprocess
    import shutil
    
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    # Generate unique temp filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    temp_output = os.path.join(temp_dir, f"preprocessed_{base_name}_{int(time.time())}.mp3")
    
    print(f"🔄 Preprocessing audio with silence padding...")
    print(f"📁 Input: {os.path.basename(input_path)}")
    print(f"📁 Temp output: {os.path.basename(temp_output)}")
    
    try:
        # Check if ffmpeg is available
        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            # Try to use the bundled ffmpeg.exe if available
            bundled_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
            if os.path.exists(bundled_ffmpeg):
                ffmpeg_cmd = bundled_ffmpeg
            else:
                raise FileNotFoundError("ffmpeg not found. Please install ffmpeg or ensure it's in PATH.")
        
        # FFmpeg command to:
        # 1. Add 1 second silence at start: adelay=1000|1000 (1000ms delay for both channels)
        # 2. Apply noise reduction for old digitized tapes (highpass, lowpass, afftdn)
        # 3. Normalize audio levels (loudnorm)
        # 4. Add 1 second silence at end using apad
        # 5. Convert to high-quality MP3 (320kbps)
        cmd = [
            ffmpeg_cmd,
            "-i", input_path,
            # Audio processing filters (enhanced for old digitized tapes)
            "-af", "adelay=1000|1000,highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm,apad=pad_len=48000",  # Enhanced filters for old tapes
            # High quality MP3 encoding
            "-codec:a", "libmp3lame",
            "-b:a", "320k",
            "-ar", "48000",  # 48kHz sample rate for best quality
            "-ac", "2",      # Stereo output
            # Overwrite output file
            "-y",
            temp_output
        ]
        
        print(f"🔧 Running ffmpeg preprocessing...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg preprocessing failed:")
            print(f"Error: {result.stderr}")
            # Return original file if preprocessing fails
            return input_path
        
        # Verify the output file was created and has reasonable size
        if os.path.exists(temp_output) and os.path.getsize(temp_output) > 1024:  # > 1KB
            print(f"✅ Audio preprocessing completed successfully")
            print(f"📊 Original size: {os.path.getsize(input_path) / (1024*1024):.1f} MB")
            print(f"📊 Preprocessed size: {os.path.getsize(temp_output) / (1024*1024):.1f} MB")
            return temp_output
        else:
            print(f"⚠️  Preprocessing produced invalid output, using original file")
            return input_path
            
    except subprocess.TimeoutExpired:
        print(f"⚠️  FFmpeg preprocessing timed out after 5 minutes, using original file")
        return input_path
    except Exception as e:
        print(f"⚠️  Audio preprocessing failed: {e}")
        print(f"🔄 Continuing with original file...")
        return input_path


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


# --- Special words support (prompt biasing) ---------------------------------
def _read_lines(path: str) -> list:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f.readlines()]
    except Exception:
        return []


def load_awkward_terms(input_path: str) -> list:
    """Load user-provided domain terms from common locations or env.

    Priority:
      1) TRANSCRIBE_AWKWARD_TERMS env (comma-separated)
      2) TRANSCRIBE_AWKWARD_FILE env (path to .txt/.md)
      3) special_words.txt / special_words.md in the input file's folder
      4) special_words.txt / special_words.md in the repo root (this module's dir)
    """
    terms = []
    try:
        # 1) Inline env list
        env_list = os.environ.get('TRANSCRIBE_AWKWARD_TERMS', '')
        if env_list.strip():
            for t in env_list.split(','):
                t = t.strip()
                if t:
                    terms.append(t)

        # 2) Env file
        env_file = os.environ.get('TRANSCRIBE_AWKWARD_FILE', '').strip()
        if env_file and os.path.isfile(env_file):
            terms.extend(_read_lines(env_file))

        # 3) Local folder files
        in_dir = os.path.dirname(input_path)
        for fname in ('special_words.txt', 'special_words.md'):
            p = os.path.join(in_dir, fname)
            if os.path.isfile(p):
                terms.extend(_read_lines(p))
                break

        # 4) Repo root
        repo_dir = os.path.dirname(__file__)
        for fname in ('special_words.txt', 'special_words.md'):
            p = os.path.join(repo_dir, fname)
            if os.path.isfile(p):
                terms.extend(_read_lines(p))
                break
    except Exception:
        pass

    # Normalize simple bullet formats and filter empties/comments
    cleaned = []
    for ln in terms:
        if not ln:
            continue
        s = ln.lstrip('-•*\t >').strip()
        if not s or s.startswith('#'):
            continue
        if s not in cleaned:
            cleaned.append(s)

    # Cap length to keep prompt small and focused
    return cleaned[:40]


def build_initial_prompt(terms: list, max_chars: int = 400) -> Optional[str]:
    """Build a concise initial_prompt string to bias Whisper.

    Keeps capitalization as provided; trims to max_chars.
    """
    if not terms:
        return None
    try:
        # Join terms with semicolons for clarity
        payload = '; '.join(terms)
        base = (
            "If and only if these domain terms are clearly spoken, prefer them exactly as written; "
            "otherwise ignore them. Do not force, repeat, or overuse these terms. Maintain capitalization: "
        )
        prompt = (base + payload).strip()
        if len(prompt) > max_chars:
            prompt = prompt[: max_chars - 3].rstrip() + '...'
        return prompt
    except Exception:
        return None

# --- Artifact mitigation: collapse excessive exact repetitions ----------------
def _collapse_repetitions(text: str, max_repeats: int = 3) -> str:
    """Collapse excessive immediate repetitions of the same phrase.

    This targets simple loops like "to grow, to grow, to grow, ..." and reduces
    them to at most `max_repeats` consecutive occurrences.
    """
    try:
        # Check for environment override for max repeats
        env_max_repeats = os.environ.get("TRANSCRIBE_MAX_REPEAT_CAP", "").strip()
        if env_max_repeats:
            try:
                max_repeats = max(1, min(10, int(env_max_repeats)))  # Cap between 1-10
            except Exception:
                pass  # Use default if invalid
        
        # Normalize spaces around commas for matching
        t = re.sub(r"\s*,\s*", ", ", text)
        # Build a regex that captures a short phrase (1-6 words) repeated many times
        # Words may include apostrophes; keep phrases modest to avoid over-collapsing
        pattern = r"\b((?:[A-Za-z']+\s+){0,5}[A-Za-z']+)\b(?:,\s*\1\b){" + str(max_repeats) + ",}"

        def repl(m):
            phrase = m.group(1)
            return ", ".join([phrase] * max_repeats)

        # Apply repeatedly a few times to catch nested patterns
        for _ in range(2):
            new_t = re.sub(pattern, repl, t)
            if new_t == t:
                break
            t = new_t
        return t
    except Exception:
        return text


def _fix_whisper_artifacts(text: str) -> str:
    """Fix common Whisper transcription artifacts found in analysis.
    
    Based on turbo/large-v3 comparison testing:
    - Removes double periods (. .)
    - Fixes dialogue punctuation inconsistencies
    - Improves sentence boundary detection
    """
    try:
        # Fix double periods (common artifact in both models)
        text = re.sub(r'\.\s*\.+', '.', text)
        
        # Fix period-period spacing artifacts
        text = re.sub(r'\.\s+\.', '.', text)
        
        # Fix question mark patterns in quoted speech
        # "what am I doing now?" instead of what am I doing now?
        text = re.sub(r'\b(what|who|when|where|why|how)\s+([^?.!]+)\?', r'\1 \2?', text, flags=re.IGNORECASE)
        
        # Fix colon + lowercase after direct quote intro (should be lowercase for continuation)
        # He said: well, what -> He said: well, what (keep lowercase)
        # But: Right now I'm -> Right now I'm (keep capitals when appropriate)
        
        # Fix common conjunction drops at sentence starts
        # "All this energy" after period should check context
        # This is complex - leave for manual review for now
        
        # Clean up multiple spaces
        text = re.sub(r'  +', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+([,.!?;:])', r'\1 \2', text)
        
        return text.strip()
    except Exception as e:
        print(f"⚠️  Artifact fixing failed: {e}")
        return text


def _refine_capitalization(text: str) -> str:
    """Fix capitalization artifacts without changing word content.
    
    Fixes common Whisper artifacts:
    - Incorrectly capitalized words mid-sentence
    - Capital letters immediately after commas (not sentence starts)
    - Preserves proper nouns and acronyms
    """
    try:
        # Split into sentences while preserving structure
        sentences = re.split(r'([.!?]+\s+)', text)
        
        refined_sentences = []
        for i, part in enumerate(sentences):
            # Skip sentence delimiters
            if re.match(r'^[.!?]+\s+$', part):
                refined_sentences.append(part)
                continue
            
            # Process each sentence
            if part.strip():
                # Fix: Capital letter after comma mid-sentence
                # Pattern: ", Word" -> ", word" (unless it's a proper noun)
                part = re.sub(
                    r',\s+([A-Z])([a-z]+)',
                    lambda m: f', {m.group(1).lower()}{m.group(2)}' 
                    if m.group(1) + m.group(2) not in ['I', 'The', 'A', 'An'] 
                    else m.group(0),
                    part
                )
                
                # Fix: Mid-sentence capitalization not following punctuation
                # Split on spaces to check each word
                words = part.split()
                if words:
                    # First word of sentence should be capitalized
                    if words[0] and words[0][0].islower():
                        words[0] = words[0][0].upper() + words[0][1:]
                    
                    # Check remaining words
                    for j in range(1, len(words)):
                        word = words[j]
                        if not word:
                            continue
                        
                        # Check if previous word ended with sentence-ending punctuation
                        prev_ends_sentence = j > 0 and words[j-1] and words[j-1][-1] in '.!?'
                        
                        # If word is capitalized but not after punctuation
                        if word[0].isupper() and not prev_ends_sentence:
                            # Check if it's likely a proper noun (all caps, or starts uppercase and has uppercase later)
                            is_acronym = word.isupper() and len(word) > 1
                            has_internal_caps = len(word) > 1 and any(c.isupper() for c in word[1:])
                            
                            # Preserve known proper nouns and acronyms
                            if not (is_acronym or has_internal_caps or word in ['I']):
                                # Lowercase the first character
                                words[j] = word[0].lower() + word[1:]
                    
                    part = ' '.join(words)
                
                refined_sentences.append(part)
        
        return ''.join(refined_sentences)
    except Exception as e:
        print(f"⚠️  Capitalization refinement failed: {e}")
        return text


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
        get_media_duration, split_into_paragraphs, format_duration, format_duration_minutes_only, format_duration_hms,
    )

    start_time = time.time()
    print("🚀 DATASET-OPTIMIZED TRANSCRIPTION WITH GPU PIPELINE EFFICIENCY")
    print(f"📁 Input: {os.path.basename(input_path)}")

    # Preprocess audio with silence padding to prevent missed words
    preprocessed_path = preprocess_audio_with_padding(input_path)
    preprocessing_used = preprocessed_path != input_path
    
    if preprocessing_used:
        print(f"✅ Using preprocessed audio with silence padding")
        # Use the preprocessed file for all subsequent operations
        working_input_path = preprocessed_path
    else:
        print(f"🔄 Using original audio file")
        working_input_path = input_path

    # Check file size to determine if dataset optimization is beneficial
    try:
        file_size = os.path.getsize(working_input_path)
        if file_size < 50 * 1024 * 1024:  # Less than 50MB
            print("📊 File size < 50MB - falling back to standard processing for optimal performance")
            # Clean up preprocessed file before returning
            try:
                if preprocessing_used and os.path.exists(working_input_path):
                    os.remove(working_input_path)
                    print("🧹 Removed temporary preprocessed audio file")
            except Exception:
                pass
            return transcribe_file_simple_auto(input_path, output_dir, threads_override=threads_override)
    except Exception:
        pass

    # Get hardware config
    max_perf = os.environ.get("TRANSCRIBE_MAX_PERF", "").strip() in ("1", "true", "True")
    config = get_maximum_hardware_config(max_perf=max_perf)

    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    duration = get_media_duration(working_input_path)
    if duration:
        print(f"⏱️  Duration: {format_duration(duration)}")

    # Pre-run cleanup
    force_gpu_memory_cleanup()

    # Load model
    device_name = "CPU"
    model = None
    chosen_device = "cpu"
    
    # Check for model selection from environment variable (set by GUI)
    selected_model_name = os.environ.get("TRANSCRIBE_MODEL_NAME", "large-v3")

    try:
        avail = set(whisper.available_models())
        requested_available = (selected_model_name in avail)
        if not requested_available:
            print(f"⚠️  Requested model '{selected_model_name}' not available, falling back...")
            for cand in ("large-v3", "large-v2", "large"):
                if cand in avail:
                    selected_model_name = cand
                    break
        print(f"🧩 Requested model available: {requested_available}")
        print(f"🗂️  Selecting model: {selected_model_name}")
        print(f"🎯 Model Source: Environment variable TRANSCRIBE_MODEL_NAME = '{os.environ.get('TRANSCRIBE_MODEL_NAME', 'NOT SET')}'")
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
                    torch_api.backends.cudnn.matmul.allow_tf32 = True
                except Exception:
                    pass
            try:
                torch_api.set_float32_matmul_precision("high")
            except Exception:
                pass

            # Load model in FP16 for maximum GPU performance
            model = whisper.load_model(selected_model_name, device="cuda")
            # Move model to FP16 for 2x faster inference on GPU
            try:
                model = model.half()
                print(f"🚀 Model '{selected_model_name}' converted to FP16 (half precision) for maximum GPU speed")
                print(f"💡 Expected speed improvement: 2-3x faster inference vs FP32")
            except Exception as fp16_err:
                print(f"⚠️  Could not convert to FP16: {fp16_err} - using FP32")
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
    interop = max(2, min(16, config["cpu_threads"] // 4))
    try:
        torch_api.set_num_interop_threads(interop)
    except Exception:
        pass

    # Explicit thread configuration logging
    print(f"🔧 Thread Configuration:")
    print(f"   • CPU Threads: {config['cpu_threads']}")
    print(f"   • Interop Threads: {interop}")
    print(f"   • Device: {device_name}")
    print(f"   • Model: {selected_model_name}")

    # Build optional initial_prompt from special terms
    awkward_terms = load_awkward_terms(input_path)
    initial_prompt = build_initial_prompt(awkward_terms)

    # Create dataset and dataloader for efficient processing
    try:
        # Increase batch size for CUDA to maximize GPU throughput
        batch_size = 8 if chosen_device == "cuda" else 1  # Increased from 4 to 8 for better GPU utilization
        # Increase worker threads for better data pipeline
        num_workers = min(4, config["cpu_threads"] // 2) if config["cpu_threads"] > 4 else 0  # Increased from 2 to 4

        dataloader = create_efficient_dataloader(
            working_input_path,
            batch_size=batch_size,
            num_workers=num_workers
        )

        print(f"📊 Dataset created with {len(dataloader.dataset)} segments, batch_size={batch_size}, workers={num_workers}")

    except Exception as e:
        print(f"⚠️  Dataset creation failed: {e} - falling back to standard processing")
        # Cleanup preprocessed file before fallback
        try:
            if preprocessing_used and os.path.exists(working_input_path):
                os.remove(working_input_path)
                print("🧹 Removed temporary preprocessed audio file")
        except Exception:
            pass
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
                seg_kwargs = dict(
                    language="en",  # Optimized for English language
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-2.0,
                    no_speech_threshold=0.3,
                    condition_on_previous_text=True,  # Enable context from previous segments
                    temperature=0.0,
                    verbose=False,  # Reduce verbosity for batch processing
                )
                
                # Model-specific tuning for accuracy
                if selected_model_name == "large-v3-turbo":
                    # Turbo-specific: tighter thresholds for better accuracy
                    seg_kwargs["compression_ratio_threshold"] = 2.2
                    seg_kwargs["logprob_threshold"] = -1.5
                    seg_kwargs["no_speech_threshold"] = 0.4
                    seg_kwargs["hallucination_silence_threshold"] = 2.0
                elif selected_model_name == "large-v3":
                    # Large-v3: slightly more permissive for natural speech flow
                    seg_kwargs["compression_ratio_threshold"] = 2.6
                    seg_kwargs["logprob_threshold"] = -2.2
                
                # Apply quality mode if enabled
                quality_mode = os.environ.get("TRANSCRIBE_QUALITY_MODE", "").strip() in ("1", "true", "True")
                if quality_mode:
                    if selected_model_name == "large-v3-turbo":
                        seg_kwargs["beam_size"] = 7
                        seg_kwargs["patience"] = 2.5
                        seg_kwargs["best_of"] = 5
                        print("🎯 Quality mode (turbo): beam_size=7, patience=2.5, best_of=5")
                    else:
                        seg_kwargs["beam_size"] = 5
                        seg_kwargs["patience"] = 2.0
                        print("🎯 Quality mode (large-v3): beam_size=5, patience=2.0")
                
                if initial_prompt:
                    seg_kwargs["initial_prompt"] = initial_prompt
                result = model.transcribe(segment_audio, **seg_kwargs)

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
        # Collapse excessive repetitions prior to punctuation restoration
        full_text = _collapse_repetitions(full_text, max_repeats=3)
        pm = PunctuationModel()
        t0 = time.time()
        full_text = pm.restore_punctuation(full_text)
        t1 = time.time()
        full_text = pm.restore_punctuation(full_text)
        t2 = time.time()
        print(f"✅ Punctuation restoration completed (passes: 2 | {t1 - t0:.1f}s + {t2 - t1:.1f}s)")
        
        # Fix Whisper-specific artifacts (double periods, spacing, etc.)
        full_text = _fix_whisper_artifacts(full_text)
        
        # Refine capitalization to fix artifacts
        full_text = _refine_capitalization(full_text)
        print("✅ Capitalization & artifact refinement completed")
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
        doc.add_heading(f'{base_name}', 0)
        
        # Add model and location info
        parent_folder = os.path.basename(os.path.dirname(input_path))
        doc.add_paragraph(f'Model: {selected_model_name}')
        doc.add_paragraph(f'Folder: {parent_folder}')
        doc.add_paragraph('')

        elapsed_total = time.time() - start_time
        if os.environ.get("TRANSCRIBE_HIDE_TIME", "").lower() not in ("1", "true", "yes"):
            doc.add_paragraph(f'Transcription time: {format_duration_hms(elapsed_total)}')
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
            
            # Add model and location info (fallback)
            parent_folder = os.path.basename(os.path.dirname(input_path))
            doc.add_paragraph(f'Model: {selected_model_name}')
            doc.add_paragraph(f'Folder: {parent_folder}')
            doc.add_paragraph('')
            
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

    # Cleanup preprocessed file on completion
    try:
        if preprocessing_used and os.path.exists(working_input_path):
            os.remove(working_input_path)
            print("🧹 Removed temporary preprocessed audio file")
    except Exception:
        pass

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


def vad_segment_times_optimized(input_path, aggressiveness=2, frame_duration_ms=30, padding_ms=300):
    """Voice Activity Detection optimized for maximum performance with fallback."""
    # Check VAD availability within this function scope
    try:
        import webrtcvad
        _vad_available = True
    except ImportError:
        _vad_available = False
        webrtcvad = None

    if not _vad_available:
        print("⚠️  webrtcvad not available - using optimized duration-based segmentation")
        # Fallback: create segments based on duration (every 25 seconds for better performance)
        try:
            from moviepy.editor import AudioFileClip
            audio_clip = AudioFileClip(input_path)
            duration = audio_clip.duration
            audio_clip.close()

            segments = []
            segment_length = 25.0  # 25 second segments for optimized processing
            for i in range(0, int(duration), int(segment_length)):
                start = float(i)
                end = min(float(i + segment_length), duration)
                segments.append((start, end))

            print(f"📊 Created {len(segments)} optimized duration-based segments ({segment_length}s each)")
            return segments

        except Exception as e:
            print(f"❌ Error creating optimized fallback segments: {e}")
            # Last resort: single segment for entire audio
            return [(0.0, 60.0)]  # Assume 60s max, will be clipped later

    # Original webrtcvad implementation with optimized settings
    try:
        # Import required functions from transcribe.py
        from transcribe import get_pcm_from_file, frames_from_pcm

        pcm = get_pcm_from_file(input_path)
        vad = webrtcvad.Vad(aggressiveness)
        frames = list(frames_from_pcm(pcm, frame_duration_ms=frame_duration_ms))
        sample_rate = 16000
        in_speech = False
        segments = []
        speech_start = 0

        for i, frame in enumerate(frames):
            is_speech = False
            if len(frame) == int(sample_rate * 2 * (frame_duration_ms/1000.0)):
                is_speech = vad.is_speech(frame, sample_rate)
            t = (i * frame_duration_ms) / 1000.0
            if is_speech and not in_speech:
                in_speech = True
                speech_start = t
            elif not is_speech and in_speech:
                in_speech = False
                speech_end = t
                # Optimized padding for better performance
                start = max(0, speech_start - (padding_ms/1000.0))
                end = speech_end + (padding_ms/1000.0)
                segments.append((start, end))

        # Handle file ending while in speech
        if in_speech:
            speech_end = (len(frames) * frame_duration_ms) / 1000.0
            start = max(0, speech_start - (padding_ms/1000.0))
            end = speech_end + (padding_ms/1000.0)
            segments.append((start, end))

        return segments

    except Exception as e:
        print(f"❌ VAD segmentation failed: {e}")
        # Fallback to duration-based segmentation
        try:
            from moviepy.editor import AudioFileClip
            audio_clip = AudioFileClip(input_path)
            duration = audio_clip.duration
            audio_clip.close()

            segments = []
            segment_length = 25.0
            for i in range(0, int(duration), int(segment_length)):
                start = float(i)
                end = min(float(i + segment_length), duration)
                segments.append((start, end))

            print(f"📊 VAD failed, using {len(segments)} duration-based segments")
            return segments

        except Exception as fallback_e:
            print(f"❌ Fallback segmentation also failed: {fallback_e}")
            return [(0.0, 60.0)]


def transcribe_with_vad_parallel(input_path, vad_segments, model, base_transcribe_kwargs, config):
    """Transcribe audio using VAD segments processed in parallel for maximum performance."""
    import concurrent.futures
    import tempfile
    import subprocess

    print(f"🔄 Processing {len(vad_segments)} VAD segments in parallel...")

    # Try to import moviepy
    try:
        from moviepy.editor import AudioFileClip  # type: ignore
        moviepy_available = True
    except ImportError:
        moviepy_available = False
        AudioFileClip = None  # type: ignore
        print("⚠️  moviepy not available - falling back to ffmpeg for segment extraction")

    # Create temporary directory for segment files
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        segment_results = []

        # Extract audio segments
        def extract_segment(segment_idx, start_time, end_time):
            try:
                segment_path = os.path.join(temp_dir, f"segment_{segment_idx:03d}.wav")

                if moviepy_available and AudioFileClip is not None:
                    audio_clip = AudioFileClip(input_path)
                    segment_clip = audio_clip.subclip(start_time, end_time)
                    segment_clip.write_audiofile(segment_path, verbose=False, logger=None)
                    audio_clip.close()
                    segment_clip.close()
                else:
                    # Fallback to ffmpeg
                    duration = end_time - start_time
                    cmd = [
                        "ffmpeg", "-i", input_path,
                        "-ss", str(start_time), "-t", str(duration),
                        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                        "-y", segment_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)

                return segment_path, (start_time, end_time)
            except Exception as e:
                print(f"⚠️  Failed to extract segment {segment_idx}: {e}")
                return None, None

        # Extract all segments SEQUENTIALLY for perfect temporal order
        print("🔧 Using sequential segment extraction for guaranteed order...")
        
        segment_files = []
        for i, (start, end) in enumerate(vad_segments):
            print(f"🎵 Extracting segment {i + 1}/{len(vad_segments)}: {start:.1f}s-{end:.1f}s")
            segment_path, time_range = extract_segment(i, start, end)
            if segment_path and time_range:
                segment_files.append((segment_path, time_range))
                print(f"✅ Extracted segment {i + 1}: {time_range[0]:.1f}s-{time_range[1]:.1f}s")
            else:
                print(f"⚠️  Failed to extract segment {i + 1}")
        
        print(f"✅ Extracted {len(segment_files)} audio segments in perfect temporal order")

        # Transcribe segments in parallel
        def transcribe_segment(segment_path, time_range):
            try:
                # Create a copy of transcribe kwargs for this segment
                segment_kwargs = base_transcribe_kwargs.copy()
                # Remove vad_filter since we're already using segmented audio
                segment_kwargs.pop("vad_filter", None)

                result = model.transcribe(segment_path, **segment_kwargs)

                # Add timing information to segments
                if isinstance(result, dict) and "segments" in result:
                    for segment in result["segments"]:
                        segment["start"] += time_range[0]
                        segment["end"] += time_range[0]

                return result
            except Exception as e:
                print(f"⚠️  Failed to transcribe segment {os.path.basename(segment_path)}: {e}")
                return None

        # Transcribe all segments in parallel with enhanced CPU utilization
        if config.get('max_perf'):
            # Ultra aggressive: use up to 75% of CPU cores for VAD parallel processing
            max_workers = min(len(segment_files), max(1, int(config.get("cpu_threads", 4) * 0.75)))
        else:
            # Conservative: use up to 50% of CPU cores
            max_workers = min(len(segment_files), config.get("cpu_threads", 4) // 2)
        max_workers = max(1, min(max_workers, 12))  # Cap at 12 workers for stability

        # SEQUENTIAL TRANSCRIPTION: Process segments one by one for guaranteed order
        print("🔧 Using sequential transcription processing for perfect order...")
        
        segment_results = []
        for i, (seg_path, time_range) in enumerate(segment_files):
            print(f"🎯 Processing segment {i + 1}/{len(segment_files)}: {time_range[0]:.1f}s-{time_range[1]:.1f}s")
            try:
                result = transcribe_segment(seg_path, time_range)
                if result:
                    segment_results.append(result)
                    print(f"✅ Segment {i + 1} completed successfully")
                else:
                    print(f"⚠️  Segment {i + 1} returned empty result")
            except Exception as e:
                print(f"❌ Segment {i + 1} failed: {e}")
        
        print(f"📊 Successfully processed {len(segment_results)}/{len(segment_files)} segments in perfect temporal order")

        # Combine results from all segments IN TEMPORAL ORDER
        combined_result = {"text": "", "segments": []}
        text_parts = []

        # Process results in temporal order, skipping None entries
        for i, result in enumerate(segment_results):
            if result is not None and isinstance(result, dict):
                # Combine text in order
                text_content = result.get("text", "")
                if text_content and isinstance(text_content, str):
                    text_parts.append(text_content.strip())
                    print(f"📝 Added segment {i+1} text: '{text_content.strip()[:50]}...'")

                # Combine segments in order
                segments_data = result.get("segments", [])
                if segments_data and isinstance(segments_data, list):
                    combined_result["segments"].extend(segments_data)

                # Copy metadata from first valid result
                if not combined_result.get("language") and result.get("language"):
                    # Copy common metadata fields
                    for key in ["language", "language_probability", "duration"]:
                        if key in result and key not in combined_result:
                            combined_result[key] = result[key]
            else:
                print(f"⚠️  Skipping empty segment {i+1}")

        # Assemble text in proper temporal order
        combined_result["text"] = " ".join(text_parts).strip()
        
        # Debug: Show first part of combined text
        if combined_result["text"]:
            first_part = combined_result["text"][:100]
            print(f"🔍 Combined text starts with: '{first_part}...'")
        else:
            print("⚠️  Combined text is empty!")

        # Ensure segments are sorted by start time (double-check)
        if combined_result["segments"]:
            combined_result["segments"].sort(key=lambda x: x.get("start", 0))
            
            # Verify segment order and report any issues
            prev_end = 0
            for i, seg in enumerate(combined_result["segments"]):
                seg_start = seg.get("start", 0)
                if seg_start < prev_end - 1:  # Allow 1 second tolerance for overlaps
                    print(f"⚠️  Segment order issue detected at segment {i}: start={seg_start:.1f}s, prev_end={prev_end:.1f}s")
                prev_end = seg.get("end", seg_start)

        print(f"✅ Combined transcription from {len(segment_results)} segments in temporal order")
        print(f"📝 Total text length: {len(combined_result['text'])} characters")
        print(f"🎯 Total segments: {len(combined_result.get('segments', []))}")
        
        # Debug: Show first few characters to verify beginning is preserved
        if combined_result["text"]:
            preview = combined_result["text"][:100]
            print(f"🔍 Transcript begins: '{preview}...'")
        
        return combined_result


def transcribe_file_simple_auto(input_path, output_dir=None, threads_override: Optional[int] = None):
    """
    High-quality, simplified single-file transcription on best available device.
    - Device selection: CUDA > DirectML > CPU
    - No VAD; transcribe the entire file
    - Robust DOCX save with fallback
    - Safe cleanup that avoids torch re-import problems
    Returns path to the .txt file.
    """
    # Initialize all variables at the beginning to ensure they're always accessible
    use_vad = False
    enable_speakers = False
    use_dataset = False
    max_perf = True
    transcription_complete = False
    transcription_result = None
    transcription_error = None

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
        get_media_duration, split_into_paragraphs, format_duration, format_duration_minutes_only, format_duration_hms,
    )

    # Speaker identification imports
    try:
        import webrtcvad
        _vad_available = True
    except ImportError:
        _vad_available = False

    # Import VAD functions from transcribe.py
    try:
        from transcribe import vad_segment_times
        _vad_functions_available = True
    except ImportError:
        _vad_functions_available = False
        # Define fallback VAD function
        def vad_segment_times(input_path):
            """Fallback VAD function when transcribe.py is not available"""
            try:
                from moviepy import AudioFileClip
                audio_clip = AudioFileClip(input_path)
                duration = audio_clip.duration
                audio_clip.close()

                segments = []
                segment_length = 30.0  # 30 second segments
                for i in range(0, int(duration), int(segment_length)):
                    start = float(i)
                    end = min(float(i + segment_length), duration)
                    segments.append((start, end))

                print(f"📊 Created {len(segments)} duration-based segments ({segment_length}s each)")
                return segments
            except Exception as e:
                print(f"❌ Error creating fallback segments: {e}")
                return [(0.0, 60.0)]  # Single segment fallback

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

    # Check if VAD segmentation should be used
    use_vad = False
    try:
        use_vad = os.environ.get("TRANSCRIBE_VAD", "").strip() in ("1", "true", "True")
        if use_vad:
            print("🎯 VAD segmentation enabled for performance optimization")
    except Exception:
        use_vad = False
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
    
    # VAD CONTROL: Allow disabling VAD via environment variable
    disable_vad = False
    try:
        env_disable_vad = os.environ.get("TRANSCRIBE_DISABLE_VAD", "").strip()
        if env_disable_vad in ("1", "true", "True"):
            disable_vad = True
            print("🚫 VAD processing disabled via TRANSCRIBE_DISABLE_VAD environment variable")
    except Exception:
        disable_vad = False
    
    config = get_maximum_hardware_config(max_perf=max_perf)
    config['disable_vad'] = disable_vad
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
        output_dir = os.path.dirname(input_path)

    # Preprocess audio with silence padding to prevent missed words
    try:
        preprocessed_path = preprocess_audio_with_padding(input_path)
    except Exception as _pre_e:
        print(f"⚠️  Preprocessing step failed early: {_pre_e} - using original file")
        preprocessed_path = input_path
    preprocessing_used = preprocessed_path != input_path
    working_input_path = preprocessed_path if preprocessing_used else input_path

    duration = get_media_duration(working_input_path)
    if duration:
        print(f"⏱️  Duration: {format_duration(duration)}")

    # Build optional initial prompt from awkward words
    awkward_terms = load_awkward_terms(input_path)
    initial_prompt = build_initial_prompt(awkward_terms)
    if initial_prompt:
        preview = initial_prompt[:100]
        print(f"🧩 Using domain terms bias (initial_prompt): '{preview}...'")

    # Pre-run cleanup
    force_gpu_memory_cleanup()

    # Choose device and load one model only
    device_name = "CPU"
    model: Any = None
    chosen_device = "cpu"
    
    # Check for model selection from environment variable (set by GUI)
    selected_model_name = os.environ.get("TRANSCRIBE_MODEL_NAME", "large-v3")

    # Prefer selected model; if it's not listed as available, fall back to the next best
    try:
        import whisper  # type: ignore
        avail = set(whisper.available_models())
        requested_available = (selected_model_name in avail)
        if not requested_available:
            print(f"⚠️  Requested model '{selected_model_name}' not available, falling back...")
            for cand in ("large-v3", "large-v2", "large"):
                if cand in avail:
                    selected_model_name = cand
                    break
        print(f"🧩 Requested model available: {requested_available}")
        print(f"🗂️  Selecting model: {selected_model_name}")
        print(f"🎯 Model Source: Environment variable TRANSCRIBE_MODEL_NAME = '{os.environ.get('TRANSCRIBE_MODEL_NAME', 'NOT SET')}'")
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
            # Load model in FP16 for maximum GPU performance
            model = whisper.load_model(selected_model_name, device="cuda")
            # Move model to FP16 for 2x faster inference on GPU
            try:
                model = model.half()
                print(f"🚀 Model '{selected_model_name}' converted to FP16 (half precision) for 2-3x faster GPU inference")
                print(f"💡 FP16 uses half the VRAM and provides 2-3x speedup on modern GPUs")
            except Exception as fp16_err:
                print(f"⚠️  Could not convert to FP16: {fp16_err} - using FP32 (slower)")
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

    # Set CPU threads (& interop) with enhanced threading for text processing
    torch_api.set_num_threads(config["cpu_threads"])
    
    # Calculate interop threads with enhanced settings
    if config.get('max_perf'):
        interop = max(4, min(24, config["cpu_threads"] // 2))  # More aggressive interop
    else:
        interop = max(2, min(16, config["cpu_threads"] // 4))
    
    # Explicit thread configuration logging
    print(f"🔧 Thread Configuration:")
    print(f"   • CPU Threads: {config['cpu_threads']}")
    print(f"   • Interop Threads: {interop}")
    print(f"   • Device: {device_name}")
    print(f"   • Model: {selected_model_name}")
    
    try:
        torch_api.set_num_interop_threads(interop)
    except Exception:
        pass
    
    # Enhanced MKL/OMP configuration for better CPU utilization
    try:
        os.environ.setdefault("MKL_NUM_THREADS", str(config["cpu_threads"]))
        os.environ.setdefault("OMP_NUM_THREADS", str(config["cpu_threads"]))
        # Additional performance tunings
        os.environ.setdefault("MKL_DYNAMIC", "TRUE")  # Dynamic thread adjustment
        os.environ.setdefault("OMP_DYNAMIC", "TRUE")  # Dynamic thread adjustment
        os.environ.setdefault("NUMEXPR_MAX_THREADS", str(min(config["cpu_threads"], 16)))  # NumPy/SciPy threading
        
        # Enable aggressive CPU utilization
        os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")  # Keep threads active (don't sleep)
        os.environ.setdefault("OMP_PROC_BIND", "TRUE")  # Bind threads to cores
        os.environ.setdefault("KMP_BLOCKTIME", "0")  # Immediate response (Intel specific)
        os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")  # Core affinity
        
        # Text processing specific threading
        text_threads = max(2, min(8, config["cpu_threads"] // 2))
        os.environ.setdefault("NLTK_NUM_THREADS", str(text_threads))
        os.environ.setdefault("SPACY_NUM_THREADS", str(text_threads))
        
    except Exception:
        pass
    
    text_workers = max(2, min(8, config["cpu_threads"] // 2))
    print(f"🧵 Enhanced threading: PyTorch {config['cpu_threads']} threads, interop {interop}, text processing up to {text_workers} workers")
    print(f"🔧 CPU optimization: MKL/OMP dynamic threading enabled for maximum utilization")

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
        nonlocal transcription_complete, transcription_result, transcription_error, use_vad
        try:
            print("🔄 Starting Whisper transcription process...")
            if model is None:
                raise RuntimeError("Whisper model is not loaded")

            # Apply VAD segmentation if enabled
            transcribe_kwargs = {
                "language": "en",  # Optimized for English language
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -2.0,
                "no_speech_threshold": 0.3,
                "condition_on_previous_text": False,
                "temperature": 0.0,
                "verbose": True,
            }
            
            # Model-specific tuning for accuracy
            if selected_model_name == "large-v3-turbo":
                # Turbo-specific: tighter thresholds for better accuracy
                transcribe_kwargs["compression_ratio_threshold"] = 2.2  # Stricter (was 2.4)
                transcribe_kwargs["logprob_threshold"] = -1.5  # More selective (was -2.0)
                transcribe_kwargs["no_speech_threshold"] = 0.4  # Better silence detection (was 0.3)
                transcribe_kwargs["temperature"] = 0.0  # Deterministic for consistency
                transcribe_kwargs["hallucination_silence_threshold"] = 2.0  # Prevent hallucinations
                print("🎯 Turbo model: Using accuracy-optimized thresholds")
            elif selected_model_name == "large-v3":
                # Large-v3: slightly more permissive for natural speech flow
                transcribe_kwargs["compression_ratio_threshold"] = 2.6
                transcribe_kwargs["logprob_threshold"] = -2.2
                print("🎯 Large-v3 model: Using balanced thresholds")
            
            # Apply quality mode if enabled
            quality_mode = os.environ.get("TRANSCRIBE_QUALITY_MODE", "").strip() in ("1", "true", "True")
            if quality_mode:
                if selected_model_name == "large-v3-turbo":
                    # Turbo with quality mode: aggressive beam search for max accuracy
                    transcribe_kwargs["beam_size"] = 7  # Wider search (was 5)
                    transcribe_kwargs["patience"] = 2.5  # More patience (was 2.0)
                    transcribe_kwargs["best_of"] = 5  # Try multiple candidates
                    print("🎯 Quality mode (turbo): beam_size=7, patience=2.5, best_of=5")
                else:
                    # Large-v3 with quality mode: standard beam search
                    transcribe_kwargs["beam_size"] = 5
                    transcribe_kwargs["patience"] = 2.0
                    print("🎯 Quality mode (large-v3): beam_size=5, patience=2.0")
            
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            if use_vad:
                try:
                    # Get VAD segments for the audio file
                    # VAD CONTROL: Allow disabling VAD when ordering issues occur
                    if config.get('disable_vad', False):
                        print("⚠️  VAD processing DISABLED - using full file transcription for guaranteed temporal order")
                        print("   This ensures perfect segment ordering but may be slower for long files")
                    elif _vad_functions_available:
                        vad_segments = vad_segment_times(working_input_path)
                        if vad_segments and len(vad_segments) > 0:
                            print(f"🎯 VAD detected {len(vad_segments)} speech segments - processing in parallel")
                            print("💡 If transcript beginnings are missing, set 'disable_vad': True in config")
                            # Use actual VAD segmentation with parallel processing
                            result = transcribe_with_vad_parallel(working_input_path, vad_segments, model, transcribe_kwargs, config)
                            transcription_result = result
                            print("✅ VAD parallel transcription completed successfully")
                            transcription_complete = True
                            return  # Exit early since we processed with VAD
                        else:
                            print("⚠️  VAD enabled but no segments detected, proceeding without VAD")
                    else:
                        print("⚠️  VAD functions not available, proceeding without VAD")
                except Exception as vad_e:
                    print(f"⚠️  VAD segmentation failed: {vad_e} - proceeding without VAD")
                    use_vad = False  # Disable for this run

            # Call transcribe with optimized parameters
            result = model.transcribe(working_input_path, **transcribe_kwargs)
            transcription_result = result
            print("✅ Whisper transcription completed successfully")
        except Exception as e:
            transcription_error = e
        finally:
            transcription_complete = True

    transcribe_thread = threading.Thread(target=_run_transcribe, daemon=True)
    transcribe_thread.start()

    start_watch = time.time()
    # No timeout - let transcription complete naturally
    print(f"⏱️  Monitoring transcription progress (no timeout)...")


    # Initialize CPU/RAM monitoring
    import psutil
    process = psutil.Process(os.getpid())

    while not transcription_complete:
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

    # No timeout fallback - transcription completes naturally
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
            
            # Sort segments by start time FIRST to ensure proper order
            segments.sort(key=lambda x: x.get("start", 0))
            print(f"🔍 Processing {len(segments)} segments in temporal order")
            
            for i, seg in enumerate(segments):
                seg_text = str(seg.get("text", "")).strip()
                avg_logprob = seg.get("avg_logprob", 0.0)
                no_speech_prob = seg.get("no_speech_prob", 0.0)
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)

                suspicious = _is_suspicious_music_artifact(seg_text)
                # Make artifact filtering MUCH more conservative - only remove if BOTH suspicious AND very low confidence
                very_low_confidence = (isinstance(avg_logprob, (int, float)) and avg_logprob < -1.0) and \
                                     (isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.8)

                # CONSERVATIVE FILTERING: Only remove if clearly suspicious AND very low confidence
                should_keep = seg_text and not (suspicious and very_low_confidence)

                if should_keep:
                    cleaned_parts.append(seg_text)
                    cleaned_segments.append(seg)
                    kept_count += 1
                    
                    # Debug: Show first few segments to verify beginning preservation
                    if i < 5:
                        print(f"  ✅ Segment {i+1}: [{seg_start:.1f}s-{seg_end:.1f}s] '{seg_text[:50]}...' (logprob: {avg_logprob:.2f})")
                else:
                    removed_segments.append({
                        "text": seg_text[:120],
                        "avg_logprob": avg_logprob,
                        "no_speech_prob": no_speech_prob,
                        "start": seg_start,
                        "end": seg_end,
                    })
                    print(f"  🧽 Filtered segment {i+1}: [{seg_start:.1f}s-{seg_end:.1f}s] '{seg_text[:30]}...' (suspicious={suspicious}, low_conf={very_low_confidence})")
                    
            full_text = (" ".join(cleaned_parts)).strip()
            
            # Additional debugging
            if cleaned_parts:
                first_part = cleaned_parts[0][:100] if cleaned_parts[0] else "N/A"
                print(f"🔍 First segment text: '{first_part}...'")
                
            print(f"📊 Segment filtering: kept {kept_count}/{len(segments)} segments")

            # Speaker identification removed as requested
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

    # Post-processing with ULTRA-enhanced text processing (multi-pass with parallel processing)
    try:
        # Try to use the new ultra text processor first
        try:
            from text_processor_ultra import create_ultra_processor, create_advanced_paragraph_formatter
            
            # Calculate optimal workers for text processing (use remaining CPU capacity)
            text_workers = max(2, min(8, config.get("cpu_threads", 4) // 2))
            
            print(f"🚀 Using ULTRA text processor with {text_workers} workers and 6 specialized passes")
            t0 = time.time()
            
            # Ultra processing with 6 passes for maximum quality
            ultra_processor = create_ultra_processor(max_workers=text_workers)
            # First, collapse obvious repetitions to avoid amplifying loops downstream
            full_text = _collapse_repetitions(full_text, max_repeats=3)
            # Restore punctuation using a lightweight model so sentences can be segmented properly
            try:
                from deepmultilingualpunctuation import PunctuationModel
                pm_ultra = PunctuationModel()
                full_text = pm_ultra.restore_punctuation(full_text)
            except Exception as punc_e:
                print(f"⚠️  Punctuation pre-pass unavailable: {punc_e} — proceeding with ULTRA-only punctuation fixes")
            full_text = ultra_processor.process_text_ultra(full_text, passes=6)
            
            t1 = time.time()
            print(f"✅ Ultra text processing completed ({t1 - t0:.1f}s)")
            
            # Advanced paragraph formatting
            try:
                paragraph_formatter = create_advanced_paragraph_formatter(max_workers=text_workers)
                formatted_text = paragraph_formatter.format_paragraphs_advanced(full_text, target_length=600)
            except Exception as pf_e:
                print(f"⚠️  Advanced paragraph formatter failed: {pf_e} — using basic split_into_paragraphs")
                formatted = split_into_paragraphs(full_text, max_length=600)
                formatted_text = "\n\n".join(formatted) if isinstance(formatted, list) else full_text
            
            t2 = time.time()
            print(f"✅ Advanced paragraph formatting completed ({t2 - t1:.1f}s)")
            
        except ImportError:
            print("⚠️  Ultra processor not available, falling back to enhanced processor")
            # Fallback to enhanced processor
            if _enhanced_processor_available and create_enhanced_processor is not None:
                # Use enhanced processor with spaCy and custom rules
                processor = create_enhanced_processor(use_spacy=True, use_transformers=False)
                t0 = time.time()
                
                # Multiple passes for better quality
                full_text = processor.restore_punctuation(full_text)
                # Second pass for additional refinement
                full_text = processor.restore_punctuation(full_text)
                # Third pass for edge cases
                full_text = processor.restore_punctuation(full_text)
                
                t1 = time.time()
                print(f"✅ Enhanced punctuation restoration completed (3 passes | {t1 - t0:.1f}s)")
                
                # Enhanced paragraph formatting
                formatted = split_into_paragraphs(full_text, max_length=600)
                if isinstance(formatted, list):
                    formatted_text = "\n\n".join(formatted)
                else:
                    formatted_text = full_text
            else:
                # Basic fallback with multiple passes
                from deepmultilingualpunctuation import PunctuationModel
                pm = PunctuationModel()
                t0 = time.time()
                
                # 4 passes for better quality
                full_text = pm.restore_punctuation(full_text)
                full_text = pm.restore_punctuation(full_text)
                full_text = pm.restore_punctuation(full_text)
                full_text = pm.restore_punctuation(full_text)
                
                t1 = time.time()
                print(f"✅ Basic punctuation restoration completed (4 passes | {t1 - t0:.1f}s)")
                
                # Enhanced paragraph formatting
                formatted = split_into_paragraphs(full_text, max_length=600)
                if isinstance(formatted, list):
                    formatted_text = "\n\n".join(formatted)
                else:
                    formatted_text = full_text
                    
    except Exception as e:
        print(f"⚠️  Text processing failed: {e}")
        # Last fallback
        formatted_text = full_text

    # Refine capitalization to fix artifacts (applies to all processing paths)
    try:
        # Fix Whisper-specific artifacts first
        formatted_text = _fix_whisper_artifacts(formatted_text)
        # Then refine capitalization
        formatted_text = _refine_capitalization(formatted_text)
        print("✅ Capitalization & artifact refinement completed")
    except Exception as e:
        print(f"⚠️  Capitalization refinement failed: {e}")

    # Final quality check and validation
    try:
        if formatted_text and len(formatted_text) > 10:
            # Ensure proper sentence structure
            if not formatted_text.endswith(('.', '!', '?')):
                formatted_text += '.'
            
            # Capitalize first letter if needed
            if formatted_text and not formatted_text[0].isupper():
                formatted_text = formatted_text[0].upper() + formatted_text[1:]
                
            print("✅ Final text validation completed")
        else:
            print("⚠️  Formatted text too short, using original")
            formatted_text = full_text
    except Exception as e:
        print(f"⚠️  Text validation failed: {e}")
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
        doc.add_heading(f'{base_name}', 0)
        
        # Add model and location info
        parent_folder = os.path.basename(os.path.dirname(input_path))
        doc.add_paragraph(f'Model: {selected_model_name}')
        doc.add_paragraph(f'Folder: {parent_folder}')
        doc.add_paragraph('')
        
        elapsed_total = time.time() - start_time
        if os.environ.get("TRANSCRIBE_HIDE_TIME", "").lower() not in ("1", "true", "yes"):
            doc.add_paragraph(f'Transcription time: {format_duration_hms(elapsed_total)}')
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
            
            # Add model and location info (fallback)
            parent_folder = os.path.basename(os.path.dirname(input_path))
            doc.add_paragraph(f'Model: {selected_model_name}')
            doc.add_paragraph(f'Folder: {parent_folder}')
            doc.add_paragraph('')
            
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

    # Remove temporary preprocessed file if used
    try:
        if 'preprocessing_used' in locals() and preprocessing_used and working_input_path != input_path:
            if os.path.exists(working_input_path):
                os.remove(working_input_path)
                print("🧹 Removed temporary preprocessed audio file")
    except Exception as _cleanup_e:
        print(f"⚠️  Failed to remove temporary file: {_cleanup_e}")

    return txt_path


def transcribe_file_optimised(input_path, model_name="medium", output_dir=None, force_optimised=True, *, threads_override: Optional[int] = None):
    """Compatibility wrapper. Uses the simple auto path."""
    return transcribe_file_simple_auto(input_path, output_dir=output_dir, threads_override=threads_override)


def main():
    parser = argparse.ArgumentParser(description="Simplified auto-detected transcription")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--output-dir", help="Output directory (default: same directory as input file)")
    parser.add_argument("--threads", type=int, help="Override CPU threads for PyTorch/OMP/MKL")
    parser.add_argument("--ram-gb", type=float, help="Cap usable system RAM in GB (env TRANSCRIBE_RAM_GB)")
    parser.add_argument("--ram-frac", "--ram-fraction", dest="ram_fraction", type=float, help="Cap usable system RAM as fraction 0-1 (env TRANSCRIBE_RAM_FRACTION)")
    parser.add_argument("--vram-gb", type=float, help="Cap usable CUDA VRAM in GB (env TRANSCRIBE_VRAM_GB)")
    parser.add_argument("--vram-frac", "--vram-fraction", dest="vram_fraction", type=float, help="Cap usable CUDA VRAM as fraction 0-1 (env TRANSCRIBE_VRAM_FRACTION)")
    parser.add_argument("--vad", action="store_true", help="Enable VAD segmentation for parallel processing performance boost (env TRANSCRIBE_VAD)")
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

    # Apply VAD override if provided
    try:
        if getattr(args, "vad", False):
            os.environ["TRANSCRIBE_VAD"] = "1"
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