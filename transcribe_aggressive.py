"""
Aggressive GPU+CPU hybrid transcription with maximum hardware utilization.
This implementation will fully utilize both your GTX 1070 Ti and all 32 CPU cores.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

import sys
import os
import shutil
import subprocess
import tempfile
import argparse
import whisper
import torch
from docx import Document
from moviepy import VideoFileClip, AudioFileClip
import imageio_ffmpeg as iio_ffmpeg
from deepmultilingualpunctuation import PunctuationModel
import webrtcvad
import re
import time
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import queue
import threading
import psutil
import gc


def adjust_workers_for_model(config, model_name):
    """Adjust worker counts based on actual model size - AGGRESSIVE RAM utilization."""
    model_ram_usage = {
        'tiny': 0.5, 'base': 0.8, 'small': 1.5, 'medium': 2.5, 
        'large': 4.0, 'large-v2': 4.0, 'large-v3': 4.0
    }
    
    actual_ram_per_model = model_ram_usage.get(model_name, 2.5)
    # AGGRESSIVE: Use almost all RAM - reserve only 1GB (was 4GB)
    usable_ram = max(config["available_ram_gb"] - 0.3, 1.0)  # Ultra-aggressive: Reserve only 300MB
    
    # Recalculate CPU workers based on actual model RAM usage
    max_cpu_workers = int(usable_ram / actual_ram_per_model)
    # AGGRESSIVE: Allow more workers (was limited to 8)
    safe_cpu_workers = min(config["cpu_workers"], max_cpu_workers, 20)
    
    if safe_cpu_workers != config["cpu_workers"]:
        print(f"🧠 Model '{model_name}' requires {actual_ram_per_model}GB RAM per instance")
        print(f"   AGGRESSIVE adjustment: {config['cpu_workers']} → {safe_cpu_workers} (using {usable_ram:.1f}GB)")
        
        config["cpu_workers"] = safe_cpu_workers
        config["total_workers"] = config["gpu_workers"] + safe_cpu_workers
        config["actual_ram_per_model"] = actual_ram_per_model
    
    return config


def extract_segments_aggressive(audio_path, segments, temp_dir, config):
    """Ultra-fast parallel segment extraction using all available CPU cores."""
    print(f"🎵 Extracting {len(segments)} segments with {config['segment_extraction_workers']} parallel processes...")
    
    # Send initial progress update for extraction phase
    try:
        import sys
        if hasattr(sys.stdout, 'output_queue'):
            print(f"PROGRESS:0.0|Extracting audio segments...|{config['segment_extraction_workers']}|")
    except:
        pass
    
    # Prepare parallel extraction
    extraction_start = time.time()
    segment_args = [(i, start, end, audio_path, temp_dir) for i, (start, end) in enumerate(segments)]
    
    # Try ProcessPoolExecutor first, fallback to sequential if it fails
    successful_segments = []
    
    try:
        # Use ProcessPoolExecutor for true CPU parallelism with timeout and error handling
        with ProcessPoolExecutor(max_workers=min(config["segment_extraction_workers"], 6)) as executor:
            # Submit all extraction tasks
            future_to_segment = {}
            for args in segment_args:
                future = executor.submit(extract_single_segment_worker, args)
                future_to_segment[future] = args[0]
            
            # Collect results with progress tracking
            for future in as_completed(future_to_segment, timeout=300):
                segment_index = future_to_segment[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        successful_segments.append(result)
                        print(f"Extracted segment {segment_index + 1}/{len(segments)}")
                except Exception as e:
                    print(f"Failed to extract segment {segment_index}: {e}")
                    continue
    
    except Exception as e:
        print(f"ProcessPoolExecutor failed, falling back to sequential processing: {e}")
        # Fallback to sequential processing
        for i, (start, end) in enumerate(segments):
            try:
                result = extract_single_segment_worker((i, start, end, audio_path, temp_dir))
                if result:
                    successful_segments.append(result)
                    print(f"Extracted segment {i + 1}/{len(segments)}")
            except Exception as seg_e:
                print(f"Failed to extract segment {i}: {seg_e}")
                continue
    
    extraction_time = time.time() - extraction_start
    print(f"\n⚡ Extracted {len(successful_segments)} segments in {extraction_time:.1f}s")
    if len(successful_segments) > 0:
        print(f"   Speed: {len(successful_segments) / extraction_time:.1f} segments/second")
    
    return [(path, start, end) for _, path, start, end in successful_segments]


def get_optimal_worker_counts():
    """Determine optimal worker distribution for maximum hardware utilization with RAM constraints."""
    cpu_cores = multiprocessing.cpu_count()
    has_cuda = torch.cuda.is_available()
    
    # Get system memory info
    memory = psutil.virtual_memory()
    total_ram_gb = memory.total / (1024**3)
    available_ram_gb = memory.available / (1024**3)
    
    print(f"🖥️  System Resources:")
    print(f"   RAM: {available_ram_gb:.1f}GB available / {total_ram_gb:.1f}GB total")
    print(f"   CPU cores: {cpu_cores}")
    
    # Estimate RAM usage per model (conservative estimates)
    model_ram_usage = {
        'tiny': 0.5,      # ~500MB
        'base': 0.8,      # ~800MB  
        'small': 1.5,     # ~1.5GB
        'medium': 2.5,    # ~2.5GB
        'large': 4.0,     # ~4GB
        'large-v2': 4.0,  # ~4GB
        'large-v3': 4.0   # ~4GB
    }
    
    if has_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: NVIDIA {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB VRAM)")
        
        # Conservative GPU+CPU configuration with RAM awareness
        gpu_workers = 2  # Keep GPU workers low to save VRAM
        
        # Calculate AGGRESSIVE CPU workers based on available RAM
        # Reserve only 1GB for system + other processes (was 4GB - now truly aggressive!)
        usable_ram = max(available_ram_gb - 0.3, 1.0)  # Ultra-aggressive: Reserve only 300MB
        
        # Estimate CPU workers based on RAM per model instance
        # We'll determine model size later, so use medium as baseline
        estimated_ram_per_worker = model_ram_usage.get('medium', 2.5)
        max_cpu_workers_by_ram = int(usable_ram / estimated_ram_per_worker)
        
        # AGGRESSIVE: Use ALL available CPU cores and RAM capacity
        cpu_workers = min(
            cpu_cores - 1,  # Leave only 1 core free (was 4!)
            max_cpu_workers_by_ram,  # RAM constraint
            20  # Higher maximum (was 12)
        )
        
        print(f"🚀 AGGRESSIVE RAM-Maxed Hybrid Config:")
        print(f"   GPU Workers: {gpu_workers} (CUDA parallel streams)")
        print(f"   CPU Workers: {cpu_workers} (MAXED - using {usable_ram:.1f}GB usable RAM)")
        print(f"   Total Workers: {gpu_workers + cpu_workers}")
        
        return {
            "gpu_workers": gpu_workers,
            "cpu_workers": cpu_workers,
            "total_workers": gpu_workers + cpu_workers,
            "segment_extraction_workers": min(cpu_cores, 8),  # Conservative for parallel ffmpeg extraction
            "ram_constraint": True,
            "available_ram_gb": available_ram_gb,
            "estimated_ram_per_model": estimated_ram_per_worker
        }
    else:
        # AGGRESSIVE CPU-only mode with maximum RAM utilization
        usable_ram = max(available_ram_gb - 0.5, 1.0)  # Reserve only 500MB (was 2GB)
        estimated_ram_per_worker = model_ram_usage.get('medium', 2.5)
        max_cpu_workers_by_ram = int(usable_ram / estimated_ram_per_worker)
        
        cpu_workers = min(
            cpu_cores,  # Use ALL CPU cores
            max_cpu_workers_by_ram,
            16  # Higher maximum (was 8)
        )
        
        print(f"💻 AGGRESSIVE CPU-Only Config:")
        print(f"   CPU Workers: {cpu_workers} (MAXED - using {usable_ram:.1f}GB usable RAM)")
        
        return {
            "gpu_workers": 0,
            "cpu_workers": cpu_workers,
            "total_workers": cpu_workers,
            "segment_extraction_workers": min(cpu_cores, 12),  # More extraction workers
            "ram_constraint": True,
            "available_ram_gb": available_ram_gb,
            "estimated_ram_per_model": estimated_ram_per_worker
        }


def monitor_system_usage():
    """Monitor CPU and GPU usage during processing."""
    def monitor_worker():
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # GPU monitoring if available
            gpu_info = ""
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
                    gpu_info = f" | GPU: {gpu_memory:.1f}/{gpu_memory_max:.1f}GB"
                except:
                    gpu_info = " | GPU: Active"
            
            print(f"📊 System Load: CPU {cpu_percent:5.1f}% | RAM {memory_percent:4.1f}%{gpu_info}", end="\r")
            time.sleep(2)
    
    monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
    monitor_thread.start()
    return monitor_thread


def load_models_aggressive(model_name="medium", config=None):
    """Load multiple model instances for maximum parallel processing with RAM monitoring."""
    if config is None:
        config = {"gpu_workers": 0, "cpu_workers": 1}
    
    models = {}
    
    # Safety check: ensure minimum RAM available
    memory = psutil.virtual_memory()
    if memory.available / (1024**3) < 3.0:  # Less than 3GB available
        print(f"⚠️  WARNING: Only {memory.available / (1024**3):.1f}GB RAM available")
        print("   Switching to conservative single-model mode to prevent system crash")
        
        # Emergency fallback: single model only
        try:
            if torch.cuda.is_available():
                model = whisper.load_model(model_name, device="cuda")
                models["gpu_0"] = model
                config["gpu_workers"] = 1
                config["cpu_workers"] = 0
            else:
                model = whisper.load_model(model_name, device="cpu")
                models["cpu_0"] = model
                config["gpu_workers"] = 0
                config["cpu_workers"] = 1
            
            config["total_workers"] = 1
            print(f"🚨 Emergency mode: Single {list(models.keys())[0]} model loaded")
            return models
            
        except Exception as e:
            raise RuntimeError(f"Failed to load even a single model: {e}")
    # Monitor RAM usage during model loading
    initial_memory = psutil.virtual_memory()
    print(f"📊 Initial RAM: {initial_memory.available / (1024**3):.1f}GB available")
    
    if config["gpu_workers"] > 0:
        print(f"🧠 Loading {config['gpu_workers']} GPU model instances...")
        # Load multiple GPU instances for parallel CUDA streams
        for i in range(config["gpu_workers"]):
            try:
                model = whisper.load_model(model_name, device="cuda")
                # Force model to stay on GPU
                model = model.cuda()
                models[f"gpu_{i}"] = model
                
                # Monitor RAM after each model load
                current_memory = psutil.virtual_memory()
                ram_used = (initial_memory.available - current_memory.available) / (1024**3)
                print(f"   GPU Model {i}: Loaded on {next(model.parameters()).device} (RAM used: {ram_used:.1f}GB)")
                
                # Clear cache to prevent accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   GPU Model {i}: Failed - {e}")
    
    if config["cpu_workers"] > 0:
        # Adjust CPU model count based on RAM constraints
        max_cpu_models = min(
            config["cpu_workers"] // 4,  # Share CPU models across workers
            4,  # Reasonable maximum
            int(config.get("available_ram_gb", 8) / config.get("actual_ram_per_model", 2.5))  # RAM constraint
        )
        
        print(f"🖥️  Loading {max_cpu_models} CPU model instances (RAM-limited)...")
        
        for i in range(max_cpu_models):
            try:
                current_memory = psutil.virtual_memory()
                if current_memory.available / (1024**3) < 2.0:  # Less than 2GB available
                    print(f"   ⚠️  Skipping CPU Model {i}: Insufficient RAM ({current_memory.available / (1024**3):.1f}GB)")
                    break
                    
                model = whisper.load_model(model_name, device="cpu")
                models[f"cpu_{i}"] = model
                
                # Monitor RAM usage
                new_memory = psutil.virtual_memory()
                ram_used = (current_memory.available - new_memory.available) / (1024**3)
                print(f"   CPU Model {i}: Loaded (RAM used: {ram_used:.1f}GB, {new_memory.available / (1024**3):.1f}GB available)")
                
            except Exception as e:
                print(f"   CPU Model {i}: Failed - {e}")
                break
    
    # Final memory check
    final_memory = psutil.virtual_memory()
    total_ram_used = (initial_memory.available - final_memory.available) / (1024**3)
    print(f"📊 Total models loaded: {len(models)} (Total RAM used: {total_ram_used:.1f}GB)")
    
    return models


def transcribe_segment_aggressive(model, audio_path, segment_info, worker_id):
    """Aggressive transcription with performance monitoring."""
    start_time = time.time()
    segment_path, seg_start, seg_end = segment_info
    
    try:
        # Aggressive Whisper parameters
        result = model.transcribe(
            segment_path,
            language=None,
            compression_ratio_threshold=float('inf'),  # Bypass content filtering
            logprob_threshold=-1.0,                    # Accept all confidence levels
            no_speech_threshold=0.05,                  # Very sensitive speech detection
            condition_on_previous_text=False,          # No context dependency
            temperature=0.0,                           # Deterministic
            beam_size=5,                              # Higher quality beam search
            patience=1.0                              # More thorough processing
        )
        
        text = result.get("text", "").strip()
        duration = time.time() - start_time
        
        return {
            "text": text, 
            "start": seg_start, 
            "end": seg_end,
            "worker_id": worker_id,
            "processing_time": duration
        }
        
    except Exception as e:
        print(f"\n❌ Worker {worker_id} error: {e}")
        return {
            "text": "", 
            "start": seg_start, 
            "end": seg_end,
            "worker_id": worker_id,
            "processing_time": 0
        }


def extract_single_segment_worker(args):
    """Worker function for parallel segment extraction."""
    idx, start_time, end_time, audio_path, temp_dir = args
    output_path = os.path.join(temp_dir, f"seg_{idx:04d}.mp3")
    
    try:
        # High-speed ffmpeg extraction
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-acodec", "libmp3lame",
            "-b:a", "128k",  # Lower bitrate for speed
            "-ar", "16000",
            "-ac", "1",
            "-threads", "1",  # Single thread per process
            output_path
        ]
        subprocess.run(cmd, check=True, timeout=60)
        return (idx, output_path, start_time, end_time)
    except Exception as e:
        print(f"\n❌ Segment {idx} extraction failed: {e}")
        return None


def extract_single_segment_worker_thread(args):
    """Thread-safe worker function for segment extraction fallback."""
    idx, start_time, end_time, audio_path, temp_dir = args
    output_path = os.path.join(temp_dir, f"seg_{idx:04d}.mp3")
    
    try:
        # Conservative ffmpeg extraction for thread safety
        cmd = [
            "ffmpeg", "-y", "-v", "error",  # Show only errors
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-acodec", "libmp3lame",
            "-b:a", "64k",   # Even lower bitrate for stability
            "-ar", "16000",
            "-ac", "1",
            "-threads", "1",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(output_path):
            return (idx, output_path, start_time, end_time)
        else:
            print(f"\n❌ Thread segment {idx} failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"\n❌ Thread segment {idx} exception: {e}")
        return None


def transcribe_parallel_aggressive(models, segments, config):
    """Maximum parallel transcription using all GPU and CPU resources with progress tracking."""
    
    # Safety check for None or empty segments
    if segments is None:
        print("❌ Error: segments parameter is None")
        return []
    
    if len(segments) == 0:
        print("⚠️  Warning: No segments to process")
        return []
    
    print(f"🗣️  Starting aggressive parallel transcription...")
    print(f"   Processing {len(segments)} segments with {config['total_workers']} workers")
    
    # Start system monitoring
    monitor_thread = monitor_system_usage()
    
    results = []
    segment_queue = queue.Queue()
    completed_segments = 0
    start_time = time.time()
    
    # Add all segments to processing queue
    for i, segment in enumerate(segments):
        segment_queue.put((i, segment))
    
    def send_progress_update():
        """Send progress update to GUI if running in GUI mode."""
        nonlocal completed_segments
        if completed_segments > 0:
            percentage = (completed_segments / len(segments)) * 100
            elapsed = time.time() - start_time
            active_threads = threading.active_count() - 1  # Subtract main thread
            
            status = f"Transcribing: {completed_segments}/{len(segments)} segments"
            elapsed_str = f"Elapsed: {elapsed:.0f}s"
            
            # Try to send to GUI queue if available
            try:
                # This will only work if called from GUI context
                import sys
                if hasattr(sys.stdout, 'output_queue'):
                    print(f"PROGRESS:{percentage:.1f}|{status}|{active_threads}|{elapsed_str}")
            except:
                pass  # Not in GUI context, just continue
            
            # Also print to console
            print(f"📊 Progress: {percentage:.1f}% ({completed_segments}/{len(segments)}) - {active_threads} threads active")
    
    def gpu_worker(worker_id, model_key):
        """GPU worker for high-throughput transcription."""
        nonlocal completed_segments
        model = models.get(model_key)
        if not model:
            return []
        
        worker_results = []
        processed_count = 0
        
        while not segment_queue.empty():
            try:
                seg_id, segment_info = segment_queue.get_nowait()
                
                result = transcribe_segment_aggressive(model, None, segment_info, f"GPU-{worker_id}")
                result["segment_id"] = seg_id
                worker_results.append(result)
                processed_count += 1
                
                # Update progress
                completed_segments += 1
                if completed_segments % 5 == 0:  # Update every 5 segments
                    send_progress_update()
                
                # Clear GPU cache periodically
                if processed_count % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"\n❌ GPU Worker {worker_id} error: {e}")
        
        return worker_results
    
    def cpu_worker(worker_id, model_key):
        """CPU worker for parallel processing."""
        nonlocal completed_segments
        # Share CPU models across multiple workers
        cpu_models = [k for k in models.keys() if k.startswith('cpu')]
        if not cpu_models:  # No CPU models available
            return []
        
        model_idx = worker_id % len(cpu_models)
        model = models.get(f"cpu_{model_idx}")
        if not model:
            return []
        
        worker_results = []
        processed_count = 0
        
        while not segment_queue.empty():
            try:
                seg_id, segment_info = segment_queue.get_nowait()
                
                result = transcribe_segment_aggressive(model, None, segment_info, f"CPU-{worker_id}")
                result["segment_id"] = seg_id
                worker_results.append(result)
                processed_count += 1
                
                # Update progress
                completed_segments += 1
                if completed_segments % 3 == 0:  # Update every 3 segments for CPU
                    send_progress_update()
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"\n❌ CPU Worker {worker_id} error: {e}")
        
        return worker_results
        
        while not segment_queue.empty():
            try:
                seg_id, segment_info = segment_queue.get_nowait()
                
                result = transcribe_segment_aggressive(model, None, segment_info, f"CPU-{worker_id}")
                result["segment_id"] = seg_id
                worker_results.append(result)
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"\n❌ CPU Worker {worker_id} error: {e}")
        
        return worker_results
    
    # Start all workers
    futures = []
    transcription_start = time.time()
    
    with ThreadPoolExecutor(max_workers=config["total_workers"]) as executor:
        # Start GPU workers
        for i, model_key in enumerate([k for k in models.keys() if k.startswith('gpu')]):
            future = executor.submit(gpu_worker, i, model_key)
            futures.append(future)
        
        # Start CPU workers
        cpu_models = [k for k in models.keys() if k.startswith('cpu')]
        for i in range(config["cpu_workers"]):
            if cpu_models:  # Only if CPU models exist
                model_key = f"cpu_{i % len(cpu_models)}"
                future = executor.submit(cpu_worker, i, model_key)
                futures.append(future)
        
        # Collect all results
        all_results = []
        completed_workers = 0
        for future in as_completed(futures):
            try:
                worker_results = future.result()
                if worker_results:
                    all_results.extend(worker_results)
                completed_workers += 1
                print(f"\n✅ Worker {completed_workers}/{len(futures)} completed ({len(worker_results)} segments)")
            except Exception as e:
                print(f"\n❌ Worker completion error: {e}")
    
    transcription_time = time.time() - transcription_start
    
    # Sort results by segment ID
    all_results.sort(key=lambda x: x.get('segment_id', 0))
    
    # Send final progress update
    try:
        import sys
        if hasattr(sys.stdout, 'output_queue'):
            elapsed = time.time() - start_time
            print(f"PROGRESS:100.0|Transcription Complete!|0|Total: {elapsed:.0f}s")
    except:
        pass
    
    # Calculate performance stats
    total_audio_duration = sum(r.get('end', 0) - r.get('start', 0) for r in all_results)
    processing_speedup = total_audio_duration / transcription_time if transcription_time > 0 else 0
    
    print(f"\n🎯 Transcription Complete!")
    print(f"   Total segments: {len(all_results)}")
    print(f"   Processing time: {transcription_time:.1f}s")
    print(f"   Audio duration: {total_audio_duration:.1f}s")
    print(f"   Speed: {processing_speedup:.1f}x realtime")
    
    return all_results


def transcribe_file_aggressive(input_path, model_name="medium", output_dir=None, force_aggressive=True):
    """
    Ultra-aggressive transcription using maximum available hardware resources.
    """
    from transcribe import (vad_segment_times, post_process_segments, 
                          preprocess_audio, get_media_duration, 
                          split_into_paragraphs, format_duration)
    
    start_time = time.time()
    temp_files = []
    
    try:
        print(f"🚀 AGGRESSIVE GPU+CPU HYBRID TRANSCRIPTION")
        print(f"📁 Input: {os.path.basename(input_path)}")
        
        # System info
        cpu_cores = multiprocessing.cpu_count()
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "None"
        
        print(f"🖥️  System: {cpu_cores} CPU cores, GPU: {gpu_name}")
        
        # Get optimal configuration
        config = get_optimal_worker_counts()
        
        # Adjust worker counts based on actual model size to prevent RAM exhaustion
        config = adjust_workers_for_model(config, model_name)
        
        if not output_dir:
            output_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="aggressive_transcribe_")
        temp_files.append(temp_dir)
        
        # Get media info
        duration = get_media_duration(input_path)
        if duration:
            print(f"⏱️  Duration: {format_duration(duration)}")
        print()  # Add spacing for readability
        
        # Preprocess audio
        print("🔄 Preprocessing audio...")
        tf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tf.close()
        pre_path = tf.name
        preprocess_audio(input_path, pre_path, bitrate="128k")  # Lower bitrate for speed
        audio_path = pre_path
        temp_files.append(pre_path)
        
        # Load models aggressively
        models = load_models_aggressive(model_name, config)
        if not models:
            raise Exception("Failed to load any models!")
        print()  # Add spacing after model loading
        
        # VAD segmentation with aggressive settings
        print("✂️  Running aggressive VAD segmentation...")
        segments = vad_segment_times(audio_path, aggressiveness=3,  # More aggressive
                                   frame_duration_ms=30, padding_ms=100)  # Less padding for more segments
        
        segments = post_process_segments(segments, min_duration=0.3,  # Shorter minimum duration
                                       merge_gap=0.2, max_segments=200)  # More segments, less merging
        
        print(f"📊 Final segments: {len(segments)}")
        if duration:
            print(f"   Average: {len(segments)/max(duration/60, 1):.1f} segments per minute)")
        
        if len(segments) <= 1:
            # Single segment - use GPU if available
            print("🗣️  Single segment transcription...")
            gpu_model = next((v for k, v in models.items() if k.startswith('gpu')), None)
            model = gpu_model or list(models.values())[0]
            
            result = transcribe_segment_aggressive(model, audio_path, (audio_path, 0, duration or 0), "SINGLE")
            full_text = result.get("text", "")
        else:
            # Extract segments with maximum parallelism
            segment_files = extract_segments_aggressive(audio_path, segments, temp_dir, config)
            
            # Aggressive parallel transcription
            results = transcribe_parallel_aggressive(models, segment_files, config)
            
            # Combine results
            full_text = " ".join([r.get("text", "") for r in results if r.get("text", "").strip()])
        
        if not full_text.strip():
            full_text = "[No speech detected]"
        
        # Post-processing
        print("📝 Post-processing...")
        try:
            punctuation_model = PunctuationModel()
            full_text = punctuation_model.restore_punctuation(full_text)
        except Exception as e:
            print(f"⚠️  Punctuation restoration failed: {e}")
        
        # Format text
        formatted_text = split_into_paragraphs(full_text, max_length=500)
        if isinstance(formatted_text, list):
            formatted_text = '\n\n'.join(formatted_text)
        
        # Save outputs
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}_aggressive.txt")
        docx_path = os.path.join(output_dir, f"{base_name}_aggressive.docx")
        
        # Save files
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        
        # Create Word document
        doc = Document()
        doc.add_heading(f'Aggressive Transcription: {base_name}', 0)
        
        if duration:
            doc.add_paragraph(f'Duration: {format_duration(duration)}')
        
        elapsed = time.time() - start_time
        doc.add_paragraph(f'Processing time: {format_duration(elapsed)}')
        doc.add_paragraph(f'Model: {model_name} (Aggressive GPU+CPU Hybrid)')
        doc.add_paragraph(f'Hardware: {gpu_name} + {cpu_cores} CPU cores')
        
        if duration and elapsed > 0:
            speedup = duration / elapsed
            doc.add_paragraph(f'Processing speed: {speedup:.1f}x realtime')
        
        doc.add_paragraph('')
        
        for para in formatted_text.split('\n\n'):
            if para.strip():
                doc.add_paragraph(para.strip())
        
        doc.save(docx_path)
        
        # Final stats
        elapsed = time.time() - start_time
        print(f"\n🎉 AGGRESSIVE TRANSCRIPTION COMPLETE!")
        print(f"📄 Text file: {txt_path}")
        print(f"📄 Word document: {docx_path}")
        print(f"⏱️  Total time: {format_duration(elapsed)}")
        
        if duration:
            speedup = duration / elapsed
            print(f"⚡ Final speed: {speedup:.1f}x realtime")
        
        return txt_path
        
    except Exception as e:
        print(f"❌ Aggressive transcription failed: {e}")
        raise
        
    finally:
        # Cleanup
        for temp_file in temp_files:
            try:
                if os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
                elif os.path.isfile(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Aggressive GPU+CPU hybrid transcription")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--output-dir", help="Output directory (default: Downloads)")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        transcribe_file_aggressive(
            args.input,
            model_name=args.model,
            output_dir=args.output_dir
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
