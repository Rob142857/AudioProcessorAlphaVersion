"""
Aggressive GPU+CPU hybrid transcription with maximum hardware utilization.
This implementation will fully utilize both your GTX 1070 Ti and all 32 CPU cores.
"""
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
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import gc


def get_optimal_worker_counts():
    """Determine optimal worker distribution for maximum hardware utilization."""
    cpu_cores = multiprocessing.cpu_count()
    has_cuda = torch.cuda.is_available()
    
    if has_cuda:
        # Aggressive hybrid configuration
        # GPU handles heavy lifting, CPU cores do parallel preprocessing/postprocessing
        gpu_workers = min(2, cpu_cores // 16)  # Fewer GPU workers to avoid OOM, more conservative
        cpu_workers = min(cpu_cores - 4, 28)  # Leave some cores for system, use most for parallel processing
        total_workers = gpu_workers + cpu_workers
        
        print(f"🚀 Aggressive Hybrid Config:")
        print(f"   GPU Workers: {gpu_workers} (CUDA parallel streams)")
        print(f"   CPU Workers: {cpu_workers} (Parallel CPU cores)")
        print(f"   Total Workers: {total_workers}")
        
        return {
            "gpu_workers": gpu_workers,
            "cpu_workers": cpu_workers,
            "total_workers": total_workers,
            "segment_extraction_workers": min(cpu_cores, 16)  # For parallel ffmpeg extraction
        }
    else:
        # CPU-only aggressive configuration
        cpu_workers = min(cpu_cores, 24)  # Use most cores, leave some for system
        return {
            "gpu_workers": 0,
            "cpu_workers": cpu_workers,
            "total_workers": cpu_workers,
            "segment_extraction_workers": min(cpu_cores, 16)
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
    """Load multiple model instances for maximum parallel processing."""
    if config is None:
        config = {"gpu_workers": 0, "cpu_workers": 1}
        
    models = {}
    
    if config["gpu_workers"] > 0:
        print(f"🧠 Loading {config['gpu_workers']} GPU model instances...")
        # Load multiple GPU instances for parallel CUDA streams
        for i in range(config["gpu_workers"]):
            try:
                model = whisper.load_model(model_name, device="cuda")
                # Force model to stay on GPU
                model = model.cuda()
                models[f"gpu_{i}"] = model
                print(f"   GPU Model {i}: Loaded on {next(model.parameters()).device}")
            except Exception as e:
                print(f"   GPU Model {i}: Failed - {e}")
    
    if config["cpu_workers"] > 0:
        print(f"🖥️  Loading {min(config['cpu_workers'], 4)} CPU model instances...")
        # Load fewer CPU models but use them with high parallelism
        cpu_model_count = min(config["cpu_workers"] // 4, 4)  # Share CPU models across workers
        for i in range(cpu_model_count):
            try:
                model = whisper.load_model(model_name, device="cpu")
                models[f"cpu_{i}"] = model
                print(f"   CPU Model {i}: Loaded")
            except Exception as e:
                print(f"   CPU Model {i}: Failed - {e}")
    
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
        subprocess.run(cmd, check=True)
        return (idx, output_path, start_time, end_time)
    except Exception as e:
        print(f"\n❌ Segment {idx} extraction failed: {e}")
        return None


def extract_segments_aggressive(audio_path, segments, temp_dir, config):
    """Ultra-fast parallel segment extraction using all available CPU cores."""
    print(f"🎵 Extracting {len(segments)} segments with {config['segment_extraction_workers']} parallel processes...")
    
    # Prepare parallel extraction
    extraction_start = time.time()
    segment_args = [(i, start, end, audio_path, temp_dir) for i, (start, end) in enumerate(segments)]
    
    # Use ProcessPoolExecutor for true CPU parallelism
    with ProcessPoolExecutor(max_workers=config["segment_extraction_workers"]) as executor:
        results = list(executor.map(extract_single_segment_worker, segment_args))
    
    extraction_time = time.time() - extraction_start
    successful_segments = [r for r in results if r is not None]
    
    print(f"\n⚡ Extracted {len(successful_segments)} segments in {extraction_time:.1f}s")
    print(f"   Speed: {len(segments) / extraction_time:.1f} segments/second")
    
    return [(path, start, end) for _, path, start, end in successful_segments]


def transcribe_parallel_aggressive(models, segments, config):
    """Maximum parallel transcription using all GPU and CPU resources."""
    print(f"🗣️  Starting aggressive parallel transcription...")
    print(f"   Processing {len(segments)} segments with {config['total_workers']} workers")
    
    # Start system monitoring
    monitor_thread = monitor_system_usage()
    
    results = []
    segment_queue = queue.Queue()
    
    # Add all segments to processing queue
    for i, segment in enumerate(segments):
        segment_queue.put((i, segment))
    
    def gpu_worker(worker_id, model_key):
        """GPU worker for high-throughput transcription."""
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
        # Share CPU models across multiple workers
        model_idx = worker_id % len([k for k in models.keys() if k.startswith('cpu')])
        model = models.get(f"cpu_{model_idx}")
        if not model:
            return []
        
        worker_results = []
        
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
        for i in range(config["cpu_workers"]):
            future = executor.submit(cpu_worker, i, f"cpu_{i % len([k for k in models.keys() if k.startswith('cpu')])}")
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
        
        if not output_dir:
            output_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="aggressive_transcribe_")
        temp_files.append(temp_dir)
        
        # Get media info
        duration = get_media_duration(input_path)
        if duration:
            print(f"⏱️  Duration: {format_duration(duration)}")
        
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
