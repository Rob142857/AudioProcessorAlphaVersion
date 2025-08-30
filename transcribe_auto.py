"""
Intelligent auto-mode transcription with dynamic resource utilization.
Automatically detects system capabilities and uses 80-90% of available CPU/GPU resources.
"""
import os
import psutil
import torch
import multiprocessing
import time


def get_system_capabilities():
    """Analyze system hardware and determine optimal processing strategy."""
    cpu_cores = multiprocessing.cpu_count()
    has_cuda = torch.cuda.is_available()
    
    # Get system memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Get GPU memory if available
    gpu_memory_gb = 0
    gpu_name = "None"
    if has_cuda:
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
        except:
            has_cuda = False
    
    # Get current system load to avoid overloading
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    return {
        "cpu_cores": cpu_cores,
        "has_cuda": has_cuda,
        "memory_gb": memory_gb,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_name": gpu_name,
        "current_cpu_load": cpu_percent,
        "current_memory_load": memory_percent
    }


def calculate_optimal_workers(system_info, target_cpu_utilization=85):
    """Calculate optimal worker configuration for 80-90% resource utilization."""
    cpu_cores = system_info["cpu_cores"]
    has_cuda = system_info["has_cuda"]
    current_cpu_load = system_info["current_cpu_load"]
    current_memory_load = system_info["current_memory_load"]
    gpu_memory_gb = system_info["gpu_memory_gb"]
    
    # Calculate available CPU capacity (leave headroom for system)
    available_cpu_capacity = 100 - current_cpu_load
    target_additional_load = min(target_cpu_utilization - current_cpu_load, available_cpu_capacity * 0.9)
    
    # Determine processing strategy based on system capabilities
    if has_cuda and gpu_memory_gb >= 6.0:  # Strong GPU system
        # Use aggressive GPU+CPU hybrid for maximum performance
        strategy = "optimised"
        gpu_workers = min(2, max(1, int(gpu_memory_gb / 4)))  # Conservative GPU memory usage
        cpu_workers = min(cpu_cores - 4, int(cpu_cores * target_additional_load / 100))
        cpu_workers = max(4, cpu_workers)  # Minimum viable CPU workers
        
    elif has_cuda and gpu_memory_gb >= 3.0:  # Moderate GPU system
        # Use balanced hybrid approach
        strategy = "hybrid"
        gpu_workers = 1
        cpu_workers = min(cpu_cores - 2, int(cpu_cores * target_additional_load / 100))
        cpu_workers = max(4, cpu_workers)
        
    elif has_cuda:  # Low-end GPU
        # Use GPU for main processing, minimal CPU assistance
        strategy = "hybrid" 
        gpu_workers = 1
        cpu_workers = min(8, int(cpu_cores * target_additional_load / 100))
        cpu_workers = max(2, cpu_workers)
        
    else:  # CPU-only system
        # Use maximum CPU cores with intelligent threading
        strategy = "cpu_optimized"
        gpu_workers = 0
        cpu_workers = min(cpu_cores - 1, int(cpu_cores * target_additional_load / 100))
        cpu_workers = max(2, cpu_workers)
    
    # Memory considerations - reduce workers if high memory usage
    if current_memory_load > 70:
        cpu_workers = max(2, int(cpu_workers * 0.7))
        if gpu_workers > 1:
            gpu_workers = 1
    
    total_workers = gpu_workers + cpu_workers
    
    return {
        "strategy": strategy,
        "gpu_workers": gpu_workers,
        "cpu_workers": cpu_workers,
        "total_workers": total_workers,
        "target_cpu_utilization": target_cpu_utilization,
        "segment_extraction_workers": min(16, int(cpu_cores * 0.5))
    }


def transcribe_file_auto(input_path, model_name="medium", output_dir=None, target_utilization=85):
    """
    Intelligent auto-transcription with dynamic resource utilization.
    
    Args:
        input_path: Path to audio/video file
        model_name: Whisper model to use
        output_dir: Output directory
        target_utilization: Target CPU utilization percentage (80-95)
    """
    
    print(f"🤖 AUTO-OPTIMIZED TRANSCRIPTION")
    print(f"📁 Input: {os.path.basename(input_path)}")
    
    # Analyze system capabilities
    print("🔍 Analyzing system capabilities...")
    system_info = get_system_capabilities()
    
    print(f"🖥️  System: {system_info['cpu_cores']} CPU cores, {system_info['memory_gb']:.1f}GB RAM")
    print(f"🎮 GPU: {system_info['gpu_name']}")
    if system_info['has_cuda']:
        print(f"💾 GPU Memory: {system_info['gpu_memory_gb']:.1f}GB")
    
    print(f"📊 Current Load: CPU {system_info['current_cpu_load']:.1f}%, RAM {system_info['current_memory_load']:.1f}%")
    
    # Calculate optimal worker configuration
    config = calculate_optimal_workers(system_info, target_utilization)
    
    print(f"🚀 Selected Strategy: {config['strategy'].upper()}")
    print(f"⚙️  Configuration:")
    if config['gpu_workers'] > 0:
        print(f"   GPU Workers: {config['gpu_workers']}")
    print(f"   CPU Workers: {config['cpu_workers']}")
    print(f"   Total Workers: {config['total_workers']}")
    print(f"   Target CPU Usage: {config['target_cpu_utilization']}%")
    
    # Route to appropriate transcription method
    if config['strategy'] == "optimised":
        from transcribe_aggressive import transcribe_file_aggressive
        print("🔥 Using OPTIMISED mode for maximum performance...")
        return transcribe_file_aggressive(input_path, model_name, output_dir, force_aggressive=True)
    
    elif config['strategy'] == "hybrid":
        from transcribe_hybrid import transcribe_file_hybrid
        print("⚡ Using HYBRID mode for balanced GPU+CPU processing...")
        return transcribe_file_hybrid(input_path, 
                                    model_name=model_name, 
                                    output_dir=output_dir,
                                    device_preference="auto",
                                    max_workers=config['total_workers'])
    
    else:  # cpu_optimized
        from transcribe import transcribe_file
        print("🖥️  Using CPU-OPTIMIZED mode...")
        # Use original transcribe with optimized settings
        return transcribe_file(input_path,
                             model_name=model_name,
                             keep_temp=False,
                             device_preference="cpu",
                             output_dir=output_dir,
                             preprocess=True,
                             vad=True,
                             punctuate=True)


def main():
    """Command-line interface for auto transcription."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-optimized transcription")
    parser.add_argument("--input", required=True, help="Input audio/video file")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--output-dir", help="Output directory (default: Downloads)")
    parser.add_argument("--target-cpu", type=int, default=85, 
                       help="Target CPU utilization percentage (80-95)")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        transcribe_file_auto(
            args.input,
            model_name=args.model,
            output_dir=args.output_dir,
            target_utilization=args.target_cpu
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
