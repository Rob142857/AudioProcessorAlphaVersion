"""
Real-time system monitor for transcription performance.
Run this in a separate terminal to watch CPU/GPU usage during transcription.
"""
import psutil
import time
import torch
import os
import subprocess
import threading


def get_gpu_info():
    """Get NVIDIA GPU utilization info."""
    try:
        if torch.cuda.is_available():
            # Try nvidia-ml-py if available
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "gpu_util": utilization.gpu,
                    "memory_util": (memory.used / memory.total) * 100,
                    "memory_used_gb": memory.used / 1024**3,
                    "memory_total_gb": memory.total / 1024**3
                }
            except ImportError:
                # Fallback to nvidia-smi
                try:
                    result = subprocess.run([
                        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"
                    ], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                        return {
                            "gpu_util": int(gpu_util),
                            "memory_util": (int(mem_used) / int(mem_total)) * 100,
                            "memory_used_gb": int(mem_used) / 1024,
                            "memory_total_gb": int(mem_total) / 1024
                        }
                except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                    pass
                
                # Basic PyTorch info
                return {
                    "gpu_util": "N/A",
                    "memory_util": "N/A", 
                    "memory_used_gb": torch.cuda.memory_allocated() / 1024**3,
                    "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
                }
    except Exception:
        pass
    
    return None


def monitor_system(duration_seconds=300, update_interval=1):
    """Monitor system resources for specified duration."""
    print("🖥️  Starting System Resource Monitor")
    print("="*70)
    
    # Get system info
    cpu_count = psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)
    memory_total = psutil.virtual_memory().total / 1024**3
    
    gpu_name = "None"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    print(f"System: {cpu_count} cores ({cpu_count_logical} threads), {memory_total:.1f}GB RAM")
    print(f"GPU: {gpu_name}")
    print("="*70)
    print("Time    | CPU %  | CPU Max | RAM %  | GPU %  | VRAM    | Processes")
    print("-"*70)
    
    start_time = time.time()
    max_cpu = 0
    max_memory = 0
    max_gpu = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                break
            
            # CPU and Memory info
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            max_cpu = max(max_cpu, cpu_percent)
            max_memory = max(max_memory, memory_percent)
            
            # GPU info
            gpu_info = get_gpu_info()
            gpu_str = "N/A    "
            vram_str = "N/A     "
            
            if gpu_info:
                if gpu_info["gpu_util"] != "N/A":
                    gpu_str = f"{gpu_info['gpu_util']:3}%   "
                    max_gpu = max(max_gpu, gpu_info["gpu_util"])
                else:
                    gpu_str = "Active "
                
                vram_str = f"{gpu_info['memory_used_gb']:.1f}GB   "
            
            # Count Python processes (transcription workers)
            python_processes = 0
            for proc in psutil.process_iter(['name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Print status line
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"{mins:2d}:{secs:02d}   | {cpu_percent:5.1f} | {max_cpu:5.1f}  | {memory_percent:5.1f} | {gpu_str} | {vram_str} | {python_processes:3d} Python")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("Monitoring stopped by user")
    
    print("="*70)
    print("SUMMARY:")
    print(f"Maximum CPU usage:    {max_cpu:.1f}%")
    print(f"Maximum Memory usage: {max_memory:.1f}%")
    if max_gpu > 0:
        print(f"Maximum GPU usage:    {max_gpu:.1f}%")
    print(f"Total monitoring time: {elapsed/60:.1f} minutes")
    print("="*70)
    
    # Analysis
    if max_cpu < 30:
        print("⚠️  LOW CPU UTILIZATION - Your CPU is underutilized!")
        print("   Try 'Aggressive' mode for better CPU usage")
    elif max_cpu > 80:
        print("🔥 HIGH CPU UTILIZATION - Maximum performance!")
    
    if gpu_info and max_gpu > 0:
        if max_gpu < 50:
            print("⚠️  LOW GPU UTILIZATION - GPU could work harder!")
        elif max_gpu > 80:
            print("🚀 HIGH GPU UTILIZATION - Excellent GPU usage!")


def quick_check():
    """Quick system check for current utilization."""
    print("📊 Quick System Check (5 seconds)...")
    
    # Baseline measurement
    psutil.cpu_percent(interval=1)
    
    cpu_percent = psutil.cpu_percent(interval=2)
    memory = psutil.virtual_memory()
    
    gpu_info = get_gpu_info()
    
    print(f"CPU Usage:    {cpu_percent:5.1f}% ({psutil.cpu_count()} cores)")
    print(f"Memory Usage: {memory.percent:5.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    
    if gpu_info:
        if gpu_info["gpu_util"] != "N/A":
            print(f"GPU Usage:    {gpu_info['gpu_util']:5}% (CUDA available)")
        else:
            print(f"GPU Usage:    Active    (CUDA available)")
        print(f"VRAM Usage:   {gpu_info['memory_used_gb']:.1f}GB / {gpu_info['memory_total_gb']:.1f}GB")
    else:
        print("GPU Usage:    Not available")
    
    # Count active Python processes
    python_processes = sum(1 for proc in psutil.process_iter(['name']) 
                          if 'python' in proc.info['name'].lower())
    print(f"Python Processes: {python_processes}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system resources during transcription")
    parser.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds (default: 300)")
    parser.add_argument("--quick", action="store_true", help="Quick check instead of continuous monitoring")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_check()
    else:
        print(f"Starting {args.duration} second monitoring session...")
        print("Press Ctrl+C to stop early")
        print()
        monitor_system(args.duration, args.interval)


if __name__ == "__main__":
    main()
