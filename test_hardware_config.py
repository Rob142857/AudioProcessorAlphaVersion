#!/usr/bin/env python3
"""Test script for hardware configuration."""

from transcribe_aggressive import get_maximum_hardware_config

def main():
    try:
        config = get_maximum_hardware_config()
        print('Hardware configuration test successful!')
        print(f'Devices: {config["devices"]}')
        print(f'Total workers: {config["total_workers"]}')
        print(f'CPU threads: {config["cpu_threads"]}')
        print(f'GPU workers: {config["gpu_workers"]}')

        # Additional details
        print(f'Available RAM: {config["available_ram_gb"]:.1f} GB')
        print(f'CPU cores: {config["cpu_cores"]}')
        print(f'GPU memory: {config.get("gpu_memory_gb", "N/A")} GB')

    except Exception as e:
        print(f'Error testing hardware config: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
