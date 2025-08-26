#!/usr/bin/env python3
"""Test the RAM-aware model loading functionality."""

from transcribe_aggressive import get_optimal_worker_counts, adjust_workers_for_model

def test_ram_awareness():
    """Test RAM-aware worker calculations."""
    print("🧪 Testing RAM-aware worker calculations...")
    
    # Get baseline configuration
    config = get_optimal_worker_counts()
    print(f"\nBaseline configuration: {config['cpu_workers']} CPU workers")
    
    # Test different model sizes
    models = ['tiny', 'base', 'medium', 'large']
    print(f"\nTesting model size adjustments:")
    
    for model_name in models:
        test_config = config.copy()
        adjusted = adjust_workers_for_model(test_config, model_name)
        print(f"  {model_name:8}: {adjusted['cpu_workers']} CPU workers")

if __name__ == "__main__":
    test_ram_awareness()
