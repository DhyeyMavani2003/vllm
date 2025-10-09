#!/usr/bin/env python3
"""
Download real dialogue datasets for benchmarking.
"""

import json
import os
from datasets import load_dataset


def download_hh_rlhf(output_file: str = 'large_dialogue_data.json', num_samples: int = 200):
    """Download Anthropic HH-RLHF dataset."""
    print(f"Downloading Anthropic HH-RLHF dataset ({num_samples} samples)...")
    
    dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(item)
        if (i + 1) % 50 == 0:
            print(f"  Downloaded {i + 1}/{num_samples} samples...")
    
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Saved {len(samples)} samples to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def download_oasst(output_file: str = 'oasst_data.json', num_samples: int = 200):
    """Download OpenAssistant dataset."""
    print(f"Downloading OpenAssistant dataset ({num_samples} samples)...")
    
    dataset = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(item)
        if (i + 1) % 50 == 0:
            print(f"  Downloaded {i + 1}/{num_samples} samples...")
    
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Saved {len(samples)} samples to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def main():
    print("=" * 80)
    print("DATASET DOWNLOADER")
    print("=" * 80)
    print()
    
    # Download HH-RLHF (primary dataset)
    download_hh_rlhf('large_dialogue_data.json', num_samples=200)
    print()
    
    # Download OpenAssistant (alternative dataset)
    download_oasst('oasst_data.json', num_samples=200)
    print()
    
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print()
    print("Available datasets:")
    print("  - large_dialogue_data.json (HH-RLHF)")
    print("  - oasst_data.json (OpenAssistant)")
    print()
    print("Run benchmarks with:")
    print("  python benchmark_real_sharegpt.py")
    print("  python comprehensive_analysis.py")
    print("  python multi_config_benchmark.py")
    print()


if __name__ == "__main__":
    main()
