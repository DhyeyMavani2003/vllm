# N-gram Recency Bias Benchmark Scripts

## Overview

These scripts benchmark the n-gram recency bias feature on real dialogue datasets.

## Scripts

1. **download_dataset.py** - Download real dialogue datasets
   - Downloads Anthropic HH-RLHF dataset
   - Downloads OpenAssistant dataset
   - Saves to JSON files

2. **benchmark_real_sharegpt.py** - Basic benchmark
   - Tests on 100 conversations
   - Compares first match vs last match
   - Reports acceptance rates

3. **comprehensive_analysis.py** - Detailed analysis
   - Analyzes every position in conversations
   - Finds when first and last matches differ
   - Reports which match is correct more often

4. **multi_config_benchmark.py** - Multi-configuration test
   - Tests multiple n-gram ranges
   - Tests different k values
   - Comprehensive comparison

## Usage

### Step 1: Download datasets

```bash
python download_dataset.py
```

This downloads:
- `large_dialogue_data.json` (HH-RLHF, 200 samples)
- `oasst_data.json` (OpenAssistant, 200 samples)

### Step 2: Run benchmarks

```bash
# Basic benchmark
python benchmark_real_sharegpt.py

# Detailed analysis
python comprehensive_analysis.py

# Multi-configuration test
python multi_config_benchmark.py
```

## Requirements

- Python 3.8+
- transformers
- datasets
- vllm (with n-gram recency bias feature)

## Results

See `DETAILED_REAL_DATA_REPORT.md` for comprehensive results.

Key findings:
- Average improvement: +0.75pp
- Best improvement: +2.12pp with large n-grams [5,10]
- Last match wins 72.6% when predictions differ
- Negligible performance overhead
