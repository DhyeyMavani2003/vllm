#!/usr/bin/env python3
"""
Multi-configuration benchmark for n-gram recency bias.
Tests multiple n-gram ranges and k values.
"""

import json
import time
from typing import List, Tuple
from transformers import AutoTokenizer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


def load_conversations(filename: str, max_conversations: int = 200) -> List[List[int]]:
    """Load and tokenize conversations from JSON file."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    conversations = []
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for item in data[:max_conversations]:
        if 'text' in item:
            text = item['text']
        elif 'chosen' in item:
            text = item['chosen']
        else:
            continue
        
        tokens = tokenizer.encode(text)
        if len(tokens) >= 30:
            conversations.append(tokens)
    
    return conversations


def benchmark_config(conversations: List[List[int]], min_ngram: int, max_ngram: int, k: int, use_recency_bias: bool) -> dict:
    """Benchmark a specific configuration."""
    total_proposals = 0
    total_accepted = 0
    total_tokens_proposed = 0
    total_tokens_accepted = 0
    
    for tokens in conversations:
        proposer = NgramProposer(
            prompt_tokens=tokens,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
            num_spec_tokens=k,
            use_recency_bias=use_recency_bias
        )
        
        # Test at multiple positions
        for pos in range(min_ngram, len(tokens) - k, 10):  # Sample every 10 positions
            proposer.prompt_tokens = tokens[:pos]
            proposal = proposer.propose_tokens()
            
            if proposal:
                total_proposals += 1
                actual = tokens[pos:pos+k]
                
                # Count accepted tokens
                accepted = 0
                for i, token in enumerate(proposal):
                    if i < len(actual) and token == actual[i]:
                        accepted += 1
                    else:
                        break
                
                if accepted == len(proposal):
                    total_accepted += 1
                
                total_tokens_proposed += len(proposal)
                total_tokens_accepted += accepted
    
    acceptance_rate = 100 * total_accepted / total_proposals if total_proposals > 0 else 0
    token_acceptance_rate = 100 * total_tokens_accepted / total_tokens_proposed if total_tokens_proposed > 0 else 0
    
    return {
        'proposals': total_proposals,
        'accepted': total_accepted,
        'acceptance_rate': acceptance_rate,
        'tokens_proposed': total_tokens_proposed,
        'tokens_accepted': total_tokens_accepted,
        'token_acceptance_rate': token_acceptance_rate,
    }


def main():
    print("=" * 80)
    print("N-GRAM RECENCY BIAS - MULTI-CONFIGURATION BENCHMARK")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading conversations...")
    conversations = load_conversations('large_dialogue_data.json', max_conversations=200)
    conversations = [c for c in conversations if 30 <= len(c) <= 1000]
    
    print(f"Loaded {len(conversations)} conversations")
    print(f"Total tokens: {sum(len(c) for c in conversations):,}")
    print()
    
    # Configurations to test
    configs = [
        {"name": "Small n-grams [2,3], k=3", "min": 2, "max": 3, "k": 3},
        {"name": "Medium n-grams [3,5], k=5", "min": 3, "max": 5, "k": 5},
        {"name": "Large n-grams [4,6], k=5", "min": 4, "max": 6, "k": 5},
        {"name": "Medium n-grams [3,5], k=10", "min": 3, "max": 5, "k": 10},
        {"name": "Very large n-grams [5,10], k=5", "min": 5, "max": 10, "k": 5},
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        print(f"  Configuration: min={config['min']}, max={config['max']}, k={config['k']}")
        
        # Test without recency bias
        print("  Running without recency bias...")
        start = time.time()
        result_first = benchmark_config(conversations, config['min'], config['max'], config['k'], False)
        time_first = time.time() - start
        
        # Test with recency bias
        print("  Running with recency bias...")
        start = time.time()
        result_last = benchmark_config(conversations, config['min'], config['max'], config['k'], True)
        time_last = time.time() - start
        
        results.append({
            'config': config,
            'first': result_first,
            'last': result_last,
            'time_first': time_first,
            'time_last': time_last,
        })
        
        print(f"  First match: {result_first['acceptance_rate']:.2f}% ({time_first:.2f}s)")
        print(f"  Last match:  {result_last['acceptance_rate']:.2f}% ({time_last:.2f}s)")
        print(f"  Difference:  {result_last['acceptance_rate'] - result_first['acceptance_rate']:+.2f}pp")
        print()
    
    # Print summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Configuration':<35} | {'First Match':<12} | {'Last Match':<12} | {'Difference':<12}")
    print("-" * 80)
    
    for result in results:
        config_name = result['config']['name']
        first_rate = result['first']['acceptance_rate']
        last_rate = result['last']['acceptance_rate']
        diff = last_rate - first_rate
        
        emoji = "✅" if diff > 1.0 else "⚠️" if diff < -0.5 else ""
        print(f"{config_name:<35} | {first_rate:>10.2f}% | {last_rate:>10.2f}% | {diff:>+9.2f}pp {emoji}")
    
    # Calculate average
    avg_diff = sum(r['last']['acceptance_rate'] - r['first']['acceptance_rate'] for r in results) / len(results)
    print()
    print(f"Average improvement: {avg_diff:+.2f}pp")
    print()
    
    # Print timing summary
    print("=" * 80)
    print("TIMING")
    print("=" * 80)
    print()
    print(f"{'Configuration':<35} | {'First Match':<12} | {'Last Match':<12} | {'Overhead':<12}")
    print("-" * 80)
    
    for result in results:
        config_name = result['config']['name']
        time_first = result['time_first']
        time_last = result['time_last']
        overhead = 100 * (time_last - time_first) / time_first if time_first > 0 else 0
        
        print(f"{config_name:<35} | {time_first:>10.2f}s | {time_last:>10.2f}s | {overhead:>+9.1f}%")
    
    print()


if __name__ == "__main__":
    main()
