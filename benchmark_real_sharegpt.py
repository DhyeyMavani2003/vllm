#!/usr/bin/env python3
"""
Benchmark n-gram recency bias on real ShareGPT-like dialogue data.
"""

import json
import time
from typing import List
from transformers import AutoTokenizer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


def load_conversations(filename: str, max_conversations: int = 100) -> List[List[int]]:
    """Load and tokenize conversations from JSON file."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    conversations = []
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for item in data[:max_conversations]:
        # Handle different formats
        if 'text' in item:
            text = item['text']
        elif 'chosen' in item:
            text = item['chosen']
        elif 'conversations' in item:
            # ShareGPT format
            text = "\n".join([f"{msg['from']}: {msg['value']}" for msg in item['conversations']])
        else:
            continue
        
        tokens = tokenizer.encode(text)
        if len(tokens) >= 50:  # Filter very short conversations
            conversations.append(tokens)
    
    return conversations


def benchmark_proposer(conversations: List[List[int]], use_recency_bias: bool, min_ngram: int = 3, max_ngram: int = 5, k: int = 5) -> dict:
    """Benchmark the proposer on conversations."""
    total_proposals = 0
    total_accepted = 0
    total_tokens_proposed = 0
    total_tokens_accepted = 0
    per_conversation_rates = []
    
    for conv_idx, tokens in enumerate(conversations):
        conv_proposals = 0
        conv_accepted = 0
        
        proposer = NgramProposer(
            prompt_tokens=tokens,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
            num_spec_tokens=k,
            use_recency_bias=use_recency_bias
        )
        
        # Test at multiple positions in the conversation
        for pos in range(min_ngram, len(tokens) - k, 5):  # Sample every 5 positions
            proposer.prompt_tokens = tokens[:pos]
            proposal = proposer.propose_tokens()
            
            if proposal:
                conv_proposals += 1
                total_proposals += 1
                actual = tokens[pos:pos+k]
                
                # Count how many tokens are accepted
                accepted = 0
                for i, token in enumerate(proposal):
                    if i < len(actual) and token == actual[i]:
                        accepted += 1
                    else:
                        break
                
                # Full proposal accepted?
                if accepted == len(proposal):
                    conv_accepted += 1
                    total_accepted += 1
                
                total_tokens_proposed += len(proposal)
                total_tokens_accepted += accepted
        
        if conv_proposals > 0:
            conv_rate = 100 * conv_accepted / conv_proposals
            per_conversation_rates.append(conv_rate)
    
    acceptance_rate = 100 * total_accepted / total_proposals if total_proposals > 0 else 0
    token_acceptance_rate = 100 * total_tokens_accepted / total_tokens_proposed if total_tokens_proposed > 0 else 0
    
    return {
        'proposals': total_proposals,
        'accepted': total_accepted,
        'acceptance_rate': acceptance_rate,
        'tokens_proposed': total_tokens_proposed,
        'tokens_accepted': total_tokens_accepted,
        'token_acceptance_rate': token_acceptance_rate,
        'per_conversation_rates': per_conversation_rates,
    }


def main():
    print("=" * 80)
    print("N-GRAM RECENCY BIAS - REAL DATA BENCHMARK")
    print("=" * 80)
    print()
    
    # Load conversations
    print("Loading conversations...")
    conversations = load_conversations('large_dialogue_data.json', max_conversations=100)
    
    print(f"Loaded {len(conversations)} conversations")
    print(f"Total tokens: {sum(len(c) for c in conversations):,}")
    print(f"Average length: {sum(len(c) for c in conversations) / len(conversations):.1f} tokens")
    print()
    
    # Configuration
    min_ngram = 3
    max_ngram = 5
    k = 5
    
    print(f"Configuration: min_ngram={min_ngram}, max_ngram={max_ngram}, k={k}")
    print()
    
    # Benchmark without recency bias
    print("Benchmarking WITHOUT recency bias (first match)...")
    start_time = time.time()
    results_first = benchmark_proposer(conversations, use_recency_bias=False, min_ngram=min_ngram, max_ngram=max_ngram, k=k)
    time_first = time.time() - start_time
    
    print(f"  Proposals: {results_first['proposals']:,}")
    print(f"  Accepted: {results_first['accepted']:,}")
    print(f"  Acceptance rate: {results_first['acceptance_rate']:.2f}%")
    print(f"  Token acceptance rate: {results_first['token_acceptance_rate']:.2f}%")
    print(f"  Time: {time_first:.2f}s")
    print()
    
    # Benchmark with recency bias
    print("Benchmarking WITH recency bias (last match)...")
    start_time = time.time()
    results_last = benchmark_proposer(conversations, use_recency_bias=True, min_ngram=min_ngram, max_ngram=max_ngram, k=k)
    time_last = time.time() - start_time
    
    print(f"  Proposals: {results_last['proposals']:,}")
    print(f"  Accepted: {results_last['accepted']:,}")
    print(f"  Acceptance rate: {results_last['acceptance_rate']:.2f}%")
    print(f"  Token acceptance rate: {results_last['token_acceptance_rate']:.2f}%")
    print(f"  Time: {time_last:.2f}s")
    print()
    
    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    
    diff_acceptance = results_last['acceptance_rate'] - results_first['acceptance_rate']
    diff_token = results_last['token_acceptance_rate'] - results_first['token_acceptance_rate']
    
    print(f"Acceptance rate difference: {diff_acceptance:+.2f}pp")
    print(f"Token acceptance rate difference: {diff_token:+.2f}pp")
    print()
    
    if diff_acceptance > 0:
        print(f"✅ Recency bias improves acceptance rate by {diff_acceptance:.2f}pp")
    elif diff_acceptance < 0:
        print(f"⚠️ Recency bias decreases acceptance rate by {abs(diff_acceptance):.2f}pp")
    else:
        print("No difference in acceptance rate")
    
    print()
    print(f"Performance overhead: {100 * (time_last - time_first) / time_first:+.1f}%")
    print()


if __name__ == "__main__":
    main()
