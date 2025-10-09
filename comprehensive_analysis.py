#!/usr/bin/env python3
"""
Comprehensive analysis of n-gram recency bias on real dialogue data.
Analyzes every position to find when first and last matches differ.
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
        if len(tokens) >= 30:  # Filter short conversations
            conversations.append(tokens)
    
    return conversations


def analyze_position(tokens: List[int], pos: int, min_ngram: int, max_ngram: int, k: int) -> dict:
    """Analyze a single position to see if first and last matches differ."""
    if pos < min_ngram:
        return None
    
    # Create proposers
    proposer_first = NgramProposer(
        prompt_tokens=tokens[:pos],
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        num_spec_tokens=k,
        use_recency_bias=False
    )
    
    proposer_last = NgramProposer(
        prompt_tokens=tokens[:pos],
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        num_spec_tokens=k,
        use_recency_bias=True
    )
    
    # Get proposals
    first_proposal = proposer_first.propose_tokens()
    last_proposal = proposer_last.propose_tokens()
    
    # Get actual next tokens
    actual = tokens[pos:pos+k]
    
    return {
        'first_proposal': first_proposal,
        'last_proposal': last_proposal,
        'actual': actual,
        'first_correct': first_proposal == actual[:len(first_proposal)] if first_proposal else False,
        'last_correct': last_proposal == actual[:len(last_proposal)] if last_proposal else False,
    }


def main():
    print("=" * 80)
    print("N-GRAM RECENCY BIAS - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading conversations...")
    conversations = load_conversations('large_dialogue_data.json', max_conversations=200)
    
    # Filter to reasonable length
    conversations = [c for c in conversations if 30 <= len(c) <= 1000]
    
    print(f"Loaded {len(conversations)} conversations")
    print(f"Total tokens: {sum(len(c) for c in conversations):,}")
    print()
    
    # Configuration
    min_ngram = 3
    max_ngram = 5
    k = 5
    
    print(f"Configuration: min_ngram={min_ngram}, max_ngram={max_ngram}, k={k}")
    print()
    
    # Analyze every position
    print("Analyzing positions...")
    total_positions = 0
    proposals_made = 0
    first_ne_last = 0
    first_only_correct = 0
    last_only_correct = 0
    both_correct = 0
    neither_correct = 0
    
    start_time = time.time()
    
    for conv_idx, tokens in enumerate(conversations):
        if conv_idx % 20 == 0:
            print(f"  Processing conversation {conv_idx}/{len(conversations)}...")
        
        for pos in range(min_ngram, len(tokens) - k):
            total_positions += 1
            result = analyze_position(tokens, pos, min_ngram, max_ngram, k)
            
            if result is None:
                continue
            
            if result['first_proposal'] or result['last_proposal']:
                proposals_made += 1
            
            # Check if proposals differ
            if result['first_proposal'] != result['last_proposal']:
                first_ne_last += 1
                
                first_correct = result['first_correct']
                last_correct = result['last_correct']
                
                if first_correct and not last_correct:
                    first_only_correct += 1
                elif last_correct and not first_correct:
                    last_only_correct += 1
                elif first_correct and last_correct:
                    both_correct += 1
                else:
                    neither_correct += 1
    
    elapsed = time.time() - start_time
    
    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Total positions tested: {total_positions:,}")
    print(f"Proposals made: {proposals_made:,} ({100*proposals_made/total_positions:.2f}%)")
    print()
    print(f"Cases where first != last match: {first_ne_last:,} ({100*first_ne_last/total_positions:.2f}% of all positions)")
    print()
    print("When they differ:")
    print(f"  First correct only:  {first_only_correct} ({100*first_only_correct/first_ne_last:.2f}%)")
    print(f"  Last correct only:   {last_only_correct} ({100*last_only_correct/first_ne_last:.2f}%)")
    print(f"  Both correct:        {both_correct} ({100*both_correct/first_ne_last:.2f}%)")
    print(f"  Neither correct:     {neither_correct} ({100*neither_correct/first_ne_last:.2f}%)")
    print()
    
    if first_only_correct + last_only_correct > 0:
        print("Winner when one is correct:")
        print(f"  First match wins: {first_only_correct} times")
        print(f"  Last match wins:  {last_only_correct} times")
        win_rate = 100 * last_only_correct / (first_only_correct + last_only_correct)
        print(f"  ✅ Last match advantage: {win_rate:.1f}%")
    
    print()
    print(f"Analysis completed in {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    main()
