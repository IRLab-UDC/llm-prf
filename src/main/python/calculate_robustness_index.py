#!/usr/bin/env python3
"""
Calculate Robustness Index (RI) for a PRF strategy.

Robustness Index measures the consistency of a strategy across queries:
    RI = (improved - hurt) / total_queries

Where:
    - improved: number of queries where strategy > baseline
    - hurt: number of queries where strategy < baseline
    - total_queries: total number of queries evaluated

RI ranges from -1 to 1:
    - RI > 0: More queries helped than hurt
    - RI = 0: Equal help and hurt
    - RI < 0: More queries hurt than helped
"""

import sys
import argparse


def calculate_ri(strategy_file, baseline_file):
    """
    Calculate Robustness Index comparing strategy vs baseline.
    
    Args:
        strategy_file: Path to per-query MAP file for the strategy
        baseline_file: Path to per-query MAP file for the baseline
    
    Returns:
        tuple: (ri, improved, hurt, total)
    """
    # Read strategy scores
    strategy_scores = {}
    with open(strategy_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[1]
                score = float(parts[2])
                strategy_scores[query_id] = score
    
    # Read baseline scores
    baseline_scores = {}
    with open(baseline_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[1]
                score = float(parts[2])
                baseline_scores[query_id] = score
    
    # Calculate RI
    improved = 0
    hurt = 0
    equal = 0
    
    for query_id in strategy_scores:
        if query_id not in baseline_scores:
            continue
        
        strategy_score = strategy_scores[query_id]
        baseline_score = baseline_scores[query_id]
        
        if strategy_score > baseline_score:
            improved += 1
        elif strategy_score < baseline_score:
            hurt += 1
        else:
            equal += 1
    
    total = improved + hurt + equal
    
    if total == 0:
        return 0.0, 0, 0, 0
    
    ri = (improved - hurt) / total
    
    return ri, improved, hurt, total


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Robustness Index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'strategy_file',
        help='Path to per-query MAP file for the strategy'
    )
    parser.add_argument(
        'baseline_file',
        help='Path to per-query MAP file for the baseline'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'verbose'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    args = parser.parse_args()
    
    try:
        ri, improved, hurt, total = calculate_ri(args.strategy_file, args.baseline_file)
        
        if args.format == 'csv':
            # CSV format: ri,improved,hurt,total
            print(f"{ri:.4f},{improved},{hurt},{total}")
        else:
            # Verbose format
            print(f"Robustness Index: {ri:.4f}")
            print(f"Improved queries: {improved}")
            print(f"Hurt queries: {hurt}")
            print(f"Total queries: {total}")
            print(f"RI = ({improved} - {hurt}) / {total} = {ri:.4f}")
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
