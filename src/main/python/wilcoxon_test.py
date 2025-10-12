#!/usr/bin/env python3
"""
Perform Wilcoxon signed-rank test between two sets of query scores.

The Wilcoxon signed-rank test is a non-parametric statistical test
used to determine if there is a significant difference between two
related samples (paired data).

For IR evaluation, we compare per-query scores from two different
strategies on the same set of queries.

Null hypothesis: No difference between the two strategies
Alternative hypothesis: There is a significant difference

If p-value < α (typically 0.05), we reject the null hypothesis
and conclude there is a statistically significant difference.
"""

import sys
import argparse


def load_scores(filepath):
    """
    Load per-query scores from a trec_eval output file.
    
    Args:
        filepath: Path to per-query evaluation file
    
    Returns:
        dict: {query_id: score}
    """
    scores = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[1]
                score = float(parts[2])
                scores[query_id] = score
    return scores


def wilcoxon_test(file1, file2, alternative='two-sided'):
    """
    Perform Wilcoxon signed-rank test.
    
    Args:
        file1: Path to first per-query file
        file2: Path to second per-query file
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        tuple: (statistic, p_value, n_common)
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("Error: scipy is not installed. Please install it:", file=sys.stderr)
        print("  pip install scipy", file=sys.stderr)
        sys.exit(1)
    
    # Load scores
    scores1 = load_scores(file1)
    scores2 = load_scores(file2)
    
    # Find common queries
    common_queries = sorted(set(scores1.keys()) & set(scores2.keys()))
    
    if len(common_queries) < 2:
        raise ValueError(f"Not enough common queries: {len(common_queries)}")
    
    # Extract paired scores
    paired_scores1 = [scores1[q] for q in common_queries]
    paired_scores2 = [scores2[q] for q in common_queries]
    
    # Perform Wilcoxon test
    try:
        statistic, p_value = wilcoxon(
            paired_scores1,
            paired_scores2,
            alternative=alternative,
            zero_method='wilcox'
        )
    except ValueError as e:
        # Handle case where all differences are zero
        if "zero_method" in str(e) or "all zeros" in str(e).lower():
            return 0.0, 1.0, len(common_queries)
        raise
    
    return statistic, p_value, len(common_queries)


def main():
    parser = argparse.ArgumentParser(
        description='Perform Wilcoxon signed-rank test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'file1',
        help='Path to first per-query evaluation file'
    )
    parser.add_argument(
        'file2',
        help='Path to second per-query evaluation file'
    )
    parser.add_argument(
        '--alternative',
        choices=['two-sided', 'greater', 'less'],
        default='greater',
        help='Alternative hypothesis (default: greater)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'verbose'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    args = parser.parse_args()
    
    try:
        statistic, p_value, n_common = wilcoxon_test(
            args.file1,
            args.file2,
            args.alternative
        )
        
        significant = p_value < args.alpha
        
        if args.format == 'csv':
            # CSV format: statistic,p_value,significant,n_common
            sig_str = "yes" if significant else "no"
            print(f"{statistic:.6f},{p_value:.6f},{sig_str},{n_common}")
        else:
            # Verbose format
            print(f"Wilcoxon Signed-Rank Test Results:")
            print(f"  Statistic: {statistic:.6f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Significant (α={args.alpha}): {'Yes' if significant else 'No'}")
            print(f"  Common queries: {n_common}")
            print(f"  Alternative: {args.alternative}")
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
