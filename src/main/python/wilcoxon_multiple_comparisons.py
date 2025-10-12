#!/usr/bin/env python3
"""
Perform multiple Wilcoxon signed-rank tests with Benjamini-Hochberg correction.

This script performs pairwise Wilcoxon signed-rank tests between multiple runs
and applies the Benjamini-Hochberg False Discovery Rate (FDR) correction to 
control for family-wise error rate when making multiple comparisons.
"""

import argparse
import sys
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def parse_trec_eval_perquery(filepath: str) -> Dict[str, float]:
    """
    Parse per-query evaluation results from trec_eval output.
    
    Args:
        filepath: Path to trec_eval per-query output file
        
    Returns:
        Dictionary mapping query_id -> score
    """
    scores = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == 'map':
                query_id = parts[1]
                if query_id != 'all':  # Skip the aggregate 'all' line
                    score = float(parts[2])
                    scores[query_id] = score
    return scores


def align_scores(scores1: Dict[str, float], scores2: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """
    Align scores from two runs by common query IDs.
    
    Args:
        scores1: First run's scores
        scores2: Second run's scores
        
    Returns:
        Tuple of (aligned_scores1, aligned_scores2)
    """
    common_queries = sorted(set(scores1.keys()) & set(scores2.keys()))
    aligned1 = [scores1[q] for q in common_queries]
    aligned2 = [scores2[q] for q in common_queries]
    return aligned1, aligned2


def perform_comparisons(run_files: Dict[str, str]) -> Tuple[List[str], List[float], List[str]]:
    """
    Perform all pairwise Wilcoxon signed-rank tests.
    
    Args:
        run_files: Dictionary mapping run_name -> filepath
        
    Returns:
        Tuple of (comparison_names, p_values, comparison_labels)
    """
    # Load all scores
    all_scores = {}
    for name, filepath in run_files.items():
        all_scores[name] = parse_trec_eval_perquery(filepath)
    
    # Perform all pairwise comparisons
    comparisons = []
    p_values = []
    labels = []
    
    run_names = list(run_files.keys())
    for i in range(len(run_names)):
        for j in range(i + 1, len(run_names)):
            name1 = run_names[i]
            name2 = run_names[j]
            
            scores1, scores2 = align_scores(all_scores[name1], all_scores[name2])
            
            if len(scores1) < 2:
                print(f"Warning: Not enough common queries for {name1} vs {name2}", file=sys.stderr)
                continue
            
            # Perform Wilcoxon signed-rank test (two-sided)
            try:
                statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
            except ValueError as e:
                print(f"Warning: Wilcoxon test failed for {name1} vs {name2}: {e}", file=sys.stderr)
                continue
            
            comparisons.append(f"{name1} > {name2}")
            p_values.append(p_value)
            labels.append(f"{name1}_{name2}")
    
    return comparisons, p_values, labels


def main():
    parser = argparse.ArgumentParser(
        description='Perform multiple Wilcoxon tests with Benjamini-Hochberg correction'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Per-query evaluation files in format: name1:file1 name2:file2 ...'
    )
    parser.add_argument(
        '--format',
        choices=['table', 'csv'],
        default='table',
        help='Output format (default: table)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for FDR correction (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    # Parse file arguments (format: name:filepath)
    run_files = {}
    for file_arg in args.files:
        if ':' not in file_arg:
            print(f"Error: Invalid format '{file_arg}'. Expected 'name:filepath'", file=sys.stderr)
            sys.exit(1)
        name, filepath = file_arg.split(':', 1)
        run_files[name] = filepath
    
    if len(run_files) < 2:
        print("Error: Need at least 2 runs to compare", file=sys.stderr)
        sys.exit(1)
    
    # Perform all comparisons
    comparisons, p_values, labels = perform_comparisons(run_files)
    
    if len(p_values) == 0:
        print("Error: No valid comparisons could be made", file=sys.stderr)
        sys.exit(1)
    
    # Apply Benjamini-Hochberg FDR correction
    reject, p_values_corrected = fdrcorrection(p_values, alpha=args.alpha, method='indep')
    
    # Output results
    if args.format == 'csv':
        # CSV format: comparison,p_value,p_adjusted,significant_uncorrected,significant_corrected
        print("comparison,p_value,p_adjusted,significant_uncorrected,significant_corrected")
        for i in range(len(comparisons)):
            sig_uncorrected = "yes" if p_values[i] < args.alpha else "no"
            sig_corrected = "yes" if reject[i] else "no"
            print(f"{comparisons[i]},{p_values[i]:.6f},{p_values_corrected[i]:.6f},{sig_uncorrected},{sig_corrected}")
    else:
        # Table format
        print(f"\nMultiple Comparison Results with Benjamini-Hochberg Correction (α={args.alpha})")
        print("=" * 100)
        print(f"{'Comparison':<40} {'p-value':<12} {'p-adjusted':<12} {'Uncorrected':<15} {'BH Corrected':<15}")
        print("=" * 100)
        
        for i in range(len(comparisons)):
            sig_uncorrected = "✓ Significant" if p_values[i] < args.alpha else "Not sig."
            sig_corrected = "✓ Significant" if reject[i] else "Not sig."
            print(f"{comparisons[i]:<40} {p_values[i]:<12.6f} {p_values_corrected[i]:<12.6f} {sig_uncorrected:<15} {sig_corrected:<15}")
        
        print("=" * 100)
        print(f"Total comparisons: {len(comparisons)}")
        print(f"Significant (uncorrected, α={args.alpha}): {sum(1 for p in p_values if p < args.alpha)}")
        print(f"Significant (BH corrected, α={args.alpha}): {sum(reject)}")
        print()


if __name__ == '__main__':
    main()
