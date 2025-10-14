#!/usr/bin/env python3
"""
Generate a comprehensive Markdown report from grid search results.
Supports multiple RF strategies: PRF, MONOT5, MONOT5-PROB, OLLAMA, ORACLE, ORACLE-K

Usage:
  python generate_report.py [collection_name]
  
Examples:
  python generate_report.py ap8889_index
  python generate_report.py robust04_index
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

def get_results_dir(collection_name=None):
    """Get results directory based on collection name."""
    base_dir = Path("/home/javier/data/grid_results")
    if collection_name:
        # Collection name already includes _index suffix (e.g., ap8889_index)
        return base_dir / collection_name
    else:
        # Default to ap8889_index for backward compatibility
        return base_dir / "ap8889_index"

def load_per_query_map(results_dir, run_name):
    """Load per-query MAP scores for a run."""
    per_query_file = results_dir / "per_query" / f"perquery_{run_name}.txt"
    if not per_query_file.exists():
        return None
    
    query_scores = {}
    with open(per_query_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == 'map':
                query_id = parts[1]
                score = float(parts[2])
                query_scores[query_id] = score
    
    return query_scores

def calculate_robustness_index(results_dir, run_name, baseline_name):
    """
    Calculate Robustness Index (RI) comparing a run to baseline.
    RI = (num_improved - num_hurt) / total_queries
    
    Returns: (RI, num_improved, num_hurt, total_queries) or None if data not available
    """
    run_scores = load_per_query_map(results_dir, run_name)
    baseline_scores = load_per_query_map(results_dir, baseline_name)
    
    if run_scores is None or baseline_scores is None:
        return None
    
    # Find common queries
    common_queries = set(run_scores.keys()) & set(baseline_scores.keys())
    if len(common_queries) == 0:
        return None
    
    num_improved = 0
    num_hurt = 0
    
    for qid in common_queries:
        diff = run_scores[qid] - baseline_scores[qid]
        if diff > 0:
            num_improved += 1
        elif diff < 0:
            num_hurt += 1
    
    total_queries = len(common_queries)
    ri = (num_improved - num_hurt) / total_queries
    
    return (ri, num_improved, num_hurt, total_queries)

def discover_summary_files(results_dir):
    """Discover all summary*.tsv files and extract strategy names and query types."""
    summary_files = {}
    
    # Find all summary*.tsv files
    for file_path in results_dir.glob("summary*.tsv"):
        filename = file_path.name
        
        # Extract strategy and query type from filename
        # Patterns:
        # - summary_baseline_QUERYTYPE.tsv
        # - summary_monot5_rerank_QUERYTYPE.tsv
        # - summary_prf_STRATEGY_QUERYTYPE.tsv
        
        if filename.startswith("summary_baseline_"):
            # Extract query type from summary_baseline_QUERYTYPE.tsv
            query_type = filename.replace("summary_baseline_", "").replace(".tsv", "")
            key = f'baseline_{query_type}'
            summary_files[key] = file_path
            
        elif filename.startswith("summary_monot5_rerank_"):
            # Extract query type from summary_monot5_rerank_QUERYTYPE.tsv
            query_type = filename.replace("summary_monot5_rerank_", "").replace(".tsv", "")
            key = f'rerank_{query_type}'
            summary_files[key] = file_path
            
        elif filename.startswith("summary_prf_"):
            # Extract strategy and query type from summary_prf_STRATEGY_QUERYTYPE.tsv
            rest = filename.replace("summary_prf_", "").replace(".tsv", "")
            
            # Split by known query types (from right to left)
            query_types = ["title-only", "title-plus-narrative", "title-plus-description"]
            strategy = None
            query_type = None
            
            for qt in query_types:
                if rest.endswith(f"_{qt}"):
                    query_type = qt
                    strategy = rest[:-len(f"_{qt}")]
                    break
            
            if strategy and query_type:
                key = f'prf_{strategy}_{query_type}'
                summary_files[key] = file_path
        
        # Also handle legacy files without query type
        elif filename == "summary_baseline.tsv":
            summary_files['baseline'] = file_path
        elif filename == "summary_monot5_rerank.tsv":
            summary_files['rerank'] = file_path
        elif filename.startswith("summary_prf_") and '_title-' not in filename:
            strategy = filename.replace("summary_prf_", "").replace(".tsv", "")
            summary_files[f'prf_{strategy}'] = file_path
    
    return summary_files

def get_file_paths(results_dir):
    """Get all file paths for a given results directory (DEPRECATED - use discover_summary_files)."""
    # Keep for backward compatibility but mark as deprecated
    summary_files = discover_summary_files(results_dir)
    
    # Add report file path
    summary_files['report_file'] = results_dir / "GRID_SEARCH_REPORT.md"
    
    return summary_files

def load_data(results_dir):
    """Load all summary files automatically."""
    data = {}
    
    # Discover available summary files
    summary_files = discover_summary_files(results_dir)
    
    if not summary_files:
        print("Warning: No summary files found in", results_dir)
        return data
    
    # Load each discovered file
    for key, filepath in summary_files.items():
        try:
            df = pd.read_csv(filepath, sep='\t')
            if len(df) > 0:
                data[key] = df
                print(f"✓ Loaded {len(df)} {key} configurations from {filepath.name}")
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return data
    
    # Load reranker if exists
    if file_paths['summary_rerank'].exists():
        data['rerank'] = pd.read_csv(file_paths['summary_rerank'], sep='\t')
    
    # Load PRF strategies
    prf_files = {
        'prf': file_paths['summary_prf_prf'],
        'monot5': file_paths['summary_prf_monot5'],
        'monot5_prob': file_paths['summary_prf_monot5_prob'],
        'ollama': file_paths['summary_prf_ollama'],
        'vllm': file_paths['summary_prf_vllm'],
        'vllm_prob': file_paths['summary_prf_vllm_prob'],
        'oracle': file_paths['summary_prf_oracle'],
        'oracle_k': file_paths['summary_prf_oracle_k']
    }
    
    for strategy, filepath in prf_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath, sep='\t')
            if len(df) > 0:
                data[f'prf_{strategy}'] = df
    
    return data

def generate_overview(data):
    """Generate overview section."""
    lines = ["## Overview\n"]
    
    if 'baseline' in data:
        lines.append(f"- **Baseline configurations**: {len(data['baseline'])}")
    
    if 'rerank' in data:
        lines.append(f"- **MonoT5 Rerank configurations**: {len(data['rerank'])}")
    
    # Count PRF configurations by strategy
    prf_total = 0
    for key in ['prf_prf', 'prf_monot5', 'prf_monot5_prob', 'prf_ollama', 'prf_vllm', 'prf_vllm_prob', 'prf_oracle', 'prf_oracle_k']:
        if key in data:
            count = len(data[key])
            strategy_name = key.replace('prf_', '').upper().replace('_', '-')
            lines.append(f"- **PRF {strategy_name} configurations**: {count}")
            prf_total += count
    
    total = sum(len(df) for df in data.values())
    lines.append(f"- **Total experiments**: {total}\n")
    
    return lines

def generate_parameter_ranges(data):
    """Generate parameter ranges section."""
    lines = ["## Parameter Ranges\n"]
    
    # For each PRF strategy
    for key in ['prf_prf', 'prf_monot5', 'prf_monot5_prob', 'prf_ollama', 'prf_vllm', 'prf_vllm_prob', 'prf_oracle', 'prf_oracle_k']:
        if key in data:
            df = data[key]
            strategy_name = key.replace('prf_', '').upper().replace('_', '-')
            
            # Convert numpy types to native Python types for clean display
            # Handle 'all' for ORACLE depth
            depths = []
            for x in sorted(df['depth'].unique()):
                if isinstance(x, str) and x == 'all':
                    depths.append('all')
                else:
                    depths.append(int(x))
            
            e_values = [int(x) for x in sorted(df['e'].unique())]
            lambdas = [float(x) for x in sorted(df['lambda'].unique())]
            
            lines.append(f"### PRF {strategy_name}")
            lines.append(f"- **Depths (k)**: {depths}")
            lines.append(f"- **Expansion terms (e)**: {e_values}")
            lines.append(f"- **Lambda (λ)**: {lambdas}\n")
    
    # Reranker
    if 'rerank' in data:
        depths = [int(x) for x in sorted(data['rerank']['depth'].unique())]
        lines.append("### MonoT5 Reranker")
        lines.append(f"- **Depths**: {depths}\n")
    
    return lines

def generate_best_configurations(data, results_dir):
    """Generate best configurations section with Robustness Index."""
    lines = ["## Best Configurations by Strategy\n"]
    
    # Get baseline run name if available
    baseline_name = None
    if 'baseline' in data and len(data['baseline']) > 0:
        baseline_name = data['baseline'].iloc[0]['run_name']
    
    for metric in ['map', 'P@10', 'ndcg@100']:
        lines.append(f"### By {metric.upper()}\n")
        
        # Baseline
        if 'baseline' in data and len(data['baseline']) > 0:
            baseline = data['baseline'].iloc[0]
            lines.append(f"**Baseline (LM Dirichlet):**")
            lines.append(f"- **{metric.upper()}**: {baseline[metric]:.4f}")
            lines.append(f"- MAP: {baseline['map']:.4f}, P@10: {baseline['P@10']:.4f}, ndcg@100: {baseline['ndcg@100']:.4f}\n")
        
        # MonoT5 Reranker
        if 'rerank' in data and len(data['rerank']) > 0:
            best_rerank = data['rerank'].loc[data['rerank'][metric].idxmax()]
            lines.append(f"**Best MonoT5 Rerank:**")
            lines.append(f"- Depth: {best_rerank['depth']:.0f}")
            lines.append(f"- **{metric.upper()}**: {best_rerank[metric]:.4f}")
            lines.append(f"- MAP: {best_rerank['map']:.4f}, P@10: {best_rerank['P@10']:.4f}, ndcg@100: {best_rerank['ndcg@100']:.4f}")
            
            # Add Robustness Index if baseline available
            if baseline_name:
                ri_result = calculate_robustness_index(results_dir, best_rerank['run_name'], baseline_name)
                if ri_result:
                    ri, num_improved, num_hurt, total = ri_result
                    lines.append(f"- **Robustness Index**: {ri:.4f} ({num_improved}↑ / {num_hurt}↓ / {total} queries)\n")
                else:
                    lines.append("")
            else:
                lines.append("")
        
        # PRF strategies
        prf_strategies = [
            ('prf_prf', 'PRF (blind)'),
            ('prf_monot5', 'PRF + MonoT5'),
            ('prf_monot5_prob', 'PRF + MonoT5-PROB'),
            ('prf_ollama', 'PRF + OLLAMA'),
            ('prf_vllm', 'PRF + VLLM'),
            ('prf_vllm_prob', 'PRF + VLLM-PROB'),
            ('prf_oracle_k', 'PRF + ORACLE-K'),
            ('prf_oracle', 'PRF + ORACLE')
        ]
        
        for key, name in prf_strategies:
            if key in data and len(data[key]) > 0:
                best = data[key].loc[data[key][metric].idxmax()]
                lines.append(f"**Best {name}:**")
                
                # Handle depth - could be 'all' for ORACLE
                depth_val = best['depth']
                if isinstance(depth_val, str) and depth_val == 'all':
                    lines.append(f"- Depth (k): all")
                else:
                    lines.append(f"- Depth (k): {depth_val:.0f}")
                
                lines.append(f"- Expansion terms (e): {best['e']:.0f}")
                lines.append(f"- Lambda (λ): {best['lambda']:.2f}")
                lines.append(f"- **{metric.upper()}**: {best[metric]:.4f}")
                lines.append(f"- MAP: {best['map']:.4f}, P@10: {best['P@10']:.4f}, ndcg@100: {best['ndcg@100']:.4f}")
                
                # Add Robustness Index if baseline available
                if baseline_name:
                    ri_result = calculate_robustness_index(results_dir, best['run_name'], baseline_name)
                    if ri_result:
                        ri, num_improved, num_hurt, total = ri_result
                        lines.append(f"- **Robustness Index**: {ri:.4f} ({num_improved}↑ / {num_hurt}↓ / {total} queries)\n")
                    else:
                        lines.append("")
                else:
                    lines.append("")
        
        # Comparison table
        if 'baseline' in data and len(data['baseline']) > 0:
            lines.append(f"**Improvements over Baseline ({metric.upper()}):**\n")
            baseline_val = data['baseline'][metric].iloc[0]
            
            if 'rerank' in data and len(data['rerank']) > 0:
                best_rerank = data['rerank'].loc[data['rerank'][metric].idxmax()]
                improvement = ((best_rerank[metric] - baseline_val) / baseline_val) * 100
                lines.append(f"- MonoT5 Rerank: {improvement:+.2f}%")
            
            for key, name in prf_strategies:
                if key in data and len(data[key]) > 0:
                    best = data[key].loc[data[key][metric].idxmax()]
                    improvement = ((best[metric] - baseline_val) / baseline_val) * 100
                    lines.append(f"- {name}: {improvement:+.2f}%")
            lines.append("")
    
    return lines

def generate_top_configs(data, strategy_key, strategy_name, metric='map', n=10):
    """Generate top-N configurations table for a strategy."""
    lines = []
    
    if strategy_key not in data or len(data[strategy_key]) == 0:
        return lines
    
    df = data[strategy_key]
    lines.append(f"### Top-{n} {strategy_name} Configurations (by {metric.upper()})\n")
    lines.append("| Rank | Depth | E | Lambda | MAP | P@10 | ndcg@100 |")
    lines.append("|------|-------|---|--------|-----|------|---------|")
    
    topN = df.nlargest(n, metric)
    for idx, (i, row) in enumerate(topN.iterrows(), 1):
        # Handle depth - could be 'all' for ORACLE
        depth_val = row['depth']
        if isinstance(depth_val, str) and depth_val == 'all':
            depth_str = 'all'
        else:
            depth_str = f"{depth_val:.0f}"
        
        lines.append(f"| {idx} | {depth_str} | {row['e']:.0f} | "
                     f"{row['lambda']:.2f} | {row['map']:.4f} | "
                     f"{row['P@10']:.4f} | {row['ndcg@100']:.4f} |")
    lines.append("")
    
    return lines

def generate_parameter_analysis(df, strategy_name):
    """Generate parameter analysis for a PRF strategy."""
    lines = [f"### Parameter Analysis: {strategy_name}\n"]
    
    # Lambda analysis
    lines.append("#### Lambda (Interpolation Weight)\n")
    lambda_stats = df.groupby('lambda')[['map', 'P@10', 'ndcg@100']].agg(['mean', 'std', 'max'])
    lines.append("| Lambda | MAP (mean±std) | MAP (max) | P@10 (mean±std) | P@10 (max) |")
    lines.append("|--------|----------------|-----------|-----------------|------------|")
    for lambda_val in sorted(df['lambda'].unique()):
        stats = lambda_stats.loc[lambda_val]
        lines.append(f"| {lambda_val:.1f} | "
                     f"{stats[('map', 'mean')]:.4f}±{stats[('map', 'std')]:.4f} | "
                     f"{stats[('map', 'max')]:.4f} | "
                     f"{stats[('P@10', 'mean')]:.4f}±{stats[('P@10', 'std')]:.4f} | "
                     f"{stats[('P@10', 'max')]:.4f} |")
    lines.append("")
    
    # E analysis
    lines.append("#### Expansion Terms (e)\n")
    e_stats = df.groupby('e')[['map', 'P@10', 'ndcg@100']].agg(['mean', 'std', 'max'])
    lines.append("| E | MAP (mean±std) | MAP (max) | P@10 (mean±std) | P@10 (max) |")
    lines.append("|---|----------------|-----------|-----------------|------------|")
    for e_val in sorted(df['e'].unique()):
        stats = e_stats.loc[e_val]
        lines.append(f"| {e_val:.0f} | "
                     f"{stats[('map', 'mean')]:.4f}±{stats[('map', 'std')]:.4f} | "
                     f"{stats[('map', 'max')]:.4f} | "
                     f"{stats[('P@10', 'mean')]:.4f}±{stats[('P@10', 'std')]:.4f} | "
                     f"{stats[('P@10', 'max')]:.4f} |")
    lines.append("")
    
    # Depth analysis
    lines.append("#### Depth (k)\n")
    depth_stats = df.groupby('depth')[['map', 'P@10', 'ndcg@100']].agg(['mean', 'std', 'max'])
    lines.append("| Depth | MAP (mean±std) | MAP (max) | P@10 (mean±std) | P@10 (max) |")
    lines.append("|-------|----------------|-----------|-----------------|------------|")
    
    # Sort depths, handling 'all' specially
    depth_values = df['depth'].unique()
    sorted_depths = []
    has_all = False
    for d in depth_values:
        if isinstance(d, str) and d == 'all':
            has_all = True
        else:
            sorted_depths.append(d)
    sorted_depths = sorted(sorted_depths)
    if has_all:
        sorted_depths.append('all')
    
    for depth_val in sorted_depths:
        stats = depth_stats.loc[depth_val]
        # Format depth value
        if isinstance(depth_val, str) and depth_val == 'all':
            depth_str = 'all'
        else:
            depth_str = f"{depth_val:.0f}"
        
        lines.append(f"| {depth_str} | "
                     f"{stats[('map', 'mean')]:.4f}±{stats[('map', 'std')]:.4f} | "
                     f"{stats[('map', 'max')]:.4f} | "
                     f"{stats[('P@10', 'mean')]:.4f}±{stats[('P@10', 'std')]:.4f} | "
                     f"{stats[('P@10', 'max')]:.4f} |")
    lines.append("")
    
    # Best average parameters
    best_lambda = df.groupby('lambda')['map'].mean().idxmax()
    best_e = df.groupby('e')['map'].mean().idxmax()
    best_depth = df.groupby('depth')['map'].mean().idxmax()
    
    lines.append("**Key Insights:**")
    lines.append(f"- Best average lambda: {best_lambda:.1f}")
    lines.append(f"- Best average e: {best_e:.0f}")
    
    # Handle depth - could be 'all' for ORACLE
    if isinstance(best_depth, str) and best_depth == 'all':
        lines.append(f"- Best average depth: all\n")
    else:
        lines.append(f"- Best average depth: {best_depth:.0f}\n")
    
    return lines

def get_strategy_display_name(key):
    """Convert strategy key to display name, including query type."""
    # Parse key format: baseline_QUERYTYPE, rerank_QUERYTYPE, or prf_STRATEGY_QUERYTYPE
    
    # Extract query type (last part with hyphens converted to spaces)
    query_type_candidates = ['title-only', 'title-plus-narrative', 'title-plus-description']
    query_type = None
    base_key = key
    
    # Check if key ends with a known query type
    for qt in query_type_candidates:
        if key.endswith(f'_{qt}'):
            query_type = qt.replace('-', ' ').title()
            # Remove query type from base_key
            base_key = key[:-len(f'_{qt}')]
            break
    
    # Handle different base types
    if base_key == 'baseline':
        name = 'Baseline (LM Dirichlet)'
    elif base_key == 'rerank':
        name = 'MonoT5 Rerank'
    elif base_key.startswith('prf_'):
        strategy = base_key.replace('prf_', '').upper().replace('_', '-')
        name = f'PRF + {strategy}'
    else:
        # Fallback for unknown formats
        name = key.upper().replace('_', ' ')
    
    # Add query type if detected
    if query_type:
        name += f' ({query_type})'
    
    return name

def generate_report(data, results_dir, collection_name=None):
    """Generate complete report."""
    report = []
    
    # Extract dataset name from collection_name
    dataset_display = collection_name.replace('_index', '').upper() if collection_name else "Unknown"
    
    # Header
    report.append(f"# Grid Search Results Report - {dataset_display}")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overview
    report.extend(generate_overview(data))
    
    # Parameter ranges
    report.extend(generate_parameter_ranges(data))
    
    # Best configurations
    report.extend(generate_best_configurations(data, results_dir))
    
    # Top-10 configurations for each PRF strategy
    report.append("## Top Configurations by Strategy\n")
    
    # Get all PRF strategies dynamically (excluding baseline and rerank)
    prf_keys = sorted([k for k in data.keys() if k.startswith('prf_')])
    
    for key in prf_keys:
        if key in data and len(data[key]) > 0:
            name = get_strategy_display_name(key)
            report.extend(generate_top_configs(data, key, name))
    
    # Parameter analysis for each strategy
    report.append("## Detailed Parameter Analysis\n")
    
    for key in prf_keys:
        if key in data and len(data[key]) > 0:
            name = get_strategy_display_name(key)
            report.extend(generate_parameter_analysis(data[key], name))
    
    # Overall comparison
    report.append("## Overall Strategy Comparison\n")
    
    # Find baseline with the correct query type for each comparison
    # Group strategies by query type for fair comparison
    query_types = set()
    for key in data.keys():
        for qt in ['title-only', 'title-plus-narrative', 'title-plus-description']:
            if f'_{qt}' in key:
                query_types.add(qt)
    
    # Also check for legacy keys without query type
    has_legacy = any(k in data for k in ['baseline', 'rerank']) or \
                 any(k.startswith('prf_') and not any(f'_{qt}' in k for qt in query_types) 
                     for k in data.keys())
    
    # Generate comparison tables for each query type
    for qt in sorted(query_types):
        qt_display = qt.replace('-', ' ').title()
        report.append(f"### Performance Summary - {qt_display} (MAP)\n")
        
        baseline_key = f'baseline_{qt}'
        if baseline_key in data and len(data[baseline_key]) > 0:
            baseline_map = data[baseline_key]['map'].iloc[0]
            baseline_name = data[baseline_key].iloc[0]['run_name']
            
            report.append("| Strategy | Best MAP | Avg MAP | Improvement | RI (Best) |")
            report.append("|----------|----------|---------|-------------|-----------|")
            report.append(f"| Baseline | {baseline_map:.4f} | {baseline_map:.4f} | - | - |")
            
            # MonoT5 Rerank for this query type
            rerank_key = f'rerank_{qt}'
            if rerank_key in data and len(data[rerank_key]) > 0:
                best_idx = data[rerank_key]['map'].idxmax()
                best = data[rerank_key]['map'].max()
                avg = data[rerank_key]['map'].mean()
                improvement = ((best - baseline_map) / baseline_map) * 100
                
                # Calculate RI for best config
                best_run_name = data[rerank_key].loc[best_idx, 'run_name']
                ri_result = calculate_robustness_index(results_dir, best_run_name, baseline_name)
                ri_str = f"{ri_result[0]:.3f}" if ri_result else "N/A"
                
                name = get_strategy_display_name(rerank_key)
                report.append(f"| {name} | {best:.4f} | {avg:.4f} | {improvement:+.2f}% | {ri_str} |")
            
            # PRF strategies for this query type
            prf_keys = sorted([k for k in data.keys() 
                              if k.startswith('prf_') and k.endswith(f'_{qt}')])
            
            for key in prf_keys:
                if len(data[key]) > 0:
                    best_idx = data[key]['map'].idxmax()
                    best = data[key]['map'].max()
                    avg = data[key]['map'].mean()
                    improvement = ((best - baseline_map) / baseline_map) * 100
                    
                    # Calculate RI for best config
                    best_run_name = data[key].loc[best_idx, 'run_name']
                    ri_result = calculate_robustness_index(results_dir, best_run_name, baseline_name)
                    ri_str = f"{ri_result[0]:.3f}" if ri_result else "N/A"
                    
                    name = get_strategy_display_name(key)
                    report.append(f"| {name} | {best:.4f} | {avg:.4f} | {improvement:+.2f}% | {ri_str} |")
            
            report.append("")
    
    # Also handle legacy data without query type (backward compatibility)
    if has_legacy:
        report.append("### Performance Summary - Legacy (No Query Type) (MAP)\n")
        
        if 'baseline' in data and len(data['baseline']) > 0:
            baseline_map = data['baseline']['map'].iloc[0]
            baseline_name = data['baseline'].iloc[0]['run_name']
            
            report.append("| Strategy | Best MAP | Avg MAP | Improvement | RI (Best) |")
            report.append("|----------|----------|---------|-------------|-----------|")
            report.append(f"| Baseline | {baseline_map:.4f} | {baseline_map:.4f} | - | - |")
            
            if 'rerank' in data and len(data['rerank']) > 0:
                best_idx = data['rerank']['map'].idxmax()
                best = data['rerank']['map'].max()
                avg = data['rerank']['map'].mean()
                improvement = ((best - baseline_map) / baseline_map) * 100
                
                # Calculate RI for best config
                best_run_name = data['rerank'].loc[best_idx, 'run_name']
                ri_result = calculate_robustness_index(results_dir, best_run_name, baseline_name)
                ri_str = f"{ri_result[0]:.3f}" if ri_result else "N/A"
                
                report.append(f"| MonoT5 Rerank | {best:.4f} | {avg:.4f} | {improvement:+.2f}% | {ri_str} |")
            
            # Legacy PRF strategies
            legacy_prf_keys = sorted([k for k in data.keys() 
                                     if k.startswith('prf_') and 
                                     not any(f'_{qt}' in k for qt in query_types)])
            
            for key in legacy_prf_keys:
                if len(data[key]) > 0:
                    best_idx = data[key]['map'].idxmax()
                    best = data[key]['map'].max()
                    avg = data[key]['map'].mean()
                    improvement = ((best - baseline_map) / baseline_map) * 100
                    
                    # Calculate RI for best config
                    best_run_name = data[key].loc[best_idx, 'run_name']
                    ri_result = calculate_robustness_index(results_dir, best_run_name, baseline_name)
                    ri_str = f"{ri_result[0]:.3f}" if ri_result else "N/A"
                    
                    name = get_strategy_display_name(key)
                    report.append(f"| {name} | {best:.4f} | {avg:.4f} | {improvement:+.2f}% | {ri_str} |")
            
            report.append("")
    
    # Visualizations with embedded images
    report.append("## Visualizations\n")
    
    # Check if plots directory exists
    plots_dir = results_dir / "plots"
    has_plots = plots_dir.exists()
    
    if has_plots:
        report.append("### Overall Strategy Comparison\n")
        
        # Overall comparison plots
        for metric in ['map', 'P@10', 'ndcg@100']:
            plot_file = f"comparison_all_strategies_{metric}.png"
            plot_path = plots_dir / plot_file
            if plot_path.exists():
                report.append(f"#### {metric.upper()} Comparison\n")
                report.append(f'<img src="plots/{plot_file}" alt="{metric.upper()} Comparison" width="800">\n')
        
        # Strategy-specific plots
        report.append("### Strategy-Specific Parameter Analysis\n")
        
        # List available strategies from plots directory
        strategy_plots = {}
        if plots_dir.exists():
            for plot_file in plots_dir.glob("lambda_impact_*.png"):
                # Extract strategy from filename: lambda_impact_map_prf.png -> prf
                parts = plot_file.stem.split('_')
                if len(parts) >= 4:
                    metric = parts[2]
                    strategy = '_'.join(parts[3:])
                    if strategy not in strategy_plots:
                        strategy_plots[strategy] = []
                    strategy_plots[strategy].append((metric, 'lambda'))
            
            for plot_file in plots_dir.glob("e_impact_*.png"):
                parts = plot_file.stem.split('_')
                if len(parts) >= 4:
                    metric = parts[2]
                    strategy = '_'.join(parts[3:])
                    if strategy not in strategy_plots:
                        strategy_plots[strategy] = []
                    strategy_plots[strategy].append((metric, 'e'))
            
            for plot_file in plots_dir.glob("depth_impact_*.png"):
                parts = plot_file.stem.split('_')
                if len(parts) >= 4:
                    metric = parts[2]
                    strategy = '_'.join(parts[3:])
                    if strategy not in strategy_plots:
                        strategy_plots[strategy] = []
                    strategy_plots[strategy].append((metric, 'depth'))
            
            for plot_file in plots_dir.glob("heatmap_*.png"):
                parts = plot_file.stem.split('_')
                if len(parts) >= 5:
                    metric = parts[3]
                    strategy = '_'.join(parts[4:])
                    if strategy not in strategy_plots:
                        strategy_plots[strategy] = []
                    strategy_plots[strategy].append((metric, 'heatmap'))
        
        # Display plots organized by strategy
        for strategy in sorted(strategy_plots.keys()):
            strategy_name = strategy.upper().replace('_', '-')
            report.append(f"#### Strategy: {strategy_name}\n")
            
            # Lambda impact
            for metric in ['map', 'P@10', 'ndcg@100']:
                plot_file = f"lambda_impact_{metric}_{strategy}.png"
                if (plots_dir / plot_file).exists():
                    report.append(f"**Lambda Impact on {metric.upper()}**\n")
                    report.append(f'<img src="plots/{plot_file}" alt="Lambda Impact {metric.upper()}" width="800">\n')
            
            # E impact
            for metric in ['map', 'P@10', 'ndcg@100']:
                plot_file = f"e_impact_{metric}_{strategy}.png"
                if (plots_dir / plot_file).exists():
                    report.append(f"**Expansion Terms (e) Impact on {metric.upper()}**\n")
                    report.append(f'<img src="plots/{plot_file}" alt="E Impact {metric.upper()}" width="800">\n')
            
            # Depth impact
            for metric in ['map', 'P@10', 'ndcg@100']:
                plot_file = f"depth_impact_{metric}_{strategy}.png"
                if (plots_dir / plot_file).exists():
                    report.append(f"**Depth Impact on {metric.upper()}**\n")
                    report.append(f'<img src="plots/{plot_file}" alt="Depth Impact {metric.upper()}" width="800">\n')
            
            # Heatmaps
            for metric in ['map', 'P@10', 'ndcg@100']:
                plot_file = f"heatmap_depth_e_{metric}_{strategy}.png"
                if (plots_dir / plot_file).exists():
                    report.append(f"**Depth vs E Heatmap ({metric.upper()})**\n")
                    report.append(f'<img src="plots/{plot_file}" alt="Heatmap {metric.upper()}" width="800">\n')
    else:
        report.append("*Plots will be generated in `grid_results/plots/` after running `visualize_grid_results.py`*\n")
    
    return "\n".join(report)

def main():
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        print("\nAvailable collections:")
        base_dir = Path("/home/javier/data/grid_results")
        if base_dir.exists():
            for collection_dir in base_dir.iterdir():
                if collection_dir.is_dir():
                    print(f"  - {collection_dir.name}")
        return
    
    collection_name = None
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
        print(f"Using collection: {collection_name}")
    else:
        collection_name = "ap8889_index"  # Default
        print(f"No collection specified, using default: {collection_name}")
    
    # Get results directory
    results_dir = get_results_dir(collection_name)
    
    print(f"Looking for results in: {results_dir}")
    print("Discovering and loading data...")
    
    # Load data directly from results_dir
    data = load_data(results_dir)
    
    if len(data) == 0:
        print("Error: No summary files found")
        print("Please run analyze_grid_results.sh first")
        print(f"Expected files in: {results_dir}")
        return
    
    print("Generating Markdown report...")
    report = generate_report(data, results_dir, collection_name)
    
    report_file = results_dir / "GRID_SEARCH_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to: {report_file}")
    print(f"\nView with:")
    print(f"  cat {report_file}")
    print(f"  or open in VS Code for formatted Markdown view")

if __name__ == "__main__":
    main()
