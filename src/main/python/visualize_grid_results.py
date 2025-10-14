#!/usr/bin/env python3
"""
Grid Search Results Visualization
Generates plots for analyzing the impact of different parameters on retrieval performance.
Automatically detects available RF strategies from summary files.

Usage:
  python visualize_grid_results.py [collection_name]
  
Examples:
  python visualize_grid_results.py ap8889_index
  python visualize_grid_results.py robust04_index
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import re
import subprocess
from pathlib import Path

def get_results_dir(collection_name=None):
    """Get results directory based on collection name or from config."""
    base_dir = Path("/home/javier/data/grid_results")
    
    if collection_name:
        # Use specified collection (already includes _index suffix)
        return base_dir / collection_name
    else:
        # Try to get from shared config file
        try:
            script_dir = Path(__file__).parent.parent / "scripts"
            config_script = script_dir / "dataset_config.sh"
            
            if config_script.exists():
                cmd = f'source "{config_script}" && echo "$RESULTS_DIR"'
                result = subprocess.run(
                    ["bash", "-c", cmd], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                results_dir = Path(result.stdout.strip())
                print(f"Using dataset configuration: {results_dir}")
                return results_dir
        except Exception as e:
            print(f"Warning: Could not read config file: {e}")
        
        # Default fallback
        print("Using default configuration for AP8889")
        return base_dir / "ap8889_index"

def get_dataset_config():
    """Get dataset configuration from shared config file (legacy compatibility)."""
    return get_results_dir()

# Parse command line arguments
collection_name = None
if len(sys.argv) > 1:
    if sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        print("\nAvailable collections:")
        base_dir = Path("/home/javier/data/grid_results")
        if base_dir.exists():
            for collection_dir in base_dir.iterdir():
                if collection_dir.is_dir():
                    print(f"  - {collection_dir.name}")
        sys.exit(0)
    else:
        collection_name = sys.argv[1]
        print(f"Using collection: {collection_name}")

# Configuration
RESULTS_DIR = get_results_dir(collection_name)
PLOTS_DIR = RESULTS_DIR / "plots"

# Create plots directory
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def discover_summary_files():
    """Discover all summary*.tsv files and extract strategy names and query types."""
    summary_files = {}
    
    # Find all summary*.tsv files
    for file_path in RESULTS_DIR.glob("summary*.tsv"):
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
            # The pattern is: summary_prf_{strategy}_{query_type}.tsv
            # where query_type ends the filename and strategy comes before it
            
            # Remove prefix and suffix
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
    
    return summary_files

def load_data():
    """Load all summary files automatically."""
    data = {}
    
    # Discover available summary files
    summary_files = discover_summary_files()
    
    if not summary_files:
        print("Error: No summary files found")
        print("Please run analyze_grid_results.sh first")
        sys.exit(1)
    
    # Load each discovered file
    for key, filepath in summary_files.items():
        try:
            df = pd.read_csv(filepath, sep='\t')
            if len(df) > 0:
                data[key] = df
                print(f"✓ Loaded {len(df)} {key} configurations from {filepath.name}")
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    if len(data) == 0:
        print("Error: No valid summary files could be loaded")
        sys.exit(1)
    
    return data

def format_depth(depth):
    """Format depth value, handling both numeric and 'all' values."""
    if isinstance(depth, str) and depth == 'all':
        return 'all'
    try:
        return str(int(depth))
    except (ValueError, TypeError):
        return str(depth)

def format_e(e):
    """Format e value, handling both numeric and special values."""
    try:
        return str(int(e))
    except (ValueError, TypeError):
        return str(e)

def plot_lambda_impact(df, metric='map', strategy='', dataset=''):
    """Plot impact of lambda on performance."""
    plt.figure(figsize=(14, 8))
    
    for depth in sorted(df['depth'].unique()):
        for e in sorted(df['e'].unique()):
            subset = df[(df['depth'] == depth) & (df['e'] == e)]
            if len(subset) > 0:
                # Sort by lambda to avoid back-and-forth lines
                subset = subset.sort_values('lambda')
                plt.plot(subset['lambda'], subset[metric], 
                        marker='o', alpha=0.6, 
                        label=f'd={format_depth(depth)}, e={format_e(e)}')
    
    plt.xlabel('Lambda (interpolation weight)', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    title = f'Impact of Lambda on {metric.upper()}'
    if strategy:
        title += f' (PRF {strategy})'
    if dataset:
        title += f' - {dataset}'
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'lambda_impact_{metric}_{strategy}.png' if strategy else f'lambda_impact_{metric}.png'
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_e_impact(df, metric='map', strategy='', dataset=''):
    """Plot impact of e (expansion terms) on performance."""
    plt.figure(figsize=(14, 8))
    
    # Group by depth and plot e vs metric
    for depth in sorted(df['depth'].unique()):
        for lambda_val in [0.3, 0.5, 0.7]:  # Selected lambda values
            subset = df[(df['depth'] == depth) & (df['lambda'] == lambda_val)]
            if len(subset) > 0:
                # Sort by e to avoid back-and-forth lines
                subset = subset.sort_values('e')
                plt.plot(subset['e'], subset[metric], 
                        marker='o', alpha=0.6,
                        label=f'd={format_depth(depth)}, λ={lambda_val}')
    
    plt.xlabel('Number of Expansion Terms (e)', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    title = f'Impact of Expansion Terms on {metric.upper()}'
    if strategy:
        title += f' (PRF {strategy})'
    if dataset:
        title += f' - {dataset}'
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'e_impact_{metric}_{strategy}.png' if strategy else f'e_impact_{metric}.png'
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_depth_impact(df, metric='map', strategy='', dataset=''):
    """Plot impact of depth on performance."""
    plt.figure(figsize=(14, 8))
    
    # Best lambda and e for each depth
    best_per_depth = df.loc[df.groupby('depth')[metric].idxmax()]
    # Sort by depth to avoid back-and-forth lines
    best_per_depth = best_per_depth.sort_values('depth')
    
    plt.plot(best_per_depth['depth'], best_per_depth[metric], 
            marker='o', linewidth=2, markersize=10, label='Best config per depth')
    
    # Average performance per depth
    avg_per_depth = df.groupby('depth')[metric].mean()
    # Sort index to ensure proper ordering
    avg_per_depth = avg_per_depth.sort_index()
    plt.plot(avg_per_depth.index, avg_per_depth.values, 
            marker='s', linewidth=2, markersize=8, label='Average per depth', alpha=0.7)
    
    plt.xlabel('Depth (k)', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    title = f'Impact of Depth on {metric.upper()}'
    if strategy:
        title += f' (PRF {strategy})'
    if dataset:
        title += f' - {dataset}'
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'depth_impact_{metric}_{strategy}.png' if strategy else f'depth_impact_{metric}.png'
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap(df, metric='map', strategy='', dataset=''):
    """Plot heatmap for different parameter combinations."""
    # Average over lambda for simplicity
    pivot_data = df.groupby(['depth', 'e'])[metric].mean().reset_index()
    pivot_table = pivot_data.pivot(index='e', columns='depth', values=metric)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': metric.upper()})
    plt.xlabel('Depth (k)', fontsize=12)
    plt.ylabel('Expansion Terms (e)', fontsize=12)
    title = f'{metric.upper()} Heatmap: Depth vs E (averaged over λ)'
    if strategy:
        title += f' (PRF {strategy})'
    if dataset:
        title += f' - {dataset}'
    plt.title(title, fontsize=14)
    plt.tight_layout()
    filename = f'heatmap_depth_e_{metric}_{strategy}.png' if strategy else f'heatmap_depth_e_{metric}.png'
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def get_strategy_display_name(key):
    """Convert strategy key to display name, including query type."""
    # Parse key format: baseline_QUERYTYPE, rerank_QUERYTYPE, or prf_STRATEGY_QUERYTYPE
    
    parts = key.split('_')
    
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
        name += f'\n({query_type})'
    
    return name

def get_strategy_color(key, index):
    """Get color for strategy based on key (extracts base strategy, ignoring query type)."""
    # Extract base strategy key (without query type suffix)
    query_type_candidates = ['title-only', 'title-plus-narrative', 'title-plus-description']
    base_key = key
    
    for qt in query_type_candidates:
        if key.endswith(f'_{qt}'):
            base_key = key[:-len(f'_{qt}')]
            break
    
    # Predefined colors for known base strategies
    color_map = {
        'baseline': '#808080',
        'rerank': '#FF6B6B',
        'prf_prf': '#4ECDC4',
        'prf_monot5': '#45B7D1',
        'prf_monot5_prob': '#5DADE2',
        'prf_ollama': '#AF7AC5',
        'prf_vllm': '#F39C12',
        'prf_vllm_prob': '#E67E22',
        'prf_oracle_k': '#FFA07A',
        'prf_oracle': '#98D8C8'
    }
    
    if base_key in color_map:
        return color_map[base_key]
    else:
        # Generate colors for unknown strategies
        default_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F1C40F', '#9B59B6', 
                         '#E67E22', '#1ABC9C', '#34495E', '#E91E63', '#795548']
        return default_colors[index % len(default_colors)]

def plot_comparison_all_strategies(data, metric='map', dataset=''):
    """Plot comparison of baseline vs best of each strategy."""
    plt.figure(figsize=(14, 8))
    
    strategies = []
    values = []
    colors = []
    labels = []
    
    # Sort strategies: baseline first, then rerank, then PRF strategies
    # Also group by query type
    sorted_keys = []
    
    # Get all query types present in the data
    query_types = set()
    for key in data.keys():
        for qt in ['title-only', 'title-plus-narrative', 'title-plus-description']:
            if f'_{qt}' in key:
                query_types.add(qt)
    
    # Sort query types for consistent ordering
    query_types = sorted(query_types)
    
    # For each query type, add baseline, rerank, then PRF strategies
    for qt in query_types:
        # Add baseline for this query type
        baseline_key = f'baseline_{qt}'
        if baseline_key in data:
            sorted_keys.append(baseline_key)
        
        # Add rerank for this query type
        rerank_key = f'rerank_{qt}'
        if rerank_key in data:
            sorted_keys.append(rerank_key)
        
        # Add PRF strategies for this query type, sorted alphabetically
        prf_keys = [k for k in sorted(data.keys()) 
                   if k.startswith('prf_') and k.endswith(f'_{qt}')]
        sorted_keys.extend(prf_keys)
    
    # Also handle any keys without query type (legacy support)
    if 'baseline' in data and 'baseline' not in sorted_keys:
        sorted_keys.insert(0, 'baseline')
    if 'rerank' in data and 'rerank' not in sorted_keys:
        sorted_keys.insert(1, 'rerank')
    legacy_prf = [k for k in sorted(data.keys()) 
                  if k.startswith('prf_') and k not in sorted_keys]
    sorted_keys.extend(legacy_prf)
    
    for i, key in enumerate(sorted_keys):
        df = data[key]
        if len(df) == 0:
            continue
            
        display_name = get_strategy_display_name(key)
        color = get_strategy_color(key, i)
        
        # Check if this is a baseline (with or without query type)
        is_baseline = key.startswith('baseline')
        # Check if this is rerank (with or without query type)
        is_rerank = key.startswith('rerank')
        
        if is_baseline:
            # Baseline has only one value
            val = df[metric].iloc[0]
            strategies.append(display_name)
            values.append(val)
            colors.append(color)
            labels.append(f'{val:.4f}')
        elif is_rerank:
            # MonoT5 Reranker - best configuration
            best = df.loc[df[metric].idxmax()]
            strategies.append(f'{display_name}\n(depth={format_depth(best["depth"])})')
            values.append(best[metric])
            colors.append(color)
            labels.append(f'{best[metric]:.4f}')
        else:
            # PRF strategies - best configuration
            best = df.loc[df[metric].idxmax()]
            
            strategies.append(f'{display_name}\n(k={format_depth(best["depth"])},\n e={format_e(best["e"])},\n λ={best["lambda"]:.2f})')
            values.append(best[metric])
            colors.append(color)
            labels.append(f'{best[metric]:.4f}')
    
    # Create bar plot
    x_pos = np.arange(len(strategies))
    bars = plt.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, label) in enumerate(zip(bars, labels)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add percentage improvement over corresponding baseline
    # Each system is compared to the baseline with the same query type
    for i, (key, val) in enumerate(zip(sorted_keys, values)):
        # Determine the baseline to compare against
        baseline_key = None
        baseline_val = None
        
        # Extract query type from current key
        for qt in ['title-only', 'title-plus-narrative', 'title-plus-description']:
            if f'_{qt}' in key:
                baseline_key = f'baseline_{qt}'
                break
        
        # If no query type found, try legacy baseline
        if baseline_key is None:
            baseline_key = 'baseline'
        
        # Get baseline value if it exists
        if baseline_key in data and len(data[baseline_key]) > 0:
            baseline_val = data[baseline_key][metric].iloc[0]
        
        # Skip if this IS the baseline or if no baseline found
        if key == baseline_key or baseline_val is None:
            continue
        
        # Calculate and display improvement
        improvement = ((val - baseline_val) / baseline_val) * 100
        plt.text(i, val * 0.5, f'{improvement:+.1f}%',
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.xlabel('Strategy', fontsize=12, fontweight='bold')
    plt.ylabel(metric.upper(), fontsize=12, fontweight='bold')
    title = f'Comparison of Best Configurations by Strategy ({metric.upper()})'
    if dataset:
        title += f' - {dataset}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(x_pos, strategies, fontsize=8)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(PLOTS_DIR / f'comparison_all_strategies_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOTS_DIR / f'comparison_all_strategies_{metric}.png'}")

def print_summary(data):
    """Print statistical summary."""
    print("\n" + "="*60)
    print("GRID SEARCH SUMMARY")
    print("="*60 + "\n")
    
    # Count configurations
    total_configs = 0
    for key, df in data.items():
        if 'prf_' in key or 'rerank' in key:
            total_configs += len(df)
    
    print(f"Total configurations evaluated: {total_configs}")
    print()
    
    for metric in ['map', 'P@10', 'ndcg@100']:
        print(f"\n{metric.upper()} Results:")
        print("-" * 60)
        
        # Get all query types present in the data
        query_types = set()
        for key in data.keys():
            for qt in ['title-only', 'title-plus-narrative', 'title-plus-description']:
                if f'_{qt}' in key:
                    query_types.add(qt)
        
        # Sort query types for consistent ordering
        query_types = sorted(query_types)
        
        # For each query type, show baseline, rerank, then PRF strategies
        for qt in query_types:
            qt_display = qt.replace('-', ' ').title()
            print(f"\n  Query Type: {qt_display}")
            print(f"  {'-' * 55}")
            
            # Baseline for this query type
            baseline_key = f'baseline_{qt}'
            if baseline_key in data and len(data[baseline_key]) > 0:
                df = data[baseline_key]
                display_name = get_strategy_display_name(baseline_key).replace('\n', ' ')
                baseline_val = df[metric].iloc[0]
                print(f"  {display_name}: {baseline_val:.4f}")
            
            # Rerank for this query type
            rerank_key = f'rerank_{qt}'
            if rerank_key in data and len(data[rerank_key]) > 0:
                df = data[rerank_key]
                best = df.loc[df[metric].idxmax()]
                display_name = get_strategy_display_name(rerank_key).replace('\n', ' ')
                print(f"  Best {display_name}: {best[metric]:.4f} (depth={format_depth(best['depth'])})")
            
            # PRF strategies for this query type
            prf_keys = [k for k in sorted(data.keys()) 
                       if k.startswith('prf_') and k.endswith(f'_{qt}')]
            
            for key in prf_keys:
                df = data[key]
                if len(df) == 0:
                    continue
                
                best = df.loc[df[metric].idxmax()]
                display_name = get_strategy_display_name(key).replace('\n', ' ')
                strategy_name = display_name.replace('PRF + ', '')
                print(f"  Best {strategy_name}: {best[metric]:.4f} "
                      f"(k={format_depth(best['depth'])}, e={format_e(best['e'])}, λ={best['lambda']:.2f})")
        
        # Also handle legacy keys without query type
        legacy_keys = [k for k in data.keys() if not any(f'_{qt}' in k for qt in query_types)]
        if legacy_keys:
            print(f"\n  Legacy (no query type specified)")
            print(f"  {'-' * 55}")
            
            for key in sorted(legacy_keys):
                df = data[key]
                if len(df) == 0:
                    continue
                    
                display_name = get_strategy_display_name(key).replace('\n', ' ')
                
                if key == 'baseline':
                    baseline_val = df[metric].iloc[0]
                    print(f"  {display_name}: {baseline_val:.4f}")
                elif key == 'rerank':
                    best = df.loc[df[metric].idxmax()]
                    print(f"  Best {display_name}: {best[metric]:.4f} (depth={format_depth(best['depth'])})")
                elif key.startswith('prf_'):
                    best = df.loc[df[metric].idxmax()]
                    strategy_name = display_name.replace('PRF + ', '')
                    print(f"  Best {strategy_name}: {best[metric]:.4f} "
                          f"(k={format_depth(best['depth'])}, e={format_e(best['e'])}, λ={best['lambda']:.2f})")

def main():
    print("\n" + "="*60)
    print("GRID SEARCH VISUALIZATION")
    print("="*60 + "\n")
    
    # Parse command line arguments
    collection_name = None
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
        print(f"Using collection: {collection_name}")
    else:
        print("No collection specified, using default configuration")
    
    # Extract dataset display name
    dataset_display = collection_name.replace('_index', '').upper() if collection_name else ""
    
    # Load data
    print("Discovering and loading data...")
    data = load_data()
    print()
    
    # Generate plots
    print("\nGenerating plots...\n")
    
    metrics = ['map', 'P@10', 'ndcg@100']
    
    # Generate comparison plot for all strategies
    print("Generating comparison plots:")
    for metric in metrics:
        plot_comparison_all_strategies(data, metric, dataset_display)
    print()
    
    # Generate detailed plots for each PRF strategy
    for key, df in data.items():
        if 'prf_' in key and len(df) > 0:
            strategy_name = key.replace('prf_', '').upper().replace('_', '-')
            print(f"Generating plots for {strategy_name}:")
            for metric in metrics:
                plot_lambda_impact(df, metric, strategy_name, dataset_display)
                plot_e_impact(df, metric, strategy_name, dataset_display)
                plot_depth_impact(df, metric, strategy_name, dataset_display)
                plot_heatmap(df, metric, strategy_name, dataset_display)
            print()
    
    print("="*60)
    print(f"All plots saved in: {PLOTS_DIR}/")
    print("="*60)
    
    # Print summary
    print_summary(data)
    
    print("\n✓ Visualization complete!\n")

if __name__ == "__main__":
    # Check required packages
    missing_packages = []
    
    try:
        import pandas
    except ImportError:
        missing_packages.append('pandas')
    
    try:
        import matplotlib
    except ImportError:
        missing_packages.append('matplotlib')
    
    try:
        import seaborn
    except ImportError:
        missing_packages.append('seaborn')
    
    if missing_packages:
        print(f"Error: Missing required packages: {', '.join(missing_packages)}")
        print("\nPlease install required packages:")
        print(f"  pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    main()
