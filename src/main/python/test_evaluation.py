#!/usr/bin/env python3
"""
test_evaluation.py

Evaluates all PRF strategies on TEST data using the best parameters found during training.

Process:
1. Reads best parameters from grid_results/training
2. Runs each strategy with those parameters on TEST topics
3. Evaluates results with trec_eval
4. Calculates Robustness Index
5. Performs Wilcoxon signed-rank tests for significance
6. Generates comprehensive report

Usage:
    python test_evaluation.py <dataset>

Example:
    python test_evaluation.py ap8889
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
from datetime import datetime
import glob

# ANSI colors
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}{Colors.NC}\n")


def print_success(msg: str):
    """Print success message"""
    print(f"{Colors.GREEN}{msg}{Colors.NC}")


def print_warning(msg: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}{msg}{Colors.NC}")


def print_error(msg: str):
    """Print error message"""
    print(f"{Colors.RED}{msg}{Colors.NC}")


def load_dataset_config(dataset: str) -> Dict[str, str]:
    """Load dataset configuration from dataset_config.sh"""
    script_dir = Path(__file__).parent.parent / 'scripts'
    config_script = script_dir / 'dataset_config.sh'
    
    if not config_script.exists():
        raise FileNotFoundError(f"Configuration script not found: {config_script}")
    
    # Source the bash script and extract specific variables
    vars_to_extract = [
        'INDEX', 'FOLDER', 'INDEX_PATH', 'CACHE_DIR', 'MU',
        'TOPICS_TEST', 'QRELS_TEST', 'RESULTS_DIR'
    ]
    
    cmd = f'source "{config_script}" && switch_dataset "{dataset}" 2>/dev/null'
    for var in vars_to_extract:
        cmd += f' && echo "{var}=${{{var}:-}}"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to load dataset config: {result.stderr}")
    
    # Parse output
    config = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            config[key] = value
    
    return config


class TestEvaluator:
    """Main class for test evaluation"""
    
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.config = load_dataset_config(dataset)
        
        # Extract configuration
        self.index = self.config.get('INDEX', '')
        self.folder = self.config.get('FOLDER', '')
        self.index_path = self.config.get('INDEX_PATH', '')
        self.cache_dir = self.config.get('CACHE_DIR', '')
        self.mu = self.config.get('MU', '2000')
        
        # Test data paths
        self.topics_test = self.config.get('TOPICS_TEST', '')
        self.qrels_test = self.config.get('QRELS_TEST', '')
        
        if not self.topics_test or not self.qrels_test:
            raise ValueError(f"Test topics and qrels not configured for {self.index}")
        
        self.topics_test_path = Path(self.folder) / 'topics' / self.topics_test
        self.qrels_test_path = Path(self.folder) / 'topics' / self.qrels_test
        
        # Output directories
        self.test_run_folder = Path(self.folder) / 'runs' / f'{self.index}_test'
        self.test_results_dir = Path(self.folder) / 'test_results' / self.index
        self.results_dir = Path(self.folder) / 'grid_results' / self.index
        
        # Create output directories
        self.test_run_folder.mkdir(parents=True, exist_ok=True)
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Python scripts directory
        self.python_script_dir = Path(__file__).parent
        
        # JAR path (from src/main/python -> ../../../target)
        project_root = Path(__file__).parent.parent.resolve()
        self.jar_path = project_root / 'prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar'
        
        if not self.jar_path.exists():
            raise FileNotFoundError(f"JAR file not found: {self.jar_path}\nPlease run 'mvn package' first")
        
        # Data structures
        self.query_types: List[str] = []
        self.strategies = ['PRF', 'MONOT5', 'MONOT5-PROB', 'MONOT5_RERANKER', 'VLLM', 'VLLM-PROB', 'ORACLE', 'ORACLE-K']
        self.best_params: Dict[str, Dict[str, any]] = {}
        self.test_run_files: Dict[str, Path] = {}
        self.test_eval_files: Dict[str, Path] = {}
        self.test_perq_files: Dict[str, Path] = {}
        self.baseline_perq_files: Dict[str, Path] = {}
        self.test_metrics: Dict[str, Dict[str, float]] = {}
        self.wilcoxon_results: Dict[str, List[Dict[str, any]]] = {}  # Store Wilcoxon test results per query type
        self.bh_results: Dict[str, List[Dict[str, any]]] = {}  # Store BH correction results per query type
        
    def print_configuration(self):
        """Print evaluation configuration"""
        print_header(f"TEST EVALUATION - {self.index}")
        print(f"{Colors.GREEN}Configuration:{Colors.NC}")
        print(f"  Training Grid Results: {self.results_dir}")
        print(f"  Test Topics: {self.topics_test_path}")
        print(f"  Test Qrels: {self.qrels_test_path}")
        print(f"  Test Runs Output: {self.test_run_folder}")
        print(f"  Test Results Output: {self.test_results_dir}")
        print()
    
    def discover_query_types(self):
        """Discover available query types from training results"""
        print_header("STEP 0.5: Discover Available Query Types from Training")
        
        for qt in ['title_only', 'title_plus_narrative', 'title_plus_description']:
            qt_file = qt.replace('_', '-')
            pattern = str(self.results_dir / f'summary_*_{qt_file}.tsv')
            
            if glob.glob(pattern):
                self.query_types.append(qt)
                print_success(f"  Found query type: {qt}")
        
        if not self.query_types:
            print_warning("Warning: No query-type-specific files found, assuming title_only")
            self.query_types = ['title_only']
    
    def extract_best_params(self, strategy: str, query_type: str) -> Optional[Dict[str, any]]:
        """Extract best parameters from training results"""
        qt_file = query_type.replace('_', '-')
        
        # Map strategy to summary file
        file_map = {
            'PRF': f'summary_prf_prf_{qt_file}.tsv',
            'MONOT5': f'summary_prf_monot5_{qt_file}.tsv',
            'MONOT5-PROB': f'summary_prf_monot5_prob_{qt_file}.tsv',
            'MONOT5_RERANKER': f'summary_monot5_rerank_{qt_file}.tsv',
            'VLLM': f'summary_prf_vllm_{qt_file}.tsv',
            'VLLM-PROB': f'summary_prf_vllm_prob_{qt_file}.tsv',
            'ORACLE': f'summary_prf_oracle_{qt_file}.tsv',
            'ORACLE-K': f'summary_prf_oracle_k_{qt_file}.tsv',
        }
        
        summary_file = self.results_dir / file_map.get(strategy, '')
        
        if not summary_file.exists():
            return None
        
        try:
            with open(summary_file, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
            
            if not rows:
                return None
            
            # Check if this is a reranker (different format)
            if 'RERANKER' in strategy:
                # Reranker format: run_name depth map P@10 ndcg@100
                # Sort by MAP (column 'map')
                best_row = max(rows, key=lambda r: float(r.get('map', 0)))
                
                return {
                    'depth': best_row.get('depth', 'N/A'),
                    'e': 'N/A',
                    'lambda': 'N/A',
                    'train_map': float(best_row.get('map', 0))
                }
            else:
                # PRF format: run_name depth e lambda map P@10 ndcg@100
                # Sort by MAP
                best_row = max(rows, key=lambda r: float(r.get('map', 0)))
                
                return {
                    'depth': best_row.get('depth', 'N/A'),
                    'e': int(best_row.get('e', 0)),
                    'lambda': float(best_row.get('lambda', 0)),
                    'train_map': float(best_row.get('map', 0))
                }
        except Exception as e:
            print_warning(f"Error reading {summary_file}: {e}")
            return None
    
    def extract_all_best_params(self):
        """Extract best parameters for all strategies and query types"""
        print_header("STEP 1: Extract Best Parameters from Training")
        
        for query_type in self.query_types:
            print(f"{Colors.BLUE}=== Query Type: {query_type} ==={Colors.NC}")
            
            for strategy in self.strategies:
                key = f"{strategy}::{query_type}"
                print(f"{Colors.GREEN}  {strategy}...{Colors.NC}")
                
                params = self.extract_best_params(strategy, query_type)
                
                if params:
                    self.best_params[key] = params
                    
                    if 'RERANKER' in strategy:
                        print(f"    depth={params['depth']} (Training MAP={params['train_map']:.4f})")
                    else:
                        print(f"    depth={params['depth']}, e={params['e']}, λ={params['lambda']:.2f} (Training MAP={params['train_map']:.4f})")
                else:
                    print_warning("    (no training results)")
            
            print()
    
    def run_java_command(self, cmd: str, output_file: Path) -> bool:
        """Run a Java command and capture output"""
        try:
            with open(output_file.with_suffix('.log'), 'w') as log:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            return result.returncode == 0
        except Exception as e:
            print_error(f"    ✗ Failed: {e}")
            return False
    
    def run_baseline(self, query_type: str) -> Tuple[Path, Path]:
        """Run baseline (LM Dirichlet) for a query type"""
        print(f"{Colors.GREEN}Running baseline (LM Dirichlet μ={self.mu})...{Colors.NC}")
        
        baseline_run = self.test_run_folder / f"LMDirichlet-{self.mu}_{query_type}"
        
        cmd = f'''java -cp "{self.jar_path}" org.irlab.prfllm.searcher.TRECSearcherLucene \
            --index_path "{self.index_path}" \
            --topics_path "{self.topics_test_path}" \
            --trec_run_folder "{self.test_run_folder}" \
            --qrels_path "{self.qrels_test_path}" \
            --search_by {query_type} \
            --rerank_method none \
            --prf_strategy none \
            --mu {self.mu}'''
        
        if not self.run_java_command(cmd, baseline_run):
            raise RuntimeError("Baseline run failed")
        
        # Evaluate baseline
        baseline_eval = self.test_results_dir / f"BASELINE_{query_type}_eval.txt"
        baseline_perq = self.test_results_dir / f"BASELINE_{query_type}_perquery.txt"
        
        self.evaluate_run(baseline_run, baseline_eval)
        self.evaluate_per_query(baseline_run, baseline_perq)
        
        baseline_key = f"BASELINE::{query_type}"
        self.test_eval_files[baseline_key] = baseline_eval
        self.test_perq_files[baseline_key] = baseline_perq
        self.baseline_perq_files[query_type] = baseline_perq
        
        print_success("  ✓ Baseline completed")
        return baseline_eval, baseline_perq
    
    def run_test_configuration(self, strategy: str, query_type: str, params: Dict[str, any]) -> Optional[Path]:
        """Run a single PRF configuration on test data"""
        depth = params['depth']
        e = params.get('e', 'N/A')
        lambda_val = params.get('lambda', 'N/A')
        
        # Handle "all" depth for ORACLE strategies
        # Java expects an integer, so we use 1000 to represent "all documents"
        depth_for_java = depth
        if depth == "all":
            depth_for_java = 1000
        
        if 'RERANKER' in strategy:
            print(f"{Colors.GREEN}  → Running {strategy} ({query_type}, depth={depth}){Colors.NC}")
            
            cmd = f'''java -cp "{self.jar_path}" org.irlab.prfllm.searcher.TRECSearcherLucene \
                --index_path "{self.index_path}" \
                --topics_path "{self.topics_test_path}" \
                --trec_run_folder "{self.test_run_folder}" \
                --qrels_path "{self.qrels_test_path}" \
                --search_by {query_type} \
                --rerank_method monot5 \
                --rerank_depth {depth_for_java} \
                --cache_dir "{self.cache_dir}" \
                --mu {self.mu}'''
            
            output_file = self.test_run_folder / f"LMDirichlet-{self.mu}_{query_type}_rerank-monoT5_topK-{depth}"
        else:
            lambda_formatted = f"{lambda_val:.2f}"
            print(f"{Colors.GREEN}  → Running {strategy} ({query_type}, depth={depth}, e={e}, λ={lambda_formatted}){Colors.NC}")
            
            cmd = f'''java -cp "{self.jar_path}" org.irlab.prfllm.searcher.TRECSearcherLucene \
                --index_path "{self.index_path}" \
                --topics_path "{self.topics_test_path}" \
                --trec_run_folder "{self.test_run_folder}" \
                --qrels_path "{self.qrels_test_path}" \
                --search_by {query_type} \
                --mu {self.mu} \
                --rerank_method prf \
                --prf_strategy {strategy} \
                --rf_model RM3 \
                --rerank_depth {depth_for_java} \
                -e {e} \
                --lambda {lambda_val} \
                --cache_dir "{self.cache_dir}" \
                --prf_smoothing Additive \
                --prf_smoothing_param 0.1'''
            
            output_file = self.test_run_folder / f"LMDirichlet-{self.mu}_{query_type}_prf-true_rfStrategy-{strategy}_rfModel-RM3_prfSmoothing-Additive-0.1000_topK-{depth}_lambda-{lambda_formatted}_e-{e}"
        
        if self.run_java_command(cmd, output_file):
            print_success("  ✓ Completed")
            return output_file
        else:
            print_warning("  ⚠ Run failed")
            return None
    
    def evaluate_run(self, run_file: Path, output_file: Path) -> bool:
        """Evaluate a run file with rec_eval"""
        if not run_file.exists():
            print_error(f"Run file not found: {run_file}")
            return False
        
        cmd = f'rec_eval -m map -m P.10 -m ndcg_cut.100 "{self.qrels_test_path}" "{run_file}"'
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                output_file.write_text(result.stdout)
                return True
            else:
                print_error(f"rec_eval failed: {result.stderr}")
                return False
        except Exception as e:
            print_error(f"Evaluation failed: {e}")
            return False
    
    def evaluate_per_query(self, run_file: Path, output_file: Path) -> bool:
        """Evaluate per-query results for Robustness Index"""
        if not run_file.exists():
            print_error(f"Run file not found: {run_file}")
            return False
        
        cmd = f'rec_eval -q -m map "{self.qrels_test_path}" "{run_file}"'
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                output_file.write_text(result.stdout)
                return True
            else:
                print_error(f"rec_eval failed: {result.stderr}")
                return False
        except Exception as e:
            print_error(f"Evaluation failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all test configurations"""
        print_header("STEP 2: Run Configurations on Test Data")
        
        for query_type in self.query_types:
            print(f"{Colors.BLUE}=== Testing with Query Type: {query_type} ==={Colors.NC}\n")
            
            # Run baseline
            self.run_baseline(query_type)
            print()
            
            # Run each strategy
            for strategy in self.strategies:
                key = f"{strategy}::{query_type}"
                
                if key not in self.best_params:
                    continue
                
                params = self.best_params[key]
                print(f"{Colors.GREEN}Testing {strategy}...{Colors.NC}")
                
                run_file = self.run_test_configuration(strategy, query_type, params)
                
                if run_file and run_file.exists():
                    self.test_run_files[key] = run_file
                    
                    # Evaluate
                    eval_file = self.test_results_dir / f"{strategy}_{query_type}_eval.txt"
                    perq_file = self.test_results_dir / f"{strategy}_{query_type}_perquery.txt"
                    
                    self.evaluate_run(run_file, eval_file)
                    self.evaluate_per_query(run_file, perq_file)
                    
                    self.test_eval_files[key] = eval_file
                    self.test_perq_files[key] = perq_file
            
            print()
    
    def extract_metric(self, eval_file: Path, metric: str) -> float:
        """Extract a metric value from rec_eval output"""
        try:
            content = eval_file.read_text()
            for line in content.split('\n'):
                if line.startswith(metric):
                    parts = line.split()
                    if len(parts) >= 3:
                        return float(parts[2])
        except Exception as e:
            print_warning(f"Error extracting {metric} from {eval_file}: {e}")
        return 0.0
    
    def calculate_ri(self, test_file: Path, baseline_file: Path) -> Tuple[float, int, int]:
        """Calculate Robustness Index"""
        script = self.python_script_dir / 'calculate_robustness_index.py'
        
        try:
            result = subprocess.run(
                [sys.executable, str(script), str(test_file), str(baseline_file), '--format', 'csv'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    return float(parts[0]), int(parts[1]), int(parts[2])
        except Exception as e:
            print_warning(f"Error calculating RI: {e}")
        
        return 0.0, 0, 0
    
    def extract_all_metrics(self):
        """Extract metrics from all evaluation files"""
        print_header("STEP 3: Extract Metrics")
        
        for query_type in self.query_types:
            print(f"{Colors.BLUE}=== Query Type: {query_type} ==={Colors.NC}\n")
            
            # Display baseline
            baseline_key = f"BASELINE::{query_type}"
            if baseline_key in self.test_eval_files:
                eval_file = self.test_eval_files[baseline_key]
                
                baseline_map = self.extract_metric(eval_file, 'map')
                baseline_p10 = self.extract_metric(eval_file, 'P_10')
                baseline_ndcg100 = self.extract_metric(eval_file, 'ndcg_cut_100')
                
                print(f"{Colors.GREEN}BASELINE (LM Dirichlet μ={self.mu}):{Colors.NC}")
                print(f"  MAP: {baseline_map:.4f}")
                print(f"  P@10: {baseline_p10:.4f}")
                print(f"  NDCG@100: {baseline_ndcg100:.4f}")
                print()
                
                self.test_metrics[baseline_key] = {
                    'map': baseline_map,
                    'p10': baseline_p10,
                    'ndcg100': baseline_ndcg100
                }
            
            # Display other strategies
            for strategy in self.strategies:
                key = f"{strategy}::{query_type}"
                
                if key not in self.test_eval_files:
                    continue
                
                eval_file = self.test_eval_files[key]
                perq_file = self.test_perq_files[key]
                baseline_perq = self.baseline_perq_files.get(query_type)
                
                # Extract metrics
                map_val = self.extract_metric(eval_file, 'map')
                p10 = self.extract_metric(eval_file, 'P_10')
                ndcg100 = self.extract_metric(eval_file, 'ndcg_cut_100')
                
                # Calculate RI
                ri_val, improved, hurt = 0.0, 0, 0
                if baseline_perq and baseline_perq.exists():
                    ri_val, improved, hurt = self.calculate_ri(perq_file, baseline_perq)
                
                self.test_metrics[key] = {
                    'map': map_val,
                    'p10': p10,
                    'ndcg100': ndcg100,
                    'ri': ri_val,
                    'improved': improved,
                    'hurt': hurt
                }
                
                print(f"{Colors.GREEN}{strategy}:{Colors.NC}")
                print(f"  MAP: {map_val:.4f}")
                print(f"  P@10: {p10:.4f}")
                print(f"  NDCG@100: {ndcg100:.4f}")
                print(f"  RI: {ri_val:.2f} (↑{improved}/↓{hurt})")
            
            print()
    
    def run_wilcoxon_test(self, file1: Path, file2: Path) -> Tuple[float, str]:
        """Run Wilcoxon signed-rank test"""
        script = self.python_script_dir / 'wilcoxon_test.py'
        
        try:
            result = subprocess.run(
                [sys.executable, str(script), str(file1), str(file2), '--format', 'csv'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    p_value = float(parts[1])
                    significant = parts[2]
                    return p_value, significant
        except Exception as e:
            print_warning(f"Error running Wilcoxon test: {e}")
        
        return 1.0, 'error'
    
    def perform_statistical_tests(self):
        """Perform Wilcoxon signed-rank tests"""
        print_header("STEP 4: Statistical Significance Tests (Wilcoxon)")
        
        for query_type in self.query_types:
            print(f"{Colors.BLUE}=== Query Type: {query_type} ==={Colors.NC}\n")
            print("Pairwise Wilcoxon Signed-Rank Tests:")
            print("=" * 70)
            print(f"{'Comparison':<40} {'p-value':<12} {'Significant':<15}")
            print("=" * 70)
            
            baseline_key = f"BASELINE::{query_type}"
            baseline_perq = self.test_perq_files.get(baseline_key)
            
            if not baseline_perq or not baseline_perq.exists():
                print(f"No baseline found for {query_type}, skipping comparisons\n")
                continue
            
            # Initialize storage for this query type
            self.wilcoxon_results[query_type] = []
            
            # Compare each strategy against baseline
            print("# Comparisons: Strategy > Baseline (expecting improvement):")
            for strategy in self.strategies:
                key = f"{strategy}::{query_type}"
                
                if key not in self.test_perq_files:
                    continue
                
                file = self.test_perq_files[key]
                p_value, significant = self.run_wilcoxon_test(file, baseline_perq)
                
                sig_text = "✓ Yes (p<0.05)" if significant == "yes" else "No" if significant == "no" else "Error"
                print(f"{strategy + ' > BASELINE':<40} {p_value:<12.4f} {sig_text:<15}")
                
                # Store result
                self.wilcoxon_results[query_type].append({
                    'comparison': f"{strategy} > BASELINE",
                    'p_value': p_value,
                    'significant': significant
                })
            
            # Strategy-to-strategy comparisons
            print("\n# Strategy-to-Strategy Comparisons:")
            available_strategies = [s for s in self.strategies if f"{s}::{query_type}" in self.test_perq_files]
            
            for i in range(len(available_strategies)):
                for j in range(i + 1, len(available_strategies)):
                    strategy1 = available_strategies[i]
                    strategy2 = available_strategies[j]
                    
                    key1 = f"{strategy1}::{query_type}"
                    key2 = f"{strategy2}::{query_type}"
                    
                    file1 = self.test_perq_files[key1]
                    file2 = self.test_perq_files[key2]
                    
                    p_value, significant = self.run_wilcoxon_test(file1, file2)
                    sig_text = "✓ Yes (p<0.05)" if significant == "yes" else "No" if significant == "no" else "Error"
                    
                    comparison = f"{strategy1} > {strategy2}"
                    print(f"{comparison:<40} {p_value:<12.4f} {sig_text:<15}")
                    
                    # Store result
                    self.wilcoxon_results[query_type].append({
                        'comparison': comparison,
                        'p_value': p_value,
                        'significant': significant
                    })
            
            print("=" * 70)
            print()
    
    def perform_bh_correction(self):
        """Perform Benjamini-Hochberg FDR correction"""
        print_header("STEP 4.5: Multiple Comparisons with BH Correction")
        
        for query_type in self.query_types:
            print(f"{Colors.BLUE}=== Query Type: {query_type} ==={Colors.NC}\n")
            
            # Prepare arguments for multiple comparison script
            mc_args = []
            
            baseline_key = f"BASELINE::{query_type}"
            if baseline_key in self.test_perq_files and self.test_perq_files[baseline_key].exists():
                mc_args.append(f"BASELINE:{self.test_perq_files[baseline_key]}")
            
            for strategy in self.strategies:
                key = f"{strategy}::{query_type}"
                if key in self.test_perq_files and self.test_perq_files[key].exists():
                    mc_args.append(f"{strategy}:{self.test_perq_files[key]}")
            
            if not mc_args:
                print(f"No results for {query_type}, skipping BH correction\n")
                continue
            
            # Run multiple comparisons with BH correction
            script = self.python_script_dir / 'wilcoxon_multiple_comparisons.py'
            
            print("Running Benjamini-Hochberg FDR correction for multiple comparisons...\n")
            
            # Display table
            subprocess.run(
                [sys.executable, str(script)] + mc_args + ['--format', 'table', '--alpha', '0.05'],
                check=False
            )
            
            # Save CSV version
            mc_csv = self.test_results_dir / f"multiple_comparisons_bh_{query_type}.csv"
            with open(mc_csv, 'w') as f:
                subprocess.run(
                    [sys.executable, str(script)] + mc_args + ['--format', 'csv', '--alpha', '0.05'],
                    stdout=f,
                    check=False
                )
            
            # Read and store BH results
            self.bh_results[query_type] = []
            try:
                with open(mc_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.bh_results[query_type].append({
                            'comparison': row.get('comparison', ''),
                            'p_value': float(row.get('p_value', 1.0)) if row.get('p_value', '') not in ['', 'nan'] else float('nan'),
                            'p_adjusted': float(row.get('p_adjusted', 1.0)) if row.get('p_adjusted', '') not in ['', 'nan'] else float('nan'),
                            'significant_uncorrected': row.get('significant_uncorrected', 'no'),
                            'significant_corrected': row.get('significant_corrected', 'no')
                        })
            except Exception as e:
                print_warning(f"Could not read BH results: {e}")
            
            print()
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        print_header("STEP 5: Generate Report")
        
        report_file = self.test_results_dir / 'TEST_EVALUATION_REPORT.md'
        
        with open(report_file, 'w') as f:
            # Header
            f.write("# Test Evaluation Report\n\n")
            f.write(f"**Dataset**: {self.index}  \n")
            f.write(f"**Test Topics**: {self.topics_test}  \n")
            f.write(f"**Test Qrels**: {self.qrels_test}  \n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            f.write("---\n\n")
            f.write("## Overview\n\n")
            f.write("This report presents the evaluation results on **TEST data** using the best parameters identified during training.\n\n")
            f.write("**Note**: Results are organized by query type (e.g., title_only, title_plus_narrative). Each query type is evaluated independently with its own baseline for comparison.\n\n")
            
            # Best parameters table
            f.write("---\n\n")
            f.write("## Best Parameters from Training\n\n")
            f.write("The following parameters were selected based on the highest MAP scores on training data:\n\n")
            f.write("| Strategy | Query Type | Depth | E | Lambda | Training MAP |\n")
            f.write("|----------|------------|-------|---|--------|--------------|\n")
            
            for query_type in self.query_types:
                for strategy in self.strategies:
                    key = f"{strategy}::{query_type}"
                    if key in self.best_params:
                        params = self.best_params[key]
                        if 'RERANKER' in strategy:
                            f.write(f"| {strategy} | {query_type} | {params['depth']} | - | - | {params['train_map']:.4f} |\n")
                        else:
                            f.write(f"| {strategy} | {query_type} | {params['depth']} | {params['e']} | {params['lambda']:.2f} | {params['train_map']:.4f} |\n")
            
            # Test results
            f.write("\n---\n\n")
            f.write("## Test Results\n\n")
            
            for query_type in self.query_types:
                f.write(f"### Query Type: {query_type}\n\n")
                
                # Baseline
                baseline_key = f"BASELINE::{query_type}"
                if baseline_key in self.test_metrics:
                    metrics = self.test_metrics[baseline_key]
                    f.write(f"**Baseline (LM Dirichlet μ={self.mu})**:\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| MAP | {metrics['map']:.4f} |\n")
                    f.write(f"| P@10 | {metrics['p10']:.4f} |\n")
                    f.write(f"| NDCG@100 | {metrics['ndcg100']:.4f} |\n\n")
                
                # PRF strategies
                f.write("**PRF Strategies Performance**:\n\n")
                f.write("| Strategy | MAP | P@10 | NDCG@100 | Robustness Index |\n")
                f.write("|----------|-----|------|----------|------------------|\n")
                
                for strategy in self.strategies:
                    key = f"{strategy}::{query_type}"
                    if key in self.test_metrics:
                        metrics = self.test_metrics[key]
                        ri_str = f"{metrics['ri']:.2f} (↑{metrics['improved']}/↓{metrics['hurt']})"
                        f.write(f"| {strategy} | {metrics['map']:.4f} | {metrics['p10']:.4f} | {metrics['ndcg100']:.4f} | {ri_str} |\n")
                
                f.write("\n")
            
            # Metrics explanation
            f.write("### Metrics Explanation\n\n")
            f.write("- **MAP**: Mean Average Precision\n")
            f.write("- **P@10**: Precision at rank 10\n")
            f.write("- **NDCG@100**: Normalized Discounted Cumulative Gain at rank 100\n")
            f.write("- **Robustness Index (RI)**: (queries improved - queries hurt) / total queries\n")
            f.write("  - ↑ = number of queries improved vs baseline\n")
            f.write("  - ↓ = number of queries hurt vs baseline\n\n")
            
            # Statistical Significance Tests
            f.write("---\n\n")
            f.write("## Statistical Significance Tests\n\n")
            
            for query_type in self.query_types:
                if query_type not in self.wilcoxon_results or not self.wilcoxon_results[query_type]:
                    continue
                
                f.write(f"### Query Type: {query_type}\n\n")
                
                # Wilcoxon tests
                f.write("#### Wilcoxon Signed-Rank Test Results\n\n")
                f.write("| Comparison | p-value | Significant (α=0.05) |\n")
                f.write("|------------|---------|----------------------|\n")
                
                for result in self.wilcoxon_results[query_type]:
                    comparison = result['comparison']
                    p_value = result['p_value']
                    significant = result['significant']
                    
                    # Format p-value
                    if p_value != p_value:  # Check for NaN
                        p_val_str = "nan"
                    else:
                        p_val_str = f"{p_value:.6f}"
                    
                    # Format significance
                    if significant == "yes":
                        sig_str = "✓ Yes"
                    elif significant == "no":
                        sig_str = "No"
                    else:
                        sig_str = "Error"
                    
                    f.write(f"| {comparison} | {p_val_str} | {sig_str} |\n")
                
                f.write("\n")
                f.write("**Interpretation**: p-value < 0.05 indicates statistically significant difference.\n\n")
                
                # BH correction results
                if query_type in self.bh_results and self.bh_results[query_type]:
                    f.write("#### Multiple Comparisons with Benjamini-Hochberg Correction\n\n")
                    f.write("When performing multiple hypothesis tests, the probability of finding at least one significant result by chance increases. ")
                    f.write("The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) to account for multiple comparisons.\n\n")
                    
                    f.write("| Comparison | p-value | p-adjusted (BH) | Uncorrected | BH Corrected |\n")
                    f.write("|------------|---------|-----------------|-------------|--------------|\n")
                    
                    for result in self.bh_results[query_type]:
                        comparison = result['comparison']
                        p_value = result['p_value']
                        p_adjusted = result['p_adjusted']
                        sig_uncorrected = result['significant_uncorrected']
                        sig_corrected = result['significant_corrected']
                        
                        # Format p-values
                        if p_value != p_value:  # Check for NaN
                            p_val_str = "nan"
                        else:
                            p_val_str = f"{p_value:.6f}"
                        
                        if p_adjusted != p_adjusted:  # Check for NaN
                            p_adj_str = "nan"
                        else:
                            p_adj_str = f"{p_adjusted:.6f}"
                        
                        # Format significance
                        uncorr_str = "✓ Yes" if sig_uncorrected == "yes" else "No"
                        corr_str = "✓ Yes" if sig_corrected == "yes" else "No"
                        
                        f.write(f"| {comparison} | {p_val_str} | {p_adj_str} | {uncorr_str} | {corr_str} |\n")
                    
                    f.write("\n")
                    f.write("**Interpretation**:\n")
                    f.write("- **p-value**: Raw p-value from Wilcoxon signed-rank test\n")
                    f.write("- **p-adjusted (BH)**: p-value adjusted using Benjamini-Hochberg FDR correction\n")
                    f.write("- **Uncorrected**: Significant at α=0.05 without correction\n")
                    f.write("- **BH Corrected**: Significant at α=0.05 after FDR correction\n\n")
                    f.write("The BH correction is more conservative and reduces false positives when making multiple comparisons. ")
                    f.write("Results that remain significant after correction provide stronger evidence of true differences.\n\n")
            
            # Footer
            f.write("---\n\n")
            f.write("## Files Generated\n\n")
            f.write(f"- **Test Runs**: `{self.test_run_folder}/`\n")
            f.write(f"- **Evaluation Results**: `{self.test_results_dir}/`\n")
            f.write(f"- **Report**: `{report_file}`\n\n")
            
            f.write("---\n\n")
            f.write("## Conclusion\n\n")
            f.write("This evaluation provides insights into the generalization performance of different PRF strategies on unseen test data.\n")
        
        print_success(f"✓ Report generated: {report_file}")
        
        # Try to generate HTML version
        html_report = self.test_results_dir / 'TEST_EVALUATION_REPORT.html'
        try:
            # Create custom CSS for better table formatting
            custom_css = """
            <style>
                body { 
                    max-width: 1200px; 
                    margin: 40px auto; 
                    padding: 20px; 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    line-height: 1.6;
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                }
                th { 
                    background-color: #2c3e50; 
                    color: white; 
                    padding: 12px 15px;
                    text-align: left;
                    font-weight: 600;
                }
                td { 
                    padding: 10px 15px; 
                    border-bottom: 1px solid #ddd;
                }
                tr:hover { 
                    background-color: #f5f5f5;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                h1 { 
                    color: #2c3e50; 
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 { 
                    color: #34495e; 
                    margin-top: 40px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 8px;
                }
                h3 { 
                    color: #34495e; 
                    margin-top: 30px;
                }
                h4 { 
                    color: #7f8c8d; 
                }
                code { 
                    background-color: #f4f4f4; 
                    padding: 2px 6px; 
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }
                hr {
                    border: none;
                    border-top: 2px solid #ecf0f1;
                    margin: 40px 0;
                }
                strong {
                    color: #2c3e50;
                }
            </style>
            """
            
            subprocess.run(
                ['pandoc', str(report_file), '-o', str(html_report),
                 '--standalone',
                 '--metadata', f'title=Test Evaluation Report - {self.index}',
                 '--include-in-header', '/dev/stdin'],
                input=custom_css,
                text=True,
                capture_output=True,
                check=False
            )
            
            if html_report.exists():
                print_success(f"✓ HTML report generated: {html_report}")
        except Exception as e:
            # If fancy version fails, try basic version
            try:
                subprocess.run(
                    ['pandoc', str(report_file), '-o', str(html_report),
                     '--standalone',
                     '--metadata', f'title=Test Evaluation Report - {self.index}'],
                    capture_output=True,
                    check=False
                )
                if html_report.exists():
                    print_success(f"✓ HTML report generated: {html_report}")
            except:
                pass  # Pandoc not available, skip HTML generation
    
    def run(self):
        """Run the complete test evaluation pipeline"""
        try:
            self.print_configuration()
            self.discover_query_types()
            self.extract_all_best_params()
            self.run_all_tests()
            self.extract_all_metrics()
            self.perform_statistical_tests()
            self.perform_bh_correction()
            self.generate_report()
            
            print_header("SUMMARY")
            print_success("Test evaluation completed successfully!\n")
            print(f"Reports generated:")
            print(f"  - Markdown: {self.test_results_dir / 'TEST_EVALUATION_REPORT.md'}")
            
            html_report = self.test_results_dir / 'TEST_EVALUATION_REPORT.html'
            if html_report.exists():
                print(f"  - HTML: {html_report}")
            
            print(f"\nTest runs saved to: {self.test_run_folder}")
            print(f"Evaluation files saved to: {self.test_results_dir}\n")
            
            return 0
        
        except Exception as e:
            print_error(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PRF strategies on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python test_evaluation.py robust04
  python test_evaluation.py ap8889
        '''
    )
    
    parser.add_argument('dataset', help='Dataset name (e.g., robust04, ap8889)')
    
    args = parser.parse_args()
    
    try:
        evaluator = TestEvaluator(args.dataset)
        return evaluator.run()
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
