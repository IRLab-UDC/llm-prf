#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


def format_depth(depth_val):
    if isinstance(depth_val, str):
        return depth_val.upper()
    else:
        return f"{depth_val:.0f}"


def json_safe(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


class Colors:
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    NC = "\033[0m"


def print_header(title: str):
    print(f"\n{Colors.BLUE}{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}{Colors.NC}\n")


def print_success(msg: str):
    print(f"{Colors.GREEN}{msg}{Colors.NC}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}{msg}{Colors.NC}")


def print_error(msg: str):
    print(f"{Colors.RED}{msg}{Colors.NC}")


def load_dataset_config(dataset: str) -> Dict[str, str]:
    script_dir = Path(__file__).parent.parent / "scripts"
    config_script = script_dir / "dataset_config.sh"

    if not config_script.exists():
        raise FileNotFoundError(f"Configuration script not found: {config_script}")

    vars_to_extract = [
        "INDEX",
        "FOLDER",
        "INDEX_PATH",
        "CACHE_DIR",
        "TOPICS_TEST",
        "QRELS_TEST",
        "RESULTS_DIR",
    ]

    cmd = f'source "{config_script}" && switch_dataset "{dataset}" 2>/dev/null'
    for var in vars_to_extract:
        cmd += f' && echo "{var}=${{{var}:-}}"'

    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to load dataset config: {result.stderr}")

    config = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            config[key] = value

    return config


class TestEvaluator:

    def __init__(self, dataset: str, jar_path: Optional[str] = None):
        self.dataset = dataset
        self.config = load_dataset_config(dataset)

        self.index = self.config.get("INDEX", "")
        self.folder = self.config.get("FOLDER", "")
        self.index_path = self.config.get("INDEX_PATH", "")
        self.cache_dir = self.config.get("CACHE_DIR", "")

        self.topics_test = self.config.get("TOPICS_TEST", "")
        self.qrels_test = self.config.get("QRELS_TEST", "")

        if not self.topics_test or not self.qrels_test:
            raise ValueError(f"Test topics and qrels not configured for {self.index}")

        self.topics_test_path = Path(self.folder) / "topics" / self.topics_test
        self.qrels_test_path = Path(self.folder) / "qrels" / self.qrels_test

        self.test_run_folder = Path(self.folder) / "runs" / f"{self.index}_test"
        self.test_results_dir = Path(self.folder) / "test_results" / self.index
        self.grid_results_dir = Path(self.folder) / "grid_results" / self.index

        self.test_run_folder.mkdir(parents=True, exist_ok=True)
        self.test_results_dir.mkdir(parents=True, exist_ok=True)

        if jar_path:
            self.jar_path = Path(jar_path).resolve()
        elif os.environ.get("JAR_PATH"):
            self.jar_path = Path(os.environ["JAR_PATH"]).resolve()
        else:
            project_root = Path(__file__).parent.parent.parent.parent.resolve()
            candidates = [
                project_root / "target" / "llmprf-1.0-jar-with-dependencies.jar",
                project_root / "llmprf-1.0-jar-with-dependencies.jar",
            ]
            self.jar_path = next((p for p in candidates if p.exists()), candidates[0])

        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"JAR file not found: {self.jar_path}\n"
                f"Provide the path via --jar or the JAR_PATH environment variable"
            )

        self.rf_models = ["RM3", "DMM", "MEDMM"]
        self.baselines = ["LMDirichlet", "BM25"]
        self.best_params: Dict[str, Dict[str, any]] = {}
        self.test_metrics: Dict[str, Dict[str, float]] = {}

    def print_configuration(self):
        print_header(f"TEST EVALUATION - {self.index}")
        print(f"{Colors.GREEN}Configuration:{Colors.NC}")
        print(f"  Training Grid Results: {self.grid_results_dir}")
        print(f"  Test Topics: {self.topics_test_path}")
        print(f"  Test Qrels: {self.qrels_test_path}")
        print(f"  Test Runs Output: {self.test_run_folder}")
        print(f"  Test Results Output: {self.test_results_dir}")
        print(f"  JAR: {self.jar_path}")
        print()

    def load_best_params(self):
        print_header("STEP 1: Load Best Parameters from Training")

        print(f"Looking for results in: {self.grid_results_dir}")

        if not self.grid_results_dir.exists():
            print_error(
                f"Grid results directory does not exist: {self.grid_results_dir}"
            )
            return

        baseline_file = self.grid_results_dir / "baseline.tsv"
        print(f"Checking for baseline file: {baseline_file}")
        if baseline_file.exists():
            print_success(f"Found baseline file: {baseline_file}")
            df = pd.read_csv(baseline_file, sep="\t")
            print(f"Baseline file has {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            for _, row in df.iterrows():
                baseline = row["baseline"]
                self.best_params[f"BASELINE_{baseline}"] = {
                    "baseline": baseline,
                    "map": row["map"],
                }
                print_success(f"  {baseline} baseline: MAP={row['map']:.4f}")
        else:
            print_warning(f"Baseline file not found: {baseline_file}")

        monot5_file = self.grid_results_dir / "monot5.tsv"
        print(f"Checking for MonoT5 file: {monot5_file}")
        if monot5_file.exists():
            print_success(f"Found MonoT5 file: {monot5_file}")
            df = pd.read_csv(monot5_file, sep="\t")
            print(f"MonoT5 file has {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            for baseline in self.baselines:
                baseline_data = df[df["baseline"] == baseline]
                if len(baseline_data) > 0:
                    best_row = baseline_data.loc[baseline_data["map"].idxmax()]
                    self.best_params[f"MONOT5_RERANK_{baseline}"] = {
                        "baseline": best_row["baseline"],
                        "depth": best_row["depth"],
                        "map": best_row["map"],
                    }
                    print_success(
                        f"  MonoT5 Rerank {baseline}: depth={format_depth(best_row['depth'])}, MAP={best_row['map']:.4f}"
                    )
        else:
            print_warning(f"MonoT5 file not found: {monot5_file}")

        prf_file = self.grid_results_dir / "prf.tsv"
        print(f"Checking for PRF file: {prf_file}")
        if prf_file.exists():
            print_success(f"Found PRF file: {prf_file}")
            df = pd.read_csv(prf_file, sep="\t")
            print(f"PRF file has {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            prf_strategies = [
                "PRF", "ORACLE-K", "MONOT5", "MONOT5-PROB", "VLLM", "VLLM-PROB",
                "VLLM-NARR", "VLLM-NARR-PROB", "VLLM-JUDGESPANS", "VLLM-NARR-JUDGESPANS",
            ]
            has_tf = "term_filter" in df.columns
            term_addon_strategies = {
                "PRF", "MONOT5", "VLLM", "VLLM-NARR",
                "MONOT5-PROB", "VLLM-PROB", "VLLM-NARR-PROB",
            }
            PROTOCOL_A_ONLY_STRATEGIES = {"VLLM-JUDGESPANS", "VLLM-NARR-JUDGESPANS"}

            def families_for(strategy):
                if strategy in {"VLLM-JUDGESPANS", "VLLM-NARR-JUDGESPANS"}:
                    return [("JudgeSpans", "VLLMJudgeSpans", "VLLMJUDGESPANS")]
                return [
                    ("SS2", "VLLMSpans2", "VLLMSPANS2"),
                    ("SS2-NoNarr", "VLLMSpans2-NoNarr", "VLLMSPANS2-NONARR"),
                ]

            def train_map_at(baseline, strategy, rf_model, depth, e, lam, tf):
                m = (
                    (df["baseline"] == baseline)
                    & (df["strategy"] == strategy)
                    & (df["rf_model"] == rf_model)
                    & (df["depth"] == depth)
                    & (df["e"] == e)
                )
                if has_tf:
                    m &= (df["term_filter"] == tf)
                if lam is not None and "lambda" in df.columns:
                    m &= (df["lambda"] == lam)
                rows = df[m]
                return float(rows["map"].iloc[0]) if len(rows) > 0 else None

            for baseline in self.baselines:
                for strategy in prf_strategies:
                    for rf_model in self.rf_models:
                        if strategy in PROTOCOL_A_ONLY_STRATEGIES:
                            if not has_tf:
                                continue
                            for label, stored_tf, csv_tf in families_for(strategy):
                                a_mask = (
                                    (df["baseline"] == baseline)
                                    & (df["strategy"] == strategy)
                                    & (df["rf_model"] == rf_model)
                                    & (df["term_filter"] == csv_tf)
                                )
                                a_data = df[a_mask]
                                if len(a_data) == 0:
                                    print_warning(f"  No data for {strategy}+{rf_model}+{baseline} ({csv_tf})")
                                    continue
                                a_best = a_data.loc[a_data["map"].idxmax()]
                                a_lam = a_best.get("lambda", None)
                                a_lam = a_lam if pd.notna(a_lam) else None
                                self.best_params[f"PRF_{strategy}_{rf_model}_{baseline}_{stored_tf}tuned"] = {
                                    "baseline": baseline,
                                    "strategy": strategy,
                                    "rf_model": rf_model,
                                    "depth": a_best["depth"],
                                    "e": a_best["e"],
                                    "lambda": a_lam,
                                    "term_filter": stored_tf,
                                    "map": a_best["map"],
                                }
                                a_depth_str = f", depth={format_depth(a_best['depth'])}"
                                a_lambda_str = f", λ={a_lam:.2f}" if a_lam is not None else ""
                                print_success(
                                    f"  {strategy}+{rf_model} ({baseline}){a_depth_str}, e={a_best['e']:.0f}{a_lambda_str}, +{label}[A] (tuned), train MAP={a_best['map']:.4f}"
                                )
                            continue

                        base_mask = (
                            (df["baseline"] == baseline)
                            & (df["strategy"] == strategy)
                            & (df["rf_model"] == rf_model)
                        )
                        if has_tf:
                            base_mask &= (df["term_filter"] == "none")
                        base_data = df[base_mask]
                        if len(base_data) == 0:
                            print_warning(f"  No data for {strategy}+{rf_model}+{baseline}")
                            continue

                        best_row = base_data.loc[base_data["map"].idxmax()]
                        depth = best_row["depth"]
                        e = best_row["e"]
                        lam = best_row.get("lambda", None)
                        lam = lam if pd.notna(lam) else None

                        lambda_str = f", λ={lam:.2f}" if lam is not None else ""
                        depth_str = f", depth={format_depth(depth)}"

                        self.best_params[f"PRF_{strategy}_{rf_model}_{baseline}_none"] = {
                            "baseline": baseline,
                            "strategy": strategy,
                            "rf_model": rf_model,
                            "depth": depth,
                            "e": e,
                            "lambda": lam,
                            "term_filter": "none",
                            "map": best_row["map"],
                        }
                        print_success(
                            f"  {strategy}+{rf_model} ({baseline}){depth_str}, e={e:.0f}{lambda_str}, MAP={best_row['map']:.4f}"
                        )

                        if has_tf and strategy in term_addon_strategies:
                            for label, stored_tf, csv_tf in families_for(strategy):
                                b_map = train_map_at(baseline, strategy, rf_model, depth, e, lam, csv_tf)
                                self.best_params[f"PRF_{strategy}_{rf_model}_{baseline}_{stored_tf}"] = {
                                    "baseline": baseline,
                                    "strategy": strategy,
                                    "rf_model": rf_model,
                                    "depth": depth,
                                    "e": e,
                                    "lambda": lam,
                                    "term_filter": stored_tf,
                                    "map": b_map,
                                }
                                b_tm = f"{b_map:.4f}" if b_map is not None else "n/a"
                                print_success(
                                    f"  {strategy}+{rf_model} ({baseline}){depth_str}, e={e:.0f}{lambda_str}, +{label}[B] (same params), train MAP={b_tm}"
                                )

                                a_mask = (
                                    (df["baseline"] == baseline)
                                    & (df["strategy"] == strategy)
                                    & (df["rf_model"] == rf_model)
                                    & (df["term_filter"] == csv_tf)
                                )
                                a_data = df[a_mask]
                                if len(a_data) > 0:
                                    a_best = a_data.loc[a_data["map"].idxmax()]
                                    a_lam = a_best.get("lambda", None)
                                    a_lam = a_lam if pd.notna(a_lam) else None
                                    self.best_params[f"PRF_{strategy}_{rf_model}_{baseline}_{stored_tf}tuned"] = {
                                        "baseline": baseline,
                                        "strategy": strategy,
                                        "rf_model": rf_model,
                                        "depth": a_best["depth"],
                                        "e": a_best["e"],
                                        "lambda": a_lam,
                                        "term_filter": stored_tf,
                                        "map": a_best["map"],
                                    }
                                    a_depth_str = f", depth={format_depth(a_best['depth'])}"
                                    a_lambda_str = f", λ={a_lam:.2f}" if a_lam is not None else ""
                                    print_success(
                                        f"  {strategy}+{rf_model} ({baseline}){a_depth_str}, e={a_best['e']:.0f}{a_lambda_str}, +{label}[A] (re-tuned), train MAP={a_best['map']:.4f}"
                                    )
        else:
            print_warning(f"PRF file not found: {prf_file}")

        print(f"Total configurations loaded: {len(self.best_params)}")
        if len(self.best_params) == 0:
            print_error("No configurations loaded! Check if grid search results exist.")

        print()

    def run_baseline(self, baseline: str) -> Path:
        print(f"{Colors.GREEN}Running {baseline} baseline...{Colors.NC}")

        if baseline == "LMDirichlet":
            cmd = f'''java -cp "{self.jar_path}" org.irlab.llmprf.searcher.TRECSearcherLucene \
                --index "{self.index_path}" \
                --topics "{self.topics_test_path}" \
                --runsOutputFolder "{self.test_run_folder}" \
                --baseline_model LMDirichlet'''
        else:
            cmd = f'''java -cp "{self.jar_path}" org.irlab.llmprf.searcher.TRECSearcherLucene \
                --index "{self.index_path}" \
                --topics "{self.topics_test_path}" \
                --runsOutputFolder "{self.test_run_folder}" \
                --baseline_model BM25'''

        if baseline == "LMDirichlet":
            run_file = self.test_run_folder / "LMDirichlet-2000_content"
        else:
            run_file = self.test_run_folder / "BM25_content"

        if self.run_java_command(cmd, run_file):
            print_success("  ✓ Completed")
            return run_file
        else:
            print_error("  ✗ Failed")
            return None

    def run_monot5_rerank(self, params: Dict) -> Path:
        baseline = params["baseline"]
        depth = params["depth"]

        print(
            f"{Colors.GREEN}Running MonoT5 rerank ({baseline}, depth={format_depth(depth)})...{Colors.NC}"
        )

        baseline_arg = "LMDirichlet" if baseline == "LMDirichlet" else "BM25"

        cmd = f'''java -cp "{self.jar_path}" org.irlab.llmprf.searcher.TRECSearcherLucene \
            --index "{self.index_path}" \
            --topics "{self.topics_test_path}" \
            --runsOutputFolder "{self.test_run_folder}" \
            --baseline_model {baseline_arg} \
            --rerank_method monot5 \
            --rerank_depth {format_depth(depth)} \
            --cache_dir "{self.cache_dir}"'''

        if baseline == "LMDirichlet":
            baseline_prefix = "LMDirichlet-2000_content"
        else:
            baseline_prefix = "BM25_content"

        run_file = (
            self.test_run_folder
            / f"{baseline_prefix}_rerank-monoT5_topK-{format_depth(depth)}"
        )

        if self.run_java_command(cmd, run_file):
            print_success("  ✓ Completed")
            return run_file
        else:
            print_error("  ✗ Failed")
            return None

    def run_prf(self, params: Dict) -> Path:
        baseline = params["baseline"]
        strategy = params["strategy"]
        rf_model = params["rf_model"]
        depth = params["depth"]
        e = params["e"]
        lambda_val = params.get("lambda")
        term_filter = (params.get("term_filter", "none") or "none").lower()

        lambda_str = f", λ={lambda_val:.2f}" if lambda_val is not None else ""
        depth_str = f", depth={format_depth(depth)}"
        tf_str = f", tf={term_filter}" if term_filter != "none" else ""
        print(
            f"{Colors.GREEN}Running PRF {rf_model} ({baseline}, {strategy}{depth_str}, e={e:.0f}{lambda_str}{tf_str})...{Colors.NC}"
        )

        baseline_arg = "LMDirichlet" if baseline == "LMDirichlet" else "BM25"

        cmd_parts = [
            f'java -cp "{self.jar_path}" org.irlab.llmprf.searcher.TRECSearcherLucene',
            f'--index "{self.index_path}"',
            f'--topics "{self.topics_test_path}"',
            f'--runsOutputFolder "{self.test_run_folder}"',
            f"--baseline_model {baseline_arg}",
            f"--rerank_method prf",
            f"--prf_strategy {strategy}",
            f"--prf_model {rf_model}",
            f"-e {e:.0f}",
            f'--cache_dir "{self.cache_dir}"',
            f"--term_filter {term_filter}",
        ]

        if strategy.startswith("ORACLE"):
            cmd_parts.append(f'--qrels "{self.qrels_test_path}"')

        cmd_parts.insert(-2, f"--rerank_depth {format_depth(depth)}")

        if lambda_val is not None:
            cmd_parts.append(f"--lambda {lambda_val:.2f}")

        cmd = " \\\n            ".join(cmd_parts)

        if baseline == "LMDirichlet":
            baseline_prefix = "LMDirichlet-2000_content"
        else:
            baseline_prefix = "BM25_content"

        lambda_formatted = f"{lambda_val:.2f}" if lambda_val is not None else "0.00"
        tf_suffix = f"_termFilter-{term_filter.upper()}" if term_filter != "none" else ""

        run_file = (
            self.test_run_folder
            / f"{baseline_prefix}_prf-true_rfStrategy-{strategy}_rfModel-{rf_model}_prfSmoothing-Additive-0.1000_topK-{format_depth(depth)}_lambda-{lambda_formatted}_e-{e:.0f}{tf_suffix}"
        )

        if self.run_java_command(cmd, run_file):
            print_success("  ✓ Completed")
            return run_file
        else:
            print_error("  ✗ Failed")
            return None

    def run_java_command(self, cmd: str, expected_output: Path) -> bool:
        print(f"    Executing command:")
        print(f"    {cmd}")
        print(f"    Expected output: {expected_output}")

        try:
            log_file = expected_output.with_suffix(".log")
            print(f"    Log file: {log_file}")

            with open(log_file, "w") as log:
                result = subprocess.run(
                    cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, text=True
                )

            print(f"    Return code: {result.returncode}")
            print(f"    Output file exists: {expected_output.exists()}")

            if expected_output.exists():
                file_size = expected_output.stat().st_size
                print(f"    Output file size: {file_size} bytes")

            success = result.returncode == 0 and expected_output.exists()
            if not success:
                print_error(f"    Command failed. Check log: {log_file}")
                if log_file.exists():
                    with open(log_file, "r") as f:
                        lines = f.readlines()[:10]
                        if lines:
                            print_error("    First few log lines:")
                            for line in lines:
                                print_error(f"      {line.strip()}")

            return success
        except Exception as e:
            print_error(f"    Exception: {e}")
            return False

    def evaluate_run(self, run_file: Path) -> Dict[str, float]:
        if not run_file.exists():
            print_error(f"Run file not found: {run_file}")
            return {}

        cmd = f'trec_eval -m map -m P.10 -m ndcg_cut.100 "{self.qrels_test_path}" "{run_file}"'

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                metrics = {}
                for line in result.stdout.split("\n"):
                    if line.startswith("map "):
                        parts = line.split()
                        if len(parts) >= 3:
                            metrics["map"] = float(parts[2])
                    elif line.startswith("P_10 "):
                        parts = line.split()
                        if len(parts) >= 3:
                            metrics["p10"] = float(parts[2])
                    elif line.startswith("ndcg_cut_100 "):
                        parts = line.split()
                        if len(parts) >= 3:
                            metrics["ndcg100"] = float(parts[2])
                metrics["per_query_ap"] = self._per_query_ap(run_file)
                return metrics
            else:
                print_error(f"trec_eval failed: {result.stderr}")
                return {}
        except Exception as e:
            print_error(f"Evaluation failed: {e}")
            return {}

    def _per_query_ap(self, run_file: Path) -> Dict[str, float]:
        cmd = f'trec_eval -q -m map "{self.qrels_test_path}" "{run_file}"'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            per_query_ap = {}
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == "map" and parts[1] != "all":
                        per_query_ap[parts[1]] = float(parts[2])
            return per_query_ap
        except Exception as e:
            print_error(f"Per-query evaluation failed: {e}")
            return {}

    def run_all_tests(self):
        print_header("STEP 2: Run Configurations on Test Data")

        for config_name, params in self.best_params.items():
            print(f"{Colors.BLUE}=== Testing {config_name} ==={Colors.NC}")

            run_file = None

            if config_name.startswith("BASELINE_"):
                baseline = params["baseline"]
                run_file = self.run_baseline(baseline)
            elif config_name.startswith("MONOT5_RERANK_"):
                run_file = self.run_monot5_rerank(params)
            elif config_name.startswith("PRF_"):
                run_file = self.run_prf(params)

            if run_file and run_file.exists():
                metrics = self.evaluate_run(run_file)
                if metrics:
                    self.test_metrics[config_name] = metrics
                    print_success(
                        f"  MAP: {metrics.get('map', 0):.4f}, P@10: {metrics.get('p10', 0):.4f}, NDCG@100: {metrics.get('ndcg100', 0):.4f}"
                    )
                else:
                    print_error("  Evaluation failed")
            else:
                print_error("  Run failed or output file not found")

            print()

        self._compute_robustness_index()

    def _compute_robustness_index(self):
        baseline_pq = self.test_metrics.get("BASELINE_BM25", {}).get("per_query_ap", {})
        if not baseline_pq:
            print_warning("  No BASELINE_BM25 per-query AP available -- skipping Robustness Index")
            return

        epsilon = 1e-9
        for config_name, metrics in self.test_metrics.items():
            if config_name == "BASELINE_BM25":
                continue
            per_query_ap = metrics.get("per_query_ap", {})
            common_qids = set(per_query_ap) & set(baseline_pq)
            if not common_qids:
                continue
            improved = sum(1 for q in common_qids if per_query_ap[q] - baseline_pq[q] > epsilon)
            worsened = sum(1 for q in common_qids if baseline_pq[q] - per_query_ap[q] > epsilon)
            metrics["ri"] = (improved - worsened) / len(common_qids)

    def generate_report(self):
        print_header("STEP 3: Generate Test Report")

        report_file = self.test_results_dir / "test_evaluation_report.txt"

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"TEST EVALUATION REPORT - {self.index}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Topics: {self.topics_test}\n")
            f.write(f"Test Qrels: {self.qrels_test}\n")
            f.write("\n")

            f.write("TRAINING vs TEST PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Configuration':<30} {'Train MAP':<12} {'Test MAP':<12} {'Test P@10':<12} {'Test NDCG@100':<15}\n"
            )
            f.write("-" * 80 + "\n")

            for config_name, train_params in self.best_params.items():
                train_map = train_params.get("map") or 0
                test_metrics = self.test_metrics.get(config_name, {})
                test_map = test_metrics.get("map") or 0
                test_p10 = test_metrics.get("p10") or 0
                test_ndcg = test_metrics.get("ndcg100") or 0

                f.write(
                    f"{config_name:<30} {train_map:<12.4f} {test_map:<12.4f} {test_p10:<12.4f} {test_ndcg:<15.4f}\n"
                )

            f.write("\n")
            f.write("BEST TEST PERFORMANCE BY METRIC\n")
            f.write("-" * 80 + "\n")

            if self.test_metrics:
                best_map_config = max(
                    self.test_metrics.items(), key=lambda x: x[1].get("map", 0)
                )
                f.write(
                    f"Best MAP: {best_map_config[0]} ({best_map_config[1]['map']:.4f})\n"
                )

                best_p10_config = max(
                    self.test_metrics.items(), key=lambda x: x[1].get("p10", 0)
                )
                f.write(
                    f"Best P@10: {best_p10_config[0]} ({best_p10_config[1]['p10']:.4f})\n"
                )

                best_ndcg_config = max(
                    self.test_metrics.items(), key=lambda x: x[1].get("ndcg100", 0)
                )
                f.write(
                    f"Best NDCG@100: {best_ndcg_config[0]} ({best_ndcg_config[1]['ndcg100']:.4f})\n"
                )

            f.write("\n")
            f.write("FILES GENERATED\n")
            f.write("-" * 80 + "\n")
            f.write(f"Test runs: {self.test_run_folder}/\n")
            f.write(f"Test results: {self.test_results_dir}/\n")
            f.write(f"This report: {report_file}\n")

        print_success(f"✓ Report generated: {report_file}")

        print("\n" + "=" * 80)
        print("TEST EVALUATION SUMMARY")
        print("=" * 80)

        for config_name, test_metrics in self.test_metrics.items():
            train_map = self.best_params[config_name].get("map") or 0
            test_map = test_metrics.get("map") or 0
            improvement = (
                ((test_map - train_map) / train_map * 100) if train_map > 0 else 0
            )

            print(
                f"{config_name:<25} Train: {train_map:.4f} → Test: {test_map:.4f} ({improvement:+.1f}%)"
            )

    def save_results_json(self):
        shared_file = Path(self.folder) / "test_results" / "all_results.json"
        shared_file.parent.mkdir(parents=True, exist_ok=True)

        if shared_file.exists():
            with open(shared_file) as f:
                all_results = json.load(f)
        else:
            all_results = {}

        dataset_label_map = {
            "ap8889": "AP8889",
            "robust04": "R04",
            "wt10g": "WT10G",
            "msmarco": "DL-20",
        }
        dataset_label = dataset_label_map.get(self.index, self.index)

        for config_name, metrics in self.test_metrics.items():
            if config_name not in all_results:
                all_results[config_name] = {}
            train_params = self.best_params.get(config_name, {})
            all_results[config_name][dataset_label] = {
                "map": metrics.get("map", None),
                "ndcg100": metrics.get("ndcg100", None),
                "depth": json_safe(train_params.get("depth")),
                "e": json_safe(train_params.get("e")),
                "lambda": json_safe(train_params.get("lambda")),
                "ri": metrics.get("ri", None),
            }

        with open(shared_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print_success(f"✓ Results saved to shared JSON: {shared_file}")

    def run(self):
        try:
            self.print_configuration()
            self.load_best_params()
            self.run_all_tests()
            self.generate_report()
            self.save_results_json()

            print_header("EVALUATION COMPLETED")
            print_success("Test evaluation completed successfully!")
            print(f"Results saved to: {self.test_results_dir}")
            return 0

        except Exception as e:
            print_error(f"Error: {e}")
            import traceback

            traceback.print_exc()
            return 1


DATASETS = ["AP8889", "R04", "WT10G", "DL-20"]
NARRATIVE_SUPPORT = {"AP8889": True, "R04": True, "WT10G": True, "DL-20": False}
RF_MODELS = ["RM3", "DMM", "MEDMM"]
RF_DISPLAY = {"RM3": "RM3", "DMM": "DMM", "MEDMM": "MEDMM"}
BASELINE = "BM25"


def requires_narrative(config_name: str) -> bool:
    if "VLLM-NARR" in config_name:
        return True
    tail = config_name.split("_")[-1]
    return "VLLMSpans2" in tail and "NoNarr" not in tail


def prf_cname(strat: str, rf: str, tf: str = "none", baseline: str = BASELINE) -> str:
    return f"PRF_{strat}_{rf}_{baseline}_{tf}"


SS2_SUFFIX = {
    "vllmspans2-nonarr": "VLLMSpans2-NoNarrtuned",
    "vllmspans2": "VLLMSpans2tuned",
}
JUDGESPANS_SUFFIX = "VLLMJudgeSpanstuned"

TWO_CALL_FILTER_STRATS = ["MONOT5", "MONOT5-PROB", "VLLM", "VLLM-PROB", "VLLM-NARR", "VLLM-NARR-PROB"]
LOGIT_ONLY_STRATS = {"MONOT5-PROB", "VLLM-PROB", "VLLM-NARR-PROB"}
ONE_CALL_JUDGE_STRATS = ["VLLM-JUDGESPANS", "VLLM-NARR-JUDGESPANS"]

SUMMARY_COMBOS = [
    ("PRF", "none", "+ {rf}"),
    ("MONOT5", "none", "+ MonoT5F + {rf}"),
    ("MONOT5-PROB", "none", "+ MonoT5F + {rf} w/prob"),
    ("VLLM", "none", "+ LLMF + {rf}"),
    ("VLLM-PROB", "none", "+ LLMF + {rf} w/prob"),
    ("VLLM-NARR", "none", "+ LLMF w/narr + {rf}"),
    ("VLLM-NARR-PROB", "none", "+ LLMF w/narr + {rf} w/prob"),
    ("PRF", "vllmspans2-nonarr", "+ SpanSelection + {rf}"),
    ("PRF", "vllmspans2", "+ SpanSelection w/narr + {rf}"),
    ("MONOT5", "vllmspans2-nonarr", "+ MonoT5F + SpanSelection + {rf}"),
    ("MONOT5-PROB", "vllmspans2-nonarr", "+ MonoT5F + SpanSelection + {rf} w/prob"),
    ("MONOT5", "vllmspans2", "+ MonoT5F + SpanSelection w/narr + {rf}"),
    ("MONOT5-PROB", "vllmspans2", "+ MonoT5F + SpanSelection w/narr + {rf} w/prob"),
    ("VLLM", "vllmspans2-nonarr", "+ LLMF + SpanSelection + {rf}"),
    ("VLLM-PROB", "vllmspans2-nonarr", "+ LLMF + SpanSelection + {rf} w/prob"),
    ("VLLM", "vllmspans2", "+ LLMF + SpanSelection w/narr + {rf}"),
    ("VLLM-PROB", "vllmspans2", "+ LLMF + SpanSelection w/narr + {rf} w/prob"),
    ("VLLM-NARR", "vllmspans2-nonarr", "+ LLMF w/narr + SpanSelection + {rf}"),
    ("VLLM-NARR-PROB", "vllmspans2-nonarr", "+ LLMF w/narr + SpanSelection + {rf} w/prob"),
    ("VLLM-NARR", "vllmspans2", "+ LLMF w/narr + SpanSelection w/narr + {rf}"),
    ("VLLM-NARR-PROB", "vllmspans2", "+ LLMF w/narr + SpanSelection w/narr + {rf} w/prob"),
    ("VLLM-JUDGESPANS", "vllmjudgespans", "+ LLMF+SpanSelection (1 call) + {rf}"),
    ("VLLM-NARR-JUDGESPANS", "vllmjudgespans", "+ LLMF+SpanSelection (1 call) w/narr + {rf}"),
]


def combos_for(rf: str):
    if rf == "RM3":
        return SUMMARY_COMBOS
    return [c for c in SUMMARY_COMBOS if c[0] not in LOGIT_ONLY_STRATS]


def combo_cname(strat: str, tf_cli: str, rf: str) -> str:
    if tf_cli == "none":
        tf = "none"
    elif tf_cli == "vllmjudgespans":
        tf = JUDGESPANS_SUFFIX
    else:
        tf = SS2_SUFFIX[tf_cli]
    return prf_cname(strat, rf, tf)


def two_call_span_candidates(rf: str) -> List[str]:
    strats = TWO_CALL_FILTER_STRATS if rf == "RM3" else [s for s in TWO_CALL_FILTER_STRATS if s not in LOGIT_ONLY_STRATS]
    return [prf_cname(s, rf, tf) for s in strats for tf in SS2_SUFFIX.values()]


def one_call_span_candidates(rf: str) -> List[str]:
    return [prf_cname(s, rf, JUDGESPANS_SUFFIX) for s in ONE_CALL_JUDGE_STRATS]


def unfiltered_span_candidates(rf: str) -> List[str]:
    return [prf_cname("PRF", rf, tf) for tf in SS2_SUFFIX.values()]


def generate_latex_tables(folder: str):
    shared_file = Path(folder) / "test_results" / "all_results.json"
    if not shared_file.exists():
        print_error(f"Shared results file not found: {shared_file}")
        print_error("Run evaluate_on_test.sh for each dataset first.")
        return 1

    with open(shared_file) as f:
        all_results = json.load(f)

    sig_file = Path(folder) / "test_results" / "significance_table5.json"
    significance = None
    if sig_file.exists():
        with open(sig_file) as f:
            significance = json.load(f)
    else:
        print_warning(
            f"{sig_file} not found -- run compute_significance.py first. "
            "table_cumulative_summary.tex will be generated without significance superscripts."
        )

    def getv(config_name: str, dataset: str, metric: str = "map"):
        if not NARRATIVE_SUPPORT.get(dataset, True) and requires_narrative(config_name):
            return None
        return all_results.get(config_name, {}).get(dataset, {}).get(metric, None)

    def cell(config_name: str, dataset: str, bold_set=frozenset()) -> str:
        v = getv(config_name, dataset)
        if v is None:
            return "--"
        s = f"{v:.4f}"
        return f"\\textbf{{{s}}}" if config_name in bold_set else s

    def bold_max(config_names: List[str], dataset: str) -> set:
        vals = {cn: getv(cn, dataset) for cn in config_names}
        present = [v for v in vals.values() if v is not None]
        if not present:
            return set()
        best = max(present)
        return {cn for cn, v in vals.items() if v is not None and abs(v - best) < 1e-9}

    def pick_winner(config_names: List[str], dataset: str):
        best_cname, best_val = None, None
        for cn in config_names:
            v = getv(cn, dataset)
            if v is not None and (best_val is None or v > best_val):
                best_cname, best_val = cn, v
        return best_cname

    output_dir = Path(folder) / "test_results"

    def write_table(name: str, content: str):
        out_file = output_dir / f"table_{name}.tex"
        with open(out_file, "w") as f:
            f.write(content + "\n")
        print_success(f"✓ LaTeX table written: {out_file}")
        print()
        print(content)
        print()

    def build_filtering_table() -> str:
        lines = []
        lines.append(r"\begin{table*}")
        lines.append(r"    \centering")
        lines.append(r"    \caption{}")
        lines.append(r"    \ra{1}")
        lines.append(r"    \label{tab:filtering-results}")
        lines.append(r"    \begin{tabular}{llrrrr}")
        lines.append(r"        \toprule")
        lines.append(r"        Estimator & Evidence used for feedback & AP8889 & ROBUST04 & WT10G & DL-20 \\")
        lines.append(r"        \midrule")
        bm25_cells = " & ".join(cell(f"BASELINE_{BASELINE}", ds) for ds in DATASETS)
        lines.append(f"        -- & BM25 (no feedback) & {bm25_cells} \\\\")
        lines.append(r"        \midrule")
        for i, rf in enumerate(RF_MODELS):
            rows = [
                ("Blind top-$k$", prf_cname("PRF", rf)),
                ("MonoT5F-filtered", prf_cname("MONOT5", rf)),
                ("LLMF-filtered", prf_cname("VLLM", rf)),
            ]
            row_cnames = [cn for _, cn in rows]
            for j, (label, cname) in enumerate(rows):
                bold_sets = {ds: bold_max(row_cnames, ds) for ds in DATASETS}
                cells = " & ".join(cell(cname, ds, bold_sets[ds]) for ds in DATASETS)
                prefix = RF_DISPLAY[rf] if j == 0 else ""
                lines.append(f"        {prefix} & {label} & {cells} \\\\")
            if i < len(RF_MODELS) - 1:
                lines.append(r"        \addlinespace")
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}")
        lines.append(r"\end{table*}")
        return "\n".join(lines)

    def build_confidence_narrative_table() -> str:
        lines = []
        lines.append(r"\begin{table*}")
        lines.append(r"    \centering")
        lines.append(r"    \ra{1}")
        lines.append(r"    \caption{}")
        lines.append(r"    \label{tab:confidence-narrative-results}")
        lines.append(r"    \begin{tabular}{llrrrr}")
        lines.append(r"        \toprule")
        lines.append(r"        Estimator & Judge signal or instruction & AP8889 & ROBUST04 & WT10G & DL-20 \\")
        lines.append(r"        \midrule")
        lines.append(r"        \multicolumn{6}{@{}l}{\emph{(a) Weighting documents by the judge's confidence}} \\")
        panel_a_rows = [
            ("MonoT5F", prf_cname("MONOT5", "RM3")),
            ("MonoT5F w/prob", prf_cname("MONOT5-PROB", "RM3")),
            ("LLMF", prf_cname("VLLM", "RM3")),
            ("LLMF w/prob", prf_cname("VLLM-PROB", "RM3")),
        ]
        for j, (label, cname) in enumerate(panel_a_rows):
            cells = " & ".join(cell(cname, ds) for ds in DATASETS)
            prefix = "RM3" if j == 0 else ""
            lines.append(f"        {prefix} & {label} & {cells} \\\\")
        lines.append(r"        \midrule")
        lines.append(r"        \multicolumn{6}{@{}l}{\emph{(b) Adding the topic narrative to the LLM prompt}} \\")
        for i, rf in enumerate(RF_MODELS):
            rows = [("LLMF", prf_cname("VLLM", rf)), ("LLMF w/narr", prf_cname("VLLM-NARR", rf))]
            if rf == "RM3":
                rows.append(("LLMF w/narr and prob", prf_cname("VLLM-NARR-PROB", rf)))
            for j, (label, cname) in enumerate(rows):
                cells = " & ".join(("--" if ds == "DL-20" else cell(cname, ds)) for ds in DATASETS)
                prefix = RF_DISPLAY[rf] if j == 0 else ""
                lines.append(f"        {prefix} & {label} & {cells} \\\\")
            if i < len(RF_MODELS) - 1:
                lines.append(r"        \addlinespace")
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}")
        lines.append(r"\end{table*}")
        return "\n".join(lines)

    def build_spanselection_table() -> str:
        lines = []
        lines.append(r"\begin{table*}")
        lines.append(r"    \centering")
        lines.append(r"    \caption{}")
        lines.append(r"    \label{tab:spanselection-results}")
        lines.append(r"    \begin{tabular}{llrrrr}")
        lines.append(r"        \toprule")
        lines.append(r"        Estimator & Evidence used for feedback & AP8889 & ROBUST04 & WT10G & DL-20 \\")
        lines.append(r"        \midrule")
        for i, rf in enumerate(RF_MODELS):
            blind_cn = prf_cname("PRF", rf)
            unfiltered_nonarr_cn = prf_cname("PRF", rf, SS2_SUFFIX["vllmspans2-nonarr"])
            unfiltered_narr_cn = prf_cname("PRF", rf, SS2_SUFFIX["vllmspans2"])
            row_cnames_per_ds = {
                ds: {
                    "blind": blind_cn,
                    "unfiltered": unfiltered_nonarr_cn,
                    "unfiltered_narr": unfiltered_narr_cn,
                    "filtered_1call": pick_winner(one_call_span_candidates(rf), ds),
                    "filtered_2call": pick_winner(two_call_span_candidates(rf), ds),
                }
                for ds in DATASETS
            }
            bold_sets = {
                ds: bold_max([c for c in row_cnames_per_ds[ds].values() if c is not None], ds)
                for ds in DATASETS
            }
            rows = [
                ("Blind top-$k$", "blind"),
                ("Unfiltered SpanSelection", "unfiltered"),
                ("Unfiltered SpanSelection w/narr", "unfiltered_narr"),
                ("Filtered SpanSelection (1 call)", "filtered_1call"),
                ("Filtered SpanSelection (2 calls)", "filtered_2call"),
            ]
            for j, (label, key) in enumerate(rows):
                cells = " & ".join(
                    ("--" if row_cnames_per_ds[ds][key] is None else cell(row_cnames_per_ds[ds][key], ds, bold_sets[ds]))
                    for ds in DATASETS
                )
                prefix = RF_DISPLAY[rf] if j == 0 else ""
                lines.append(f"        {prefix} & {label} & {cells} \\\\")
            if i < len(RF_MODELS) - 1:
                lines.append(r"        \addlinespace")
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}")
        lines.append(r"\end{table*}")
        return "\n".join(lines)

    def build_cumulative_summary_table() -> str:
        ROWS = [
            ("BM25", [f"BASELINE_{BASELINE}"]),
            ("MonoT5 rerank", [f"MONOT5_RERANK_{BASELINE}"]),
            ("Blind PRF", [prf_cname("PRF", rf) for rf in RF_MODELS]),
            ("Document filtering", [prf_cname(s, rf) for rf in RF_MODELS for s in ("MONOT5", "VLLM")]),
            ("Filtering + probability weight", [prf_cname(s, "RM3") for s in ("MONOT5-PROB", "VLLM-PROB")]),
            ("LLM filtering + narrative", (
                [prf_cname("VLLM-NARR", "RM3"), prf_cname("VLLM-NARR-PROB", "RM3")]
                + [prf_cname("VLLM-NARR", rf) for rf in ("DMM", "MEDMM")]
            )),
            ("Unfiltered SpanSelection", [c for rf in RF_MODELS for c in unfiltered_span_candidates(rf)]),
            ("Filtered SpanSelection (1 call)", [c for rf in RF_MODELS for c in one_call_span_candidates(rf)]),
            ("Filtered SpanSelection (2 calls)", [c for rf in RF_MODELS for c in two_call_span_candidates(rf)]),
        ]
        letters = {row: chr(ord("a") + i) for i, (row, _) in enumerate(ROWS)}

        lines = []
        lines.append(r"\begin{table*}")
        lines.append(r"    \centering")
        lines.append(r"    \caption{}")
        lines.append(r"    \label{tab:cumulative-summary}")
        lines.append(r"    \begin{tabular}{lrrrr}")
        lines.append(r"        \toprule")
        lines.append(r"        Contribution & AP8889 & ROBUST04 & WT10G & DL-20 \\")
        lines.append(r"        \midrule")

        winners = {row: {ds: pick_winner(cands, ds) for ds in DATASETS} for row, cands in ROWS}
        bold_sets = {
            ds: bold_max([w for row, _ in ROWS for w in [winners[row][ds]] if w is not None], ds)
            for ds in DATASETS
        }
        sig_by_ds = {}
        if significance is not None:
            for ds in DATASETS:
                ds_data = significance.get(ds, {})
                sig_by_ds[ds] = {r["row"]: r for r in ds_data.get("rows", [])}

        for i, (row, _) in enumerate(ROWS):
            letter = letters[row]
            beats_cells = []
            value_cells = []
            for ds in DATASETS:
                cname = winners[row][ds]
                if cname is None:
                    beats_cells.append("")
                    value_cells.append("--")
                    continue
                v = getv(cname, ds)
                s = f"{v:.4f}"
                if cname in bold_sets[ds]:
                    s = f"\\textbf{{{s}}}"
                value_cells.append(s)
                beats = ""
                if sig_by_ds and ds in sig_by_ds and row in sig_by_ds[ds]:
                    beats_letters = sig_by_ds[ds][row].get("beats_letters", [])
                    if beats_letters:
                        beats = "\\scriptsize " + ",".join(beats_letters)
                beats_cells.append(beats)
            beats_line = " & ".join(beats_cells)
            values_line = " & ".join(value_cells)
            lines.append(f"        \\multirow{{2}}{{*}}{{({letter}) {row}}} & {beats_line} \\\\")
            lines.append(f"         & {values_line} \\\\")
            if i < len(ROWS) - 1:
                lines.append(r"        \addlinespace[2pt]")

        lines.append(r"        \midrule")
        oracle_cands = [prf_cname("ORACLE-K", rf) for rf in RF_MODELS]
        oracle_cells = " & ".join(cell(pick_winner(oracle_cands, ds), ds) if pick_winner(oracle_cands, ds) else "--" for ds in DATASETS)
        lines.append(f"        Oracle & {oracle_cells} \\\\")
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}")
        lines.append(r"\end{table*}")
        return "\n".join(lines)

    def build_ri_table() -> str:
        def ri_cell(config_name, ds):
            if config_name is None:
                return "--"
            if not NARRATIVE_SUPPORT.get(ds, True) and requires_narrative(config_name):
                return "--"
            ri = all_results.get(config_name, {}).get(ds, {}).get("ri", None)
            return "--" if ri is None else f"${ri:+.2f}$"

        rows = [
            ("MonoT5 rerank", f"MONOT5_RERANK_{BASELINE}"),
            ("Blind PRF", prf_cname("PRF", "RM3")),
            ("+ MonoT5F", prf_cname("MONOT5", "RM3")),
            ("+ MonoT5F w/prob", prf_cname("MONOT5-PROB", "RM3")),
            ("+ LLMF", prf_cname("VLLM", "RM3")),
            ("+ LLMF w/narr", prf_cname("VLLM-NARR", "RM3")),
            ("+ LLMF w/narr and prob", prf_cname("VLLM-NARR-PROB", "RM3")),
            ("+ Unfiltered SpanSelection", None),
            ("+ LLMF w/narr + SpanSelection w/narr and prob", prf_cname("VLLM-NARR-PROB", "RM3", SS2_SUFFIX["vllmspans2"])),
        ]

        lines = []
        lines.append(r"\begin{table}")
        lines.append(r"    \centering")
        lines.append(r"    \caption{}")
        lines.append(r"    \label{tab:ri-results}")
        lines.append(r"    \begin{tabular}{lrrrr}")
        lines.append(r"        \toprule")
        lines.append(r"        Method (all based on RM3) & AP8889 & ROBUST04 & WT10G & DL-20 \\")
        lines.append(r"        \midrule")
        for label, cname in rows:
            if label == "+ Unfiltered SpanSelection":
                cells = " & ".join(
                    ri_cell(pick_winner(unfiltered_span_candidates("RM3"), ds), ds) for ds in DATASETS
                )
            else:
                cells = " & ".join(ri_cell(cname, ds) for ds in DATASETS)
            lines.append(f"        {label} & {cells} \\\\")
        lines.append(r"        \midrule")
        oracle_cells = " & ".join(ri_cell(prf_cname("ORACLE-K", "RM3"), ds) for ds in DATASETS)
        lines.append(f"        + Oracle & {oracle_cells} \\\\")
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def build_params_table() -> str:
        def depth_cell(config_name, ds):
            if not NARRATIVE_SUPPORT.get(ds, True) and requires_narrative(config_name):
                return "--"
            depth = all_results.get(config_name, {}).get(ds, {}).get("depth", None)
            return "--" if depth is None else format_depth(depth)

        lines = []
        lines.append(r"\begin{table}")
        lines.append(r"    \centering")
        lines.append(r"    \caption{}")
        lines.append(r"    \label{tab:params-results}")
        lines.append(r"    \begin{tabular}{llrrrr}")
        lines.append(r"        \toprule")
        lines.append(r"        Estimator & Evidence used for feedback & AP8889 & ROBUST04 & WT10G & DL-20 \\")
        lines.append(r"        \midrule")
        for i, rf in enumerate(RF_MODELS):
            rows = [
                ("Blind top-$k$", prf_cname("PRF", rf)),
                ("MonoT5F-filtered", prf_cname("MONOT5", rf)),
                ("LLMF-filtered", prf_cname("VLLM", rf)),
                ("LLMF w/narr-filtered", prf_cname("VLLM-NARR", rf)),
            ]
            for j, (label, cname) in enumerate(rows):
                cells = " & ".join(depth_cell(cname, ds) for ds in DATASETS)
                prefix = RF_DISPLAY[rf] if j == 0 else ""
                lines.append(f"        {prefix} & {label} & {cells} \\\\")
            if i < len(RF_MODELS) - 1:
                lines.append(r"        \addlinespace")
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def build_appendix_summary_table() -> str:
        def top_two(cnames, ds, metric):
            vals = sorted({v for cn in cnames if (v := getv(cn, ds, metric)) is not None}, reverse=True)
            return (vals[0] if vals else None, vals[1] if len(vals) > 1 else None)

        all_cnames = [f"BASELINE_{BASELINE}", f"MONOT5_RERANK_{BASELINE}"]
        for rf in RF_MODELS:
            all_cnames += [combo_cname(s, tf, rf) for s, tf, _ in combos_for(rf)]
        global_rank = {(ds, m): top_two(all_cnames, ds, m) for ds in DATASETS for m in ("map", "ndcg100")}

        def group_rank_for(rf):
            cnames = [combo_cname(s, tf, rf) for s, tf, _ in combos_for(rf)]
            return {(ds, m): top_two(cnames, ds, m) for ds in DATASETS for m in ("map", "ndcg100")}

        def styled(cname, ds, metric, rf_rank):
            v = getv(cname, ds, metric)
            if v is None:
                return "--"
            s = f"{v:.4f}"
            gbest, gsecond = global_rank[(ds, metric)]
            if gbest is not None and abs(v - gbest) < 1e-9:
                s = f"\\textbf{{{s}}}"
            elif gsecond is not None and abs(v - gsecond) < 1e-9:
                s = f"\\underline{{{s}}}"
            rbest, rsecond = rf_rank[(ds, metric)]
            if rbest is not None and abs(v - rbest) < 1e-9:
                s += "$^{\\dagger}$"
            elif rsecond is not None and abs(v - rsecond) < 1e-9:
                s += "$^{\\ddagger}$"
            return s

        def plain_cell(cname, ds, metric):
            v = getv(cname, ds, metric)
            return "--" if v is None else f"{v:.4f}"

        def data_row(label, cname, rf_rank=None):
            if rf_rank is None:
                map_cells = " & ".join(plain_cell(cname, ds, "map") for ds in DATASETS)
                ndcg_cells = " & ".join(plain_cell(cname, ds, "ndcg100") for ds in DATASETS)
            else:
                map_cells = " & ".join(styled(cname, ds, "map", rf_rank) for ds in DATASETS)
                ndcg_cells = " & ".join(styled(cname, ds, "ndcg100", rf_rank) for ds in DATASETS)
            return f"    {label} & {map_cells} & {ndcg_cells} \\\\"

        header = (
            "    \\multirow{2.3}{*}{Method} & \\multicolumn{4}{c}{AP@1000} & \\multicolumn{4}{c}{NDCG@100} \\\\\n"
            "    \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n"
            "    & AP8889 & R04 & WT10G & DL-20 & AP8889 & R04 & WT10G & DL-20 \\\\\n"
            "    \\midrule"
        )

        lines = []
        lines.append(r"\begin{landscape}")
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(r"\ra{1.05}")
        lines.append(r"\begin{longtable}{@{}lrrrrrrrr@{}}")
        lines.append("    \\caption{}\\label{tab:results-summary}\\\\")
        lines.append(r"    \toprule")
        lines.append(header)
        lines.append(r"    \endfirsthead")
        lines.append("    \\caption{}\\\\")
        lines.append(r"    \toprule")
        lines.append(header)
        lines.append(r"    \endhead")
        lines.append(r"    \bottomrule")
        lines.append(r"    \endfoot")
        lines.append(data_row("BM25", f"BASELINE_{BASELINE}"))
        lines.append(data_row("~~+ MonoT5 rerank", f"MONOT5_RERANK_{BASELINE}"))
        for rf in RF_MODELS:
            rf_rank = group_rank_for(rf)
            lines.append(r"    \midrule")
            lines.append(f"    \\multicolumn{{9}}{{@{{}}l}}{{\\textbf{{{rf}}}}} \\\\")
            lines.append(r"    \addlinespace[1pt]")
            for strat, tf, label_tpl in combos_for(rf):
                label = label_tpl.format(rf=rf)
                lines.append(data_row(f"~~{label}", combo_cname(strat, tf, rf), rf_rank))
            lines.append(r"    \midrule")
            lines.append(data_row(f"~~+ {rf} Oracle", prf_cname("ORACLE-K", rf)))
        lines.append(r"\end{longtable}")
        lines.append(r"\end{landscape}")
        return "\n".join(lines)

    def build_appendix_params_table() -> str:
        def params_cell(cname, ds):
            if not NARRATIVE_SUPPORT.get(ds, True) and requires_narrative(cname):
                return "--"
            entry = all_results.get(cname, {}).get(ds, {})
            depth, e, lam = entry.get("depth"), entry.get("e"), entry.get("lambda")
            if depth is None and e is None and lam is None:
                return "--"
            parts = []
            if depth is not None:
                parts.append(f"k{{=}}{format_depth(depth)}")
            if e is not None:
                parts.append(f"e{{=}}{e:.0f}")
            if lam is not None:
                parts.append(f"\\alpha{{=}}{lam:.2f}")
            return "$" + ",\\,".join(parts) + "$"

        def data_row(label, cname):
            cells = " & ".join(params_cell(cname, ds) for ds in DATASETS)
            return f"    {label} & {cells} \\\\"

        header = (
            "    \\multirow{2.3}{*}{Method} & \\multicolumn{4}{c}{Tuned parameters} \\\\\n"
            "    \\cmidrule(lr){2-5}\n"
            "    & AP8889 & R04 & WT10G & DL-20 \\\\\n"
            "    \\midrule"
        )

        lines = []
        lines.append(r"\begin{landscape}")
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(r"\ra{1.05}")
        lines.append(r"\begin{longtable}{@{}lllll@{}}")
        lines.append("    \\caption{}\\label{tab:results-summary-params}\\\\")
        lines.append(r"    \toprule")
        lines.append(header)
        lines.append(r"    \endfirsthead")
        lines.append("    \\caption{}\\\\")
        lines.append(r"    \toprule")
        lines.append(header)
        lines.append(r"    \endhead")
        lines.append(r"    \bottomrule")
        lines.append(r"    \endfoot")
        lines.append(data_row("BM25", f"BASELINE_{BASELINE}"))
        lines.append(data_row("~~+ MonoT5 rerank", f"MONOT5_RERANK_{BASELINE}"))
        for rf in RF_MODELS:
            lines.append(r"    \midrule")
            lines.append(f"    \\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{rf}}}}} \\\\")
            lines.append(r"    \addlinespace[1pt]")
            for strat, tf, label_tpl in combos_for(rf):
                label = label_tpl.format(rf=rf)
                lines.append(data_row(f"~~{label}", combo_cname(strat, tf, rf)))
            lines.append(r"    \midrule")
            lines.append(data_row(f"~~+ {rf} Oracle", prf_cname("ORACLE-K", rf)))
        lines.append(r"\end{longtable}")
        lines.append(r"\end{landscape}")
        return "\n".join(lines)

    def build_appendix_ri_table() -> str:
        def ri_cell(cname, ds):
            if not NARRATIVE_SUPPORT.get(ds, True) and requires_narrative(cname):
                return "--"
            ri = all_results.get(cname, {}).get(ds, {}).get("ri", None)
            return "--" if ri is None else f"${ri:+.2f}$"

        def data_row(label, cname):
            cells = " & ".join(ri_cell(cname, ds) for ds in DATASETS)
            return f"    {label} & {cells} \\\\"

        header = (
            "    \\multirow{2.3}{*}{Method} & \\multicolumn{4}{c}{Robustness Index} \\\\\n"
            "    \\cmidrule(lr){2-5}\n"
            "    & AP8889 & R04 & WT10G & DL-20 \\\\\n"
            "    \\midrule"
        )

        lines = []
        lines.append(r"\begin{landscape}")
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(r"\ra{1.05}")
        lines.append(r"\begin{longtable}{@{}lllll@{}}")
        lines.append("    \\caption{}\\label{tab:results-summary-ri}\\\\")
        lines.append(r"    \toprule")
        lines.append(header)
        lines.append(r"    \endfirsthead")
        lines.append("    \\caption{}\\\\")
        lines.append(r"    \toprule")
        lines.append(header)
        lines.append(r"    \endhead")
        lines.append(r"    \bottomrule")
        lines.append(r"    \endfoot")
        lines.append(data_row("BM25", f"BASELINE_{BASELINE}"))
        lines.append(data_row("~~+ MonoT5 rerank", f"MONOT5_RERANK_{BASELINE}"))
        for rf in RF_MODELS:
            lines.append(r"    \midrule")
            lines.append(f"    \\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{rf}}}}} \\\\")
            lines.append(r"    \addlinespace[1pt]")
            for strat, tf, label_tpl in combos_for(rf):
                label = label_tpl.format(rf=rf)
                lines.append(data_row(f"~~{label}", combo_cname(strat, tf, rf)))
            lines.append(r"    \midrule")
            lines.append(data_row(f"~~+ {rf} Oracle", prf_cname("ORACLE-K", rf)))
        lines.append(r"\end{longtable}")
        lines.append(r"\end{landscape}")
        return "\n".join(lines)

    write_table("filtering", build_filtering_table())
    write_table("confidence_narrative", build_confidence_narrative_table())
    write_table("spanselection", build_spanselection_table())
    write_table("cumulative_summary", build_cumulative_summary_table())
    write_table("ri", build_ri_table())
    write_table("params", build_params_table())
    write_table("appendix_summary", build_appendix_summary_table())
    write_table("appendix_params", build_appendix_params_table())
    write_table("appendix_ri", build_appendix_ri_table())

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PRF strategies on test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_evaluation.py robust04
  python test_evaluation.py ap8889
  python test_evaluation.py --latex /path/to/data/folder
        """,
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g., robust04, ap8889). Not required when using --latex.",
    )
    parser.add_argument(
        "--jar",
        default=None,
        help="Path to the JAR file. Overrides the JAR_PATH environment variable and the default relative path.",
    )
    parser.add_argument(
        "--latex",
        metavar="FOLDER",
        default=None,
        help=(
            "Generate LaTeX tables from already-collected results and exit. "
            "Pass the base data folder (FOLDER in dataset_config.sh). "
            "Run all datasets first before using this flag."
        ),
    )
    args = parser.parse_args()

    if args.latex:
        return generate_latex_tables(args.latex)

    if not args.dataset:
        parser.error("dataset is required unless --latex is used")

    try:
        evaluator = TestEvaluator(args.dataset, jar_path=args.jar)
        return evaluator.run()
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
