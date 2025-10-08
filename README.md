# PRF-LLM: LLM-Assisted Pseudo-Relevance Feedback

This repository contains the implementation of **LLM-Assisted Pseudo-Relevance Feedback**, a novel approach that leverages Large Language Models (LLMs) to enhance relevance feedback in information retrieval systems.

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{otero2026llm,
  title={LLM-Assisted Pseudo-Relevance Feedback},
  author={Otero, David and Parapar, Javier},
  booktitle={Proceedings of the 48th European Conference on Information Retrieval (ECIR)},
  year={2026},
  organization={Springer}
}
```

## 🎯 Overview

This project implements several PRF strategies that use LLMs to filter pseudo-relevant documents:

- **PRF (Blind)**: Traditional pseudo-relevance feedback (baseline)
- **MonoT5**: Uses MonoT5 model for binary relevance classification
- **MonoT5-PROB**: Uses MonoT5 relevance probabilities as document weights
- **VLLM**: Uses VLLM-served models for relevance filtering
- **VLLM-PROB**: Uses VLLM relevance probabilities as document weights
- **OLLAMA**: Uses Ollama-served models for relevance filtering
- **ORACLE**: Uses ground-truth relevance judgments (upper bound)
- **ORACLE-K**: Uses top-k ground-truth relevant documents (upper bound)

## 🏗️ Architecture

The system consists of:

- **Java backend**: Lucene-based IR system with PRF implementation
- **Python services**: LLM scoring services (MonoT5, VLLM, Ollama)
- **Analysis scripts**: Grid search, evaluation, and visualization tools

## 📋 Requirements

### Java
- Java 21 or higher
- Maven 3.6+

### Python
- Python 3.8+
- pandas
- matplotlib
- seaborn
- transformers (for MonoT5)
- torch
- vllm (optional, for VLLM service)

### Data
- TREC test collections (e.g., AP88-89, ROBUST04)
- Topic files
- Qrels (relevance judgments)

## 🚀 Quick Start

### 1. Build the Project

```bash
mvn clean package
```

This creates a JAR file in `target/prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar`

### 2. Configure Dataset

Edit `src/main/scripts/dataset_config.sh` to set your data paths:

```bash
FOLDER="/path/to/your/data/"
```

The script supports multiple datasets (AP8889, ROBUST04) with automatic path switching.

### 3. Start LLM Services (Optional)

For MonoT5-based strategies:
```bash
python src/main/python/mono_t5.py
```

For VLLM-based strategies:
```bash
# Configure and start your VLLM service
```

For Ollama-based strategies:
```bash
# Ensure Ollama is running with your chosen model
```

### 4. Run Experiments

#### Single Run
```bash
java -cp target/prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar \
  org.irlab.prfllm.searcher.TRECSearcherLucene \
  --index_path /path/to/index \
  --topics_path /path/to/topics \
  --qrels_path /path/to/qrels \
  --trec_run_folder /path/to/output \
  --rerank_method prf \
  --rf_strategy VLLM \
  --rerank_depth 100 \
  -e 20 \
  --lambda 0.7
```

#### Grid Search
```bash
cd src/main/scripts
./run_grid_search.sh [dataset] [--skip-rerank]
```

Options:
- `dataset`: ap8889 or robust04 (default: current dataset in config)
- `--skip-rerank`: Skip MonoT5 reranker experiments

The grid search automatically:
- Runs baseline LMDirichlet retrieval
- Tests multiple PRF strategies (VLLM, VLLM-PROB, etc.)
- Sweeps over parameters: depth (k), expansion terms (e), lambda (λ)
- Skips already completed experiments (resumable)

### 5. Analyze Results

```bash
cd src/main/scripts
./analyze_grid_results.sh [dataset]
```

This script:
1. Evaluates all runs with `trec_eval`
2. Generates summary TSV files
3. Creates comprehensive Markdown report
4. Generates visualizations (plots)

Outputs:
- Summary files: `/path/to/data/grid_results/{collection}/summary_*.tsv`
- Report: `/path/to/data/grid_results/{collection}/GRID_SEARCH_REPORT.md`
- Plots: `/path/to/data/grid_results/{collection}/plots/`

## 📊 Analysis Tools

### Generate Report
```bash
python src/main/python/generate_report.py [collection_name]
```

Creates a comprehensive Markdown report with:
- Best configurations by strategy
- Parameter analysis (lambda, e, depth)
- Performance comparisons
- Top-N configurations

### Generate Visualizations
```bash
python src/main/python/visualize_grid_results.py [collection_name]
```

Creates plots for:
- Strategy comparisons
- Parameter impact analysis (lambda, e, depth)
- Heatmaps for parameter interactions
- Performance trends

## 🔧 Configuration

### Dataset Configuration (`dataset_config.sh`)

Centralized configuration for:
- Dataset paths (index, topics, qrels)
- Output directories (runs, results, cache)
- Grid search parameters
- Model parameters

Switch between datasets:
```bash
source dataset_config.sh
switch_dataset robust04
```

### Grid Search Parameters

Default parameter ranges (configurable in `dataset_config.sh`):
- **Depths (k)**: [5, 10, 25, 50, 75, 100]
- **Expansion terms (e)**: [5, 10, 15, 20, 25, 30]
- **Lambda (λ)**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- **RF Strategies**: [VLLM, VLLM-PROB, MONOT5, MONOT5-PROB, etc.]

### Cache System

The system uses collection-specific caches to avoid redundant LLM queries:
- MonoT5 cache: `{cache_dir}/t5_cache.tsv`
- VLLM cache: `{cache_dir}/vllm_cache.tsv`
- Ollama cache: `{cache_dir}/ollama_cache_{model}.tsv`

Caches are automatically loaded and saved across runs.

## 📁 Project Structure

```
prf-llm/
├── src/
│   └── main/
│       ├── java/org/irlab/prfllm/
│       │   ├── indexer/          # Lucene indexing
│       │   └── searcher/         # Search and PRF
│       │       ├── rf/           # Relevance feedback models (RM3)
│       │       ├── smoothing/    # Smoothing methods
│       │       └── util/         # Scorers and utilities
│       ├── python/
│       │   ├── mono_t5.py              # MonoT5 service
│       │   ├── generate_report.py      # Report generator
│       │   └── visualize_grid_results.py # Visualization
│       ├── scripts/
│       │   ├── dataset_config.sh       # Shared configuration
│       │   ├── run_grid_search.sh      # Grid search runner
│       │   └── analyze_grid_results.sh # Analysis pipeline
│       └── resources/
│           └── log4j2.xml
├── target/                       # Compiled JARs
├── pom.xml                       # Maven configuration
└── README.md                     # This file
```

## 🔬 Supported RF Strategies

### LLM-Based Strategies

1. **MONOT5**: Binary filtering using MonoT5 relevance classification
2. **MONOT5-PROB**: Weighted by MonoT5 relevance probabilities
3. **VLLM**: Binary filtering using VLLM-served models
4. **VLLM-PROB**: Weighted by VLLM relevance probabilities
5. **OLLAMA**: Binary filtering using Ollama-served models

### Baselines

1. **PRF**: Traditional blind pseudo-relevance feedback (no LLM)
2. **ORACLE**: Uses ground-truth qrels (upper bound, all relevant docs)
3. **ORACLE-K**: Uses top-k ground-truth relevant docs (upper bound)

### Reranking

- **MonoT5 Reranker**: Direct document reranking without PRF

## 📈 Evaluation Metrics

The system evaluates using standard TREC metrics:
- **MAP** (Mean Average Precision)
- **P@10** (Precision at 10)
- **nDCG@100** (Normalized Discounted Cumulative Gain at 100)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 👥 Authors

- **David Otero** - Information Retrieval Lab, University of A Coruña
- **Javier Parapar** - Information Retrieval Lab, University of A Coruña

## 🏛️ Institution

Information Retrieval Lab  
CITIC Research Center  
University of A Coruña, Spain

## 📧 Contact

For questions or issues, please open an issue or contact the authors.

## 🙏 Acknowledgments

This work was supported by [funding sources to be added].

Special thanks to:
- The Lucene team for the excellent IR framework
- HuggingFace for the transformers library
- The TREC community for test collections and evaluation tools
