# Denoising Pseudo-Relevance Feedback with Large Language Models

This repository contains the code needed to reproduce the experiments of:

- **Denoising Pseudo-Relevance Feedback with Large Language Models** (extended journal version, in submission to ACM TOIS).
- **LLM-Assisted Pseudo-Relevance Feedback**, D. Otero and J. Parapar, ECIR 2026 (*Advances in Information Retrieval*, LNCS, Springer, pp. 452–459, [doi:10.1007/978-3-032-21300-6_36](https://doi.org/10.1007/978-3-032-21300-6_36)).

Both are reproduced by the same pipeline described below; the journal paper's grid simply sweeps a superset of what the ECIR paper needs.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Directory Layout](#directory-layout)
3. [Step 1 — Prepare the Data](#step-1--prepare-the-data)
4. [Step 2 — Configure Paths](#step-2--configure-paths)
5. [Step 3 — Build the Java Component](#step-3--build-the-java-component)
6. [Step 4 — Set Up the Python Environment](#step-4--set-up-the-python-environment)
7. [Step 5 — Index the Collections](#step-5--index-the-collections)
8. [Step 6 — Start the LLM Services](#step-6--start-the-llm-services)
9. [Step 7 — Run the Grid Search (Training)](#step-7--run-the-grid-search-training)
10. [Step 8 — Evaluate Grid Search Results](#step-8--evaluate-grid-search-results)
11. [Step 9 — Evaluate on Test Topics](#step-9--evaluate-on-test-topics)
12. [Step 10 — Statistical Significance](#step-10--statistical-significance)
13. [Step 11 — Generate the Paper's LaTeX Tables](#step-11--generate-the-papers-latex-tables)
14. [Reproducing Only the ECIR Paper](#reproducing-only-the-ecir-paper)
15. [PRF Strategies and Term Filters](#prf-strategies-and-term-filters)
16. [Cache System](#cache-system)
17. [Troubleshooting](#troubleshooting)

---

## Requirements

### Hardware

- A machine with at least one NVIDIA GPU is required to run the LLM services (MonoT5 and/or Llama 3.1). Llama 3.1 8B requires approximately 16 GB of GPU memory.
- The grid search and evaluation scripts are CPU-bound and parallelise across all available cores.

### Software

| Tool | Version |
|---|---|
| Java JDK | 21 |
| Apache Maven | 3.x |
| Python | 3.10+ |
| `trec_eval` | any |

Install `trec_eval` from [https://github.com/usnistgov/trec_eval](https://github.com/usnistgov/trec_eval) and ensure the binary is on your `PATH`. No other external tools are required.

---

## Directory Layout

The system expects a single base directory (configured as `FOLDER` in `dataset_config.sh`) with the following structure. Create it and populate it with the collection files before running anything.

```
$FOLDER/
├── indices/          # Lucene indices (created by the indexer — Step 5)
├── topics/           # TREC topic files
├── qrels/             # TREC qrels files
├── runs/              # Output run files (created automatically)
├── grid_results/      # Training evaluation summaries (created automatically)
├── cache/             # LLM score/span caches (created automatically)
└── test_results/      # Test evaluation reports + LaTeX tables (created automatically)
```

Expected filenames for each dataset (as set in `dataset_config.sh`):

| Dataset | Training topics | Training qrels | Test topics | Test qrels |
|---|---|---|---|---|
| AP8889 | `topics.51-100` | `qrels.AP8889.51-100` | `topics.101-200` | `qrels_ap8889_101_200.txt` |
| ROBUST04 | `topics.301-350.trec.txt` | `qrels.robust04.300-450.601-700.trec.txt` | `topics.351-400.trec.txt` | `qrels.robust04.300-450.601-700.trec.txt` |
| DL19/DL20 | `topics.dl-19.trec` | `qrels.dl19-passage.nist.trec.txt` | `topics.dl-20.trec` | `qrels.dl20-passage.nist.trec.txt` |
| WT10G | `topics.451-500.trec.txt` | `qrels.trec9.main_web.451-500` | `topics.501-550` | `qrels.wt10g.501-550` |

Place these files under `$FOLDER/topics/` and `$FOLDER/qrels/`. The raw document collections (for indexing) can live anywhere — their path is passed directly to the indexer in Step 5.

---

## Step 1 — Prepare the Data

Download the document collections and evaluation resources for each dataset:

- **AP8889** and **ROBUST04**.
- **DL19/DL20**: the MS MARCO passage collection, with the DL-19/DL-20 NIST topics.
- **WT10G**.

Topics and qrels for all four collections are available from the [TREC website](https://trec.nist.gov).

```bash
export FOLDER="/path/to/your/data"
mkdir -p "$FOLDER"/{indices,topics,qrels,runs,grid_results,cache,test_results}
```

Copy your topic and qrels files into `$FOLDER/topics/` and `$FOLDER/qrels/` using the filenames from the table above.

---

## Step 2 — Configure Paths

Open `src/main/scripts/dataset_config.sh` and set `FOLDER` to the base directory from Step 1:

```bash
FOLDER="/path/to/your/data"
```

This is the **only path you need to change**. Every other path (index, runs, results, cache) is derived from it. The active dataset defaults to AP8889; switch it by passing the dataset name as an argument to the scripts in Steps 7–10 (`ap8889`, `robust04`, `dl19`, `wt10g`).

---

## Step 3 — Build the Java Component

```bash
mvn clean package
```

This produces `target/llmprf-1.0-jar-with-dependencies.jar`, used by every subsequent Java command. Maven downloads all dependencies (Lucene 10.3.0, Jackson, etc.) automatically.

---

## Step 4 — Set Up the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/main/python/requirements.txt
```

`vllm` requires CUDA and takes a few minutes to install. If you only intend to run MonoT5-based strategies, you can omit it and skip Step 6's VLLM service.

---

## Step 5 — Index the Collections

Each collection must be indexed once before running experiments:

```bash
java -jar target/llmprf-1.0-jar-with-dependencies.jar index \
    --dataset /path/to/ap8889/documents \
    --index "$FOLDER/indices/ap8889"

java -jar target/llmprf-1.0-jar-with-dependencies.jar index \
    --dataset /path/to/robust04/documents \
    --index "$FOLDER/indices/robust04"

java -jar target/llmprf-1.0-jar-with-dependencies.jar index \
    --dataset /path/to/wt10g/documents \
    --index "$FOLDER/indices/wt10g"

java -jar target/llmprf-1.0-jar-with-dependencies.jar index \
    --dataset /path/to/msmarco/collection.jsonl \
    --index "$FOLDER/indices/msmarco"
```

Indexing can take from minutes to a few hours depending on collection size, and only needs to be done once.

---

## Step 6 — Start the LLM Services

The Java grid search communicates with two optional Python services over HTTP. Start only the ones needed for the strategies you intend to run.

### MonoT5 service (MONOT5, MONOT5-PROB)

```bash
source .venv/bin/activate
python src/main/python/mono_t5.py
```

Starts a Flask server on `http://localhost:5000`, serving `castorini/monot5-base-msmarco` (downloaded from HuggingFace on first run) at a `/eval` endpoint.

### VLLM service

```bash
source .venv/bin/activate
python src/main/python/serve_vllm.py
```

Starts a FastAPI server on `http://localhost:8080`, serving `meta-llama/Llama-3.1-8B-Instruct` via vLLM with guided JSON decoding, exposing `/prob` (relevance judgement), `/spans` (SpanSelection), and `/judge_spans` (single-call filtering + SpanSelection). Llama weights must be downloadable by HuggingFace (set `HF_TOKEN` if the model is gated).

If the VLLM service runs on a different machine than the Java process, set `VLLM_HOST` (host:port, no scheme) before running the Java commands in Steps 7–10:

```bash
export VLLM_HOST=gpu-box:8080
```

It defaults to `localhost:8080`.

> **LLM score caching:** once a (query, document) pair has been scored, the result is written to a persistent TSV cache under `$FOLDER/cache/<dataset>/`. Later runs — including test evaluation — read from the cache instead of calling the service again, so the LLM services only need to be running during the first pass over each collection.

---

## Step 7 — Run the Grid Search (Training)

```bash
cd src/main/scripts
./run_grid_search.sh ap8889       # or robust04 / dl19 / wt10g
```

The script runs three phases in sequence:

1. **Baselines** — LMDirichlet (µ=2000, "QLD") and BM25, no reranking or expansion.
2. **MonoT5 reranking** — the top-$k$ documents reranked with MonoT5, for each baseline and each depth in the grid.
3. **PRF grid search** — for every (baseline × PRF strategy × term filter × RF model) combination in the curated list (see `RF_STRATEGY_TERMFILTER_COMBOS` in `dataset_config.sh`), a single JVM instance internally sweeps every `(depth, e, lambda)` triple. This is the most time-consuming phase.

To skip MonoT5 reranking (e.g. if that service is unavailable):

```bash
./run_grid_search.sh ap8889 --skip-rerank
```

**The grid search is resumable.** A configuration whose run file already exists is skipped, so you can safely stop and restart at any point.

Run files are written to `$FOLDER/runs/<dataset>/` with a naming convention that encodes every parameter, e.g.:

```
BM25_content_prf-true_rfStrategy-VLLM-NARR_rfModel-RM3_prfSmoothing-Additive-0.1000_topK-100_lambda-0.05_e-150_termFilter-VLLMSPANS2
```

---

## Step 8 — Evaluate Grid Search Results

```bash
cd src/main/scripts
./evaluate_grid_results.sh ap8889
```

This finds every run file under `$FOLDER/runs/<dataset>/`, evaluates each with `trec_eval` in parallel (background subshells, one per CPU core), and writes:

- `$FOLDER/grid_results/<dataset>/baseline.tsv`, `monot5.tsv`, `prf.tsv` — one row per configuration with MAP, P@10, NDCG@100.
- `$FOLDER/grid_results/<dataset>/per_query/` — per-query AP for every run (used later for the Robustness Index and significance testing).

These TSVs are the input `test_evaluation.py` uses to pick the best training hyperparameters for each method (Step 9).

---

## Step 9 — Evaluate on Test Topics

```bash
cd src/main/scripts
./evaluate_on_test.sh ap8889
```

If running on a machine that has the run files but not the Maven project, pass the JAR explicitly (or set `JAR_PATH`):

```bash
./evaluate_on_test.sh ap8889 --jar /path/to/llmprf-1.0-jar-with-dependencies.jar
```

This reads the best training hyperparameters for every curated method from `$FOLDER/grid_results/<dataset>/`, re-runs each on the held-out test topics, evaluates with `trec_eval`, computes the per-query Robustness Index against the BM25 baseline, and appends everything to a **shared, cross-dataset** file: `$FOLDER/test_results/all_results.json`. A human-readable report is also written to `$FOLDER/test_results/<dataset>/test_evaluation_report.txt`.

Repeat this step for every dataset you want in the final tables:

```bash
./evaluate_on_test.sh ap8889
./evaluate_on_test.sh robust04
./evaluate_on_test.sh dl19
./evaluate_on_test.sh wt10g
```

LLM caches populated during Step 7 are reused automatically, so the LLM services generally do not need to be running for this step.

---

## Step 10 — Statistical Significance

The paper's cumulative-summary table (Table with rows (a)–(i)) reports, for each collection, which methods each row significantly outperforms. Compute this once all four datasets have completed Step 9:

```bash
cd src/main/python
python3 compute_significance.py
```

This re-evaluates the relevant test run files with `trec_eval -q`, runs a paired Wilcoxon signed-rank test on per-query AP@1000 for every pair of the table's 9 rows within each collection, applies Benjamini–Hochberg correction within each collection's family of comparisons, and writes `$FOLDER/test_results/significance_table5.json`. `generate_latex_tables.sh` (Step 11) reads this file to annotate the cumulative-summary table; if it is missing, that one table is generated without significance superscripts.

---

## Step 11 — Generate the Paper's LaTeX Tables

Once Steps 9–10 have been run for all four datasets:

```bash
cd src/main/scripts
./generate_latex_tables.sh
```

This writes every LaTeX table reported in the paper to `$FOLDER/test_results/`. Table **contents** match the paper exactly; captions are intentionally left empty (`\caption{}`) so they can be filled in directly in the manuscript.

| File | Paper table |
|---|---|
| `table_filtering.tex` | Blind PRF + document-level filtering (Section 5, "Establishing the PRF and filtering gains") |
| `table_confidence_narrative.tex` | Probability weighting + topic narrative ablation |
| `table_spanselection.tex` | Passage-level evidence selection (SpanSelection) |
| `table_cumulative_summary.tex` | Cumulative summary, rows (a)–(i), with significance superscripts |
| `table_ri.tex` | Robustness Index (RM3 only) |
| `table_params.tex` | Tuned feedback-set size $k$ as filtering is introduced |
| `table_appendix_summary.tex` | Appendix: full matrix of every curated method (AP@1000 + NDCG@100) |
| `table_appendix_params.tex` | Appendix: tuned $k$, $e$, $\alpha$ for every row above |
| `table_appendix_ri.tex` | Appendix: Robustness Index for every row above |
| `table_terms_148.tex` | Qualitative case study: top expanded-query terms for topic 148 |

The last table is generated separately, by re-running query expansion for one topic across a handful of methods (cache-hit only, no new LLM calls needed if Steps 7–9 already covered that topic/collection):

```bash
./generate_latex_tables.sh --terms-only   # or --aggregate-only for everything else
```

The topic/method selection for the case-study table is configured via the `TERMS_EXAMPLES` array at the top of `generate_latex_tables.sh`.