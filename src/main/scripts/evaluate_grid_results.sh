#!/bin/bash

set -e

# Source shared dataset configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_config.sh"

# Parse command line arguments
if [ $# -ge 1 ]; then
    DATASET_ARG="$1"
    echo "Switching to dataset: $DATASET_ARG"
    if ! switch_dataset "$DATASET_ARG"; then
        exit 1
    fi
    echo ""
fi

# Check if trec_eval is available
if ! command -v trec_eval &> /dev/null; then
    echo -e "${RED}Error: trec_eval not found in PATH${NC}"
    echo "Please install trec_eval or ensure it's in your PATH"
    exit 1
fi

# Validate paths
if ! validate_paths; then
    echo -e "${RED}Configuration validation failed.${NC}"
    show_config
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/per_query"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Grid Search Results Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

show_config

echo "Evaluating runs..."
echo ""

# Count total runs
TOTAL_RUNS=$(find "$RUN_FOLDER" -type f \( -name "*LMDirichlet*" -o -name "*BM25*" \) | wc -l)
echo "Total runs to evaluate: $TOTAL_RUNS"
echo ""

# Create temporary directory for parallel processing
TEMP_DIR="$RESULTS_DIR/temp_eval_$$"
mkdir -p "$TEMP_DIR"

BASELINE_SUMMARY="baseline.tsv"
MONOT5_SUMMARY="monot5.tsv"
PRF_SUMMARY="prf.tsv"
echo -e "run_name\tbaseline\tmap\tP@10\tndcg@100" > "$TEMP_DIR/$BASELINE_SUMMARY"
echo -e "run_name\tbaseline\tdepth\tmap\tP@10\tndcg@100" > "$TEMP_DIR/$MONOT5_SUMMARY"
echo -e "run_name\tbaseline\tstrategy\trf_model\tdepth\te\tlambda\tmap\tP@10\tndcg@100" > "$TEMP_DIR/$PRF_SUMMARY"

echo "Evaluating runs..."
echo ""

# Count total runs
TOTAL_RUNS=$(find "$RUN_FOLDER" -type f -name "*.trec" -o -name "*-*" | grep -E "(LMDirichlet|BM25)" | wc -l)
echo "Total runs to evaluate: $TOTAL_RUNS"
echo ""

# Function to evaluate a single run
evaluate_run() {
    local run_file=$1
    local run_name=$(basename "$run_file")

    # Skip if not a valid run file
    if [ ! -f "$run_file" ]; then
        return
    fi

    # Run trec_eval and capture metrics
    local eval_output=$(trec_eval -m map -m P.10 -m ndcg_cut.100 "$QRELS_PATH" "$run_file" 2>/dev/null)
    
    # Extract metrics
    local map=$(echo "$eval_output" | grep "^map " | awk '{print $3}')
    local p10=$(echo "$eval_output" | grep "^P_10 " | awk '{print $3}')
    local ndcg100=$(echo "$eval_output" | grep "^ndcg_cut_100 " | awk '{print $3}')

    # Skip if no valid metrics
    if [ -z "$map" ] || [ -z "$p10" ] || [ -z "$ndcg100" ]; then
        return
    fi

    # Also get per-query MAP for analysis
    trec_eval -q -m map "$QRELS_PATH" "$run_file" 2>/dev/null | grep -v "^map\s*all" > "$TEMP_DIR/perquery_${run_name}.txt"

    # Determine baseline model
    local baseline=""
    if [[ "$run_name" =~ LMDirichlet ]]; then
        baseline="LMDirichlet"
    elif [[ "$run_name" =~ BM25 ]]; then
        baseline="BM25"
    else
        return  # Skip files without recognized baseline
    fi

    # Classify run type and extract parameters
    if [[ "$run_name" =~ ^(LMDirichlet-2000|BM25)_content$ ]]; then
        # Baseline run (no reranking, no PRF)
        echo -e "$run_name\t$baseline\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/$BASELINE_SUMMARY"
        
    elif [[ "$run_name" =~ _rerank-monoT5_topK-([0-9]+)$ ]]; then
        # MonoT5 reranking
        local depth="${BASH_REMATCH[1]}"
        echo -e "$run_name\t$baseline\t$depth\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/$MONOT5_SUMMARY"
        
    elif [[ "$run_name" =~ _prf-true_rfStrategy-([^_]+)_rfModel-([^_]+)_.*_lambda-([0-9.]+)_e-([0-9]+)$ ]]; then
        # PRF run (both ORACLE and regular)
        local strategy="${BASH_REMATCH[1]}"
        local rf_model="${BASH_REMATCH[2]}"
        local lambda="${BASH_REMATCH[3]}"
        local e="${BASH_REMATCH[4]}"
        
        # Extract depth if present (regular PRF), otherwise "ALL" (ORACLE)
        local depth="ALL"
        if [[ "$run_name" =~ _topK-([0-9]+)_ ]]; then
            depth="${BASH_REMATCH[1]}"
        fi
        
        echo -e "$run_name\t$baseline\t$strategy\t$rf_model\t$depth\t$e\t$lambda\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/$PRF_SUMMARY"
    fi
}

# Export function for parallel execution
export -f evaluate_run
export QRELS_PATH TEMP_DIR
export BASELINE_SUMMARY MONOT5_SUMMARY PRF_SUMMARY

# Use parallel processing
NUM_CORES=$(nproc 2>/dev/null || echo 4)
echo "Using $NUM_CORES CPU cores for parallel evaluation..."

find "$RUN_FOLDER" -type f | grep -E "(LMDirichlet|BM25)" | parallel -j "$NUM_CORES" --progress --bar evaluate_run {} || true

echo ""
echo "Consolidating results..."

# Consolidate results from temp files
[ -f "$TEMP_DIR/$BASELINE_SUMMARY" ] && cat "$TEMP_DIR/$BASELINE_SUMMARY" > "$RESULTS_DIR/$BASELINE_SUMMARY"
[ -f "$TEMP_DIR/$MONOT5_SUMMARY" ] && cat "$TEMP_DIR/$MONOT5_SUMMARY" > "$RESULTS_DIR/$MONOT5_SUMMARY"
[ -f "$TEMP_DIR/$PRF_SUMMARY" ] && cat "$TEMP_DIR/$PRF_SUMMARY" > "$RESULTS_DIR/$PRF_SUMMARY"

# Move per-query results
mv "$TEMP_DIR"/perquery_*.txt "$RESULTS_DIR/per_query/" 2>/dev/null || true

# Clean up
rm -rf "$TEMP_DIR"

echo -e "${GREEN}✓ Evaluation completed${NC}"
echo ""

# Display best configurations
echo -e "${BLUE}=== Best Configurations ===${NC}"
echo ""

# Function to show best result from a summary file
show_best() {
    local file=$1
    local title=$2
    local sort_col=$3
    
    if [ ! -f "$file" ] || [ $(wc -l < "$file") -le 1 ]; then
        echo -e "${YELLOW}$title: No results found${NC}"
        return
    fi
    
    echo -e "${YELLOW}$title:${NC}"
    tail -n +2 "$file" | sort -t$'\t' -k$sort_col -rn | head -5 | while IFS=$'\t' read -r fields; do
        echo "  $fields" | awk -F'\t' '{
            # Format based on number of fields
            if (NF == 5) printf "  %s (%s): MAP=%.4f, P@10=%.4f, nDCG@100=%.4f\n", $1, $2, $3, $4, $5
            else if (NF == 6) printf "  %s (%s, depth=%s): MAP=%.4f, P@10=%.4f, nDCG@100=%.4f\n", $1, $2, $3, $4, $5, $6
            else if (NF == 10) printf "  %s+%s (%s, d=%s, e=%s, λ=%s): MAP=%.4f, P@10=%.4f, nDCG@100=%.4f\n", $3, $4, $2, $5, $6, $7, $8, $9, $10
        }'
    done
    echo ""
}

show_best "$RESULTS_DIR/$BASELINE_SUMMARY" "Best Baseline Models" "3"
show_best "$RESULTS_DIR/$MONOT5_SUMMARY" "Best MonoT5 Reranking" "4"
show_best "$RESULTS_DIR/$PRF_SUMMARY" "Best PRF Configurations" "8"

echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary files created:"
echo "  - $RESULTS_DIR/$BASELINE_SUMMARY"
echo "  - $RESULTS_DIR/$MONOT5_SUMMARY"
echo "  - $RESULTS_DIR/$PRF_SUMMARY"
echo "  - Per-query results: $RESULTS_DIR/per_query/"
echo ""

# Try to generate visualizations and report
PYTHON_VISUAL_PATH="$SCRIPT_DIR/../python/visualize_grid_results.py"
if [ -f "$PYTHON_VISUAL_PATH" ]; then
    echo -e "${BLUE}Generating visualizations...${NC}"
    COLLECTION_NAME=$(basename "$INDEX_PATH")
    python3 "$PYTHON_VISUAL_PATH" "$COLLECTION_NAME" 2>/dev/null || echo -e "${YELLOW}Warning: Could not generate visualizations${NC}"
fi

PYTHON_REPORT_PATH="$SCRIPT_DIR/../python/generate_grid_report.py"
if [ -f "$PYTHON_REPORT_PATH" ]; then
    echo -e "${BLUE}Generating report...${NC}"
    COLLECTION_NAME=$(basename "$INDEX_PATH")
    python3 "$PYTHON_REPORT_PATH" "$COLLECTION_NAME" 2>/dev/null || echo -e "${YELLOW}Warning: Could not generate report${NC}"
fi

echo -e "${GREEN}✓ Analysis completed!${NC}"