#!/bin/bash

# Script to evaluate and analyze grid search results
# Requires rec_eval to be installed
#
# Usage:
#   ./analyze_grid_results.sh [dataset]
#   
# Examples:
#   ./analyze_grid_results.sh         # Use default dataset
#   ./analyze_grid_results.sh ap8889  # Use AP8889 dataset
#   ./analyze_grid_results.sh robust04 # Use ROBUST04 dataset

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

# Check if rec_eval is available
if ! command -v rec_eval &> /dev/null; then
    echo -e "${YELLOW}Warning: rec_eval not found in PATH${NC}"
    echo "Please install rec_eval or ensure it's in your PATH"
    exit 1
fi

# Validate paths using shared validation function
if ! validate_paths; then
    echo -e "${RED}Configuration validation failed.${NC}"
    echo "Please check that all required files exist for the current dataset configuration."
    echo ""
    echo "Current configuration:"
    show_config
    echo ""
    echo "You may need to:"
    echo "1. Check that the dataset files exist in the expected locations"
    echo "2. Switch to a different dataset using: source dataset_config.sh && switch_dataset <name>"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/per_query"

# Show current configuration
show_config

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Grid Search Results Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Discover RF strategies dynamically from run file names
echo "Discovering RF strategies from run files..."
RF_STRATEGIES=()
if [ -d "$RUN_FOLDER" ]; then
    # Extract unique RF strategies from filenames that match the pattern _rfStrategy-XXXX_rfModel
    while IFS= read -r strategy; do
        RF_STRATEGIES+=("$strategy")
    done < <(find "$RUN_FOLDER" -name "*_rfStrategy-*_rfModel*" -type f | \
             sed 's/.*_rfStrategy-\([^_]*\).*/\1/' | \
             sort -u)
fi

echo "Detected RF strategies: ${RF_STRATEGIES[@]:-none}"
echo ""

# Initialize summary files
declare -A SUMMARY_FILES
SUMMARY_BASELINE="$RESULTS_DIR/summary_baseline.tsv"
SUMMARY_RERANK="$RESULTS_DIR/summary_monot5_rerank.tsv"

# Create summary files for each detected RF strategy
for strategy in "${RF_STRATEGIES[@]}"; do
    # Convert strategy name to lowercase and replace hyphens with underscores for file naming
    strategy_file=$(echo "$strategy" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    SUMMARY_FILES["$strategy"]="$RESULTS_DIR/summary_prf_${strategy_file}.tsv"
done

# Create headers for baseline and reranker
echo -e "run_name\tmap\tP@10\tndcg@100" > "$SUMMARY_BASELINE"
echo -e "run_name\tdepth\tmap\tP@10\tndcg@100" > "$SUMMARY_RERANK"

# Create headers for PRF strategies
for strategy in "${RF_STRATEGIES[@]}"; do
    echo -e "run_name\tdepth\te\tlambda\tmap\tP@10\tndcg@100" > "${SUMMARY_FILES[$strategy]}"
done

echo "Evaluating runs..."
echo ""

# Detect number of CPU cores
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Detected $NUM_CORES CPU cores, using parallel evaluation..."
echo ""

# Count total runs
TOTAL_RUNS=$(find "$RUN_FOLDER" -type f -name "LMDirichlet*" | wc -l)
echo "Total runs to evaluate: $TOTAL_RUNS"
echo ""

# Create temporary directory for partial results
TEMP_DIR="$RESULTS_DIR/temp_eval_$$"
mkdir -p "$TEMP_DIR"

# Function to evaluate a single run (exported for parallel execution)
evaluate_run() {
    local run_file=$1
    local QRELS_PATH=$2
    local TEMP_DIR=$3
    
    if [ ! -f "$run_file" ]; then
        return
    fi
    
    local run_name=$(basename "$run_file")
    
    # Run rec_eval and capture metrics (aggregate)
    local eval_output=$(rec_eval -m map -m P.10 -m ndcg_cut.100 "$QRELS_PATH" "$run_file" 2>/dev/null)
    
    # Extract metrics
    local map=$(echo "$eval_output" | grep "^map " | awk '{print $3}')
    local p10=$(echo "$eval_output" | grep "^P_10 " | awk '{print $3}')
    local ndcg100=$(echo "$eval_output" | grep "^ndcg_cut_100 " | awk '{print $3}')
    
    # Also get per-query MAP for robustness index calculation (saved to separate file)
    rec_eval -q -m map "$QRELS_PATH" "$run_file" 2>/dev/null | grep -v "^map\s*all" > "$TEMP_DIR/perquery_${run_name}.txt"

    # Parse run name to extract parameters and classify
    if [[ "$run_name" =~ ^LMDirichlet-[0-9]+_title_only$ ]]; then
        # Baseline (no reranking, no PRF)
        echo -e "$run_name\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/baseline.tsv"
        
    elif [[ "$run_name" =~ .*rerank-mono[tT]5_topK-([0-9]+) ]]; then
        # Check if it's not a PRF file
        local depth="${BASH_REMATCH[1]}"
        echo -e "$run_name\t$depth\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/rerank.tsv"
        
    elif [[ "$run_name" =~ _rfStrategy-([^_]+)_rfModel.*topK-([0-9]+)_lambda-([0-9.]+)_e-([0-9]+) ]]; then
        # PRF with detected strategy
        local strategy="${BASH_REMATCH[1]}"
        local depth="${BASH_REMATCH[2]}"
        local lambda="${BASH_REMATCH[3]}"
        local e="${BASH_REMATCH[4]}"
        
        # Convert strategy name for filename
        local strategy_file=$(echo "$strategy" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
        echo -e "$run_name\t$depth\t$e\t$lambda\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/prf_${strategy_file}.tsv"
    fi
}

# Export function and variables for parallel execution
export -f evaluate_run
export QRELS_PATH
export TEMP_DIR

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for evaluation..."
    echo ""
    find "$RUN_FOLDER" -type f -name "LMDirichlet*" | \
        parallel -j "$NUM_CORES" --progress --bar evaluate_run {} "$QRELS_PATH" "$TEMP_DIR"
else
    echo "Using xargs for parallel evaluation (install GNU parallel for better progress tracking)..."
    echo "  sudo apt install parallel  # Ubuntu/Debian"
    echo "  sudo yum install parallel  # CentOS/RHEL"
    echo ""
    
    # Create a progress tracker
    echo "0" > "$TEMP_DIR/counter.txt"
    
    # Modified evaluate function with progress
    evaluate_run_with_progress() {
        local run_file=$1
        local QRELS_PATH=$2
        local TEMP_DIR=$3
        local TOTAL_RUNS=$4
        
        # Call original evaluation
        evaluate_run "$run_file" "$QRELS_PATH" "$TEMP_DIR"
        
        # Update counter (using file lock)
        (
            flock -x 200
            local count=$(cat "$TEMP_DIR/counter.txt")
            count=$((count + 1))
            echo "$count" > "$TEMP_DIR/counter.txt"
            
            # Print progress every 10 runs or at start/end
            if [ $((count % 10)) -eq 0 ] || [ $count -eq 1 ] || [ $count -eq $TOTAL_RUNS ]; then
                echo "  Progress: $count/$TOTAL_RUNS runs evaluated ($(($count * 100 / $TOTAL_RUNS))%)"
            fi
        ) 200>"$TEMP_DIR/counter.lock"
    }
    
    export -f evaluate_run_with_progress
    export TOTAL_RUNS
    
    find "$RUN_FOLDER" -type f -name "LMDirichlet*" | \
        xargs -P "$NUM_CORES" -I {} bash -c 'evaluate_run_with_progress "$@"' _ {} "$QRELS_PATH" "$TEMP_DIR" "$TOTAL_RUNS"
    
    # Clean up progress files
    rm -f "$TEMP_DIR/counter.txt" "$TEMP_DIR/counter.lock"
fi

echo ""
echo "Consolidating results..."

# Move per-query results to permanent location
if compgen -G "$TEMP_DIR/perquery_*.txt" > /dev/null; then
    mv "$TEMP_DIR"/perquery_*.txt "$RESULTS_DIR/per_query/" 2>/dev/null || true
fi

# Consolidate results from temporary files
if [ -f "$TEMP_DIR/baseline.tsv" ]; then
    cat "$TEMP_DIR/baseline.tsv" >> "$SUMMARY_BASELINE"
fi

if [ -f "$TEMP_DIR/rerank.tsv" ]; then
    cat "$TEMP_DIR/rerank.tsv" >> "$SUMMARY_RERANK"
fi

# Consolidate PRF strategy results
for strategy in "${RF_STRATEGIES[@]}"; do
    strategy_file=$(echo "$strategy" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    temp_file="$TEMP_DIR/prf_${strategy_file}.tsv"
    if [ -f "$temp_file" ]; then
        cat "$temp_file" >> "${SUMMARY_FILES[$strategy]}"
    fi
done

# Clean up temporary directory
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}✓${NC} Evaluation completed"
echo ""

# Find best configurations
echo -e "${BLUE}=== Best Configurations ===${NC}"
echo ""

# Function to display best config
show_best() {
    local file=$1
    local strategy=$2
    local metric_col=$3
    local metric_name=$4
    
    if [ ! -f "$file" ] || [ $(wc -l < "$file") -le 1 ]; then
        return
    fi
    
    echo -e "${YELLOW}Best $strategy (by $metric_name):${NC}"
    tail -n +2 "$file" | sort -t$'\t' -k$metric_col -rn | head -1 | \
        awk -F'\t' -v cols=$(head -1 "$file" | awk -F'\t' '{print NF}') \
        '{if (cols == 4) printf "  MAP=%.4f, P@10=%.4f, ndcg@100=%.4f\n", $2, $3, $4; 
          else if (cols == 5) printf "  depth=%s -> MAP=%.4f, P@10=%.4f, ndcg@100=%.4f\n", $2, $3, $4, $5;
          else printf "  depth=%s, e=%s, lambda=%s -> MAP=%.4f, P@10=%.4f, ndcg@100=%.4f\n", $2, $3, $4, $5, $6, $7}'
}

# Baseline
if [ -f "$SUMMARY_BASELINE" ] && [ $(wc -l < "$SUMMARY_BASELINE") -gt 1 ]; then
    echo -e "${YELLOW}Baseline (LM Dirichlet):${NC}"
    tail -n +2 "$SUMMARY_BASELINE" | awk -F'\t' '{printf "  MAP=%.4f, P@10=%.4f, ndcg@100=%.4f\n", $2, $3, $4}'
    echo ""
fi

# MonoT5 Reranker
show_best "$SUMMARY_RERANK" "MonoT5 Reranker" "3" "MAP"
echo ""

# Dynamic PRF strategies
for strategy in "${RF_STRATEGIES[@]}"; do
    summary_file="${SUMMARY_FILES[$strategy]}"
    if [ -f "$summary_file" ] && [ $(wc -l < "$summary_file") -gt 1 ]; then
        # Create a human-readable strategy name
        strategy_display=$(echo "$strategy" | sed 's/-/ /g')
        show_best "$summary_file" "PRF + $strategy_display filter" "5" "MAP"
        echo ""
    fi
done

echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary files created:"
echo "  - $SUMMARY_BASELINE"
echo "  - $SUMMARY_RERANK"

# List dynamic PRF strategy files
for strategy in "${RF_STRATEGIES[@]}"; do
    summary_file="${SUMMARY_FILES[$strategy]}"
    if [ -f "$summary_file" ]; then
        echo "  - $summary_file"
    fi
done

echo ""
echo "Additional visualization options:"
echo "  - Excel/LibreOffice Calc (import TSV files)"
echo "  - R ggplot2"
echo "  - Custom Python scripts with pandas/matplotlib"
echo ""

# Generate visualizations FIRST (before report, so images exist!)
echo -e "${BLUE}Generating visualizations...${NC}"
PYTHON_VISUAL_PATH="../python/visualize_grid_results.py"
if [ -f "$PYTHON_VISUAL_PATH" ]; then
    COLLECTION_NAME=$(basename "$INDEX_PATH")
    python3 "$PYTHON_VISUAL_PATH" "$COLLECTION_NAME"
    echo ""
    echo -e "${GREEN}✓ Visualizations generated!${NC}"
    echo "Plots available at: $RESULTS_DIR/plots/"
    echo ""
else
    echo -e "${YELLOW}Warning: Visualization generator not found at $PYTHON_VISUAL_PATH${NC}"
    echo "To generate visualizations, run:"
    echo "  python3 ../python/visualize_grid_results.py $(basename "$INDEX_PATH")"
    echo ""
fi

# Generate comprehensive Markdown report (AFTER visualizations, so it can embed images!)
echo -e "${BLUE}Generating comprehensive Markdown report...${NC}"
PYTHON_REPORT_PATH="../python/generate_report.py"

if [ -f "$PYTHON_REPORT_PATH" ]; then
    # Extract collection name from INDEX (remove trailing path elements)
    COLLECTION_NAME=$(basename "$INDEX_PATH")
    python3 "$PYTHON_REPORT_PATH" "$COLLECTION_NAME"
    echo ""
    echo -e "${GREEN}✓ Markdown report generated!${NC}"
    echo "Report available at: $RESULTS_DIR/GRID_SEARCH_REPORT.md"
    
    # Generate HTML version with pandoc if available
    if command -v pandoc &> /dev/null; then
        echo ""
        echo -e "${BLUE}Converting report to HTML...${NC}"
        pandoc "$RESULTS_DIR/GRID_SEARCH_REPORT.md" \
            -f markdown \
            -t html \
            --standalone \
            --self-contained \
            --embed-resources \
            --metadata title="Grid Search Results Report" \
            --toc \
            --toc-depth=3 \
            -c <(echo "
                body { max-width: 1400px; margin: 40px auto; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; line-height: 1.6; color: #333; }
                h1, h2, h3, h4 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 30px; }
                h4 { border-bottom: 1px solid #95a5a6; font-size: 1.1em; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                th { background-color: #3498db; color: white; padding: 12px; text-align: left; font-weight: 600; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background-color: #f5f5f5; }
                code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
                pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
                img { max-width: 70%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); display: block; }
                figure { text-align: center; margin: 30px 0; }
                figcaption { font-style: italic; color: #666; margin-top: 10px; }
                #TOC { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
                #TOC ul { list-style-type: none; }
                #TOC a { color: #2c3e50; }
                @media print { body { max-width: 100%; } img { max-width: 80%; page-break-inside: avoid; } }
            ") \
            -o "$RESULTS_DIR/GRID_SEARCH_REPORT.html"
        
        echo -e "${GREEN}✓ HTML report generated!${NC}"
        echo "HTML report available at: $RESULTS_DIR/GRID_SEARCH_REPORT.html"
        echo ""
        echo "Open in browser with:"
        echo "  xdg-open $RESULTS_DIR/GRID_SEARCH_REPORT.html"
    else
        echo -e "${YELLOW}Note: pandoc not found - HTML version not generated${NC}"
        echo "To install pandoc:"
        echo "  sudo apt install pandoc  # Ubuntu/Debian"
        echo "  sudo yum install pandoc  # CentOS/RHEL"
    fi
    echo ""
else
    echo -e "${YELLOW}Warning: Report generator not found at $PYTHON_REPORT_PATH${NC}"
    echo "To generate a comprehensive report, run:"
    echo "  python3 ../python/generate_report.py $(basename "$INDEX_PATH")"
    echo ""
fi

echo -e "${GREEN}✓ Complete analysis finished!${NC}"
