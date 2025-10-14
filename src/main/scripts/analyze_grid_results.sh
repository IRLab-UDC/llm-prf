#!/bin/bash
# Note: Not using 'set -e' because parallel returns exit code 2 if any job fails,
# but we want to continue with consolidation even if some evaluations fail

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

# Discover RF strategies AND query types dynamically from run file names
echo "Discovering RF strategies and query types from run files..."
declare -A STRATEGY_QUERY_COMBOS  # Associative array to store unique strategy+query combinations
if [ -d "$RUN_FOLDER" ]; then
    # Extract unique combinations of RF strategy + query type
    while IFS= read -r file; do
        # Extract strategy
        if [[ "$file" =~ _rfStrategy-([^_]+) ]]; then
            strategy="${BASH_REMATCH[1]}"
            
            # Extract query type (title_only, title_plus_narrative, title_plus_description)
            if [[ "$file" =~ _(title_only|title_plus_narrative|title_plus_description)_ ]]; then
                query_type="${BASH_REMATCH[1]}"
                
                # Store combination as "STRATEGY::QUERYTYPE"
                STRATEGY_QUERY_COMBOS["${strategy}::${query_type}"]=1
            fi
        fi
    done < <(find "$RUN_FOLDER" -name "*_rfStrategy-*_rfModel*" -type f)
fi

# Also discover query types for baseline and rerank files
declare -A BASELINE_QUERY_TYPES
declare -A RERANK_QUERY_TYPES

while IFS= read -r file; do
    filename=$(basename "$file")
    if [[ "$filename" =~ _(title_only|title_plus_narrative|title_plus_description)($|_) ]]; then
        query_type="${BASH_REMATCH[1]}"
        
        # Check if it's baseline or rerank
        if [[ "$filename" =~ ^LMDirichlet-[0-9]+_${query_type}$ ]]; then
            BASELINE_QUERY_TYPES["$query_type"]=1
        elif [[ "$filename" =~ rerank-mono[tT]5 ]]; then
            RERANK_QUERY_TYPES["$query_type"]=1
        fi
    fi
done < <(find "$RUN_FOLDER" -name "LMDirichlet*" -type f)

echo "Detected strategy+query combinations:"
for combo in "${!STRATEGY_QUERY_COMBOS[@]}"; do
    echo "  - $combo"
done
echo ""

# Initialize summary files
declare -A SUMMARY_FILES

# Create summary files for baseline (one per query type)
for query_type in "${!BASELINE_QUERY_TYPES[@]}"; do
    query_file=$(echo "$query_type" | tr '_' '-')
    summary_file="$RESULTS_DIR/summary_baseline_${query_file}.tsv"
    echo -e "run_name\tmap\tP@10\tndcg@100" > "$summary_file"
    SUMMARY_FILES["baseline::$query_type"]="$summary_file"
done

# Create summary files for rerank (one per query type)
for query_type in "${!RERANK_QUERY_TYPES[@]}"; do
    query_file=$(echo "$query_type" | tr '_' '-')
    summary_file="$RESULTS_DIR/summary_monot5_rerank_${query_file}.tsv"
    echo -e "run_name\tdepth\tmap\tP@10\tndcg@100" > "$summary_file"
    SUMMARY_FILES["rerank::$query_type"]="$summary_file"
done

# Create summary files for each strategy+query combination
for combo in "${!STRATEGY_QUERY_COMBOS[@]}"; do
    # Split combo by :: using bash parameter expansion
    strategy="${combo%%::*}"     # Everything before ::
    query_type="${combo##*::}"   # Everything after ::
    
    # Convert to filename-friendly format
    strategy_file=$(echo "$strategy" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    query_file=$(echo "$query_type" | tr '_' '-')
    
    summary_file="$RESULTS_DIR/summary_prf_${strategy_file}_${query_file}.tsv"
    echo -e "run_name\tdepth\te\tlambda\tmap\tP@10\tndcg@100" > "$summary_file"
    SUMMARY_FILES["$combo"]="$summary_file"
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

    # Extract query type from run name
    local query_type=""
    if [[ "$run_name" =~ _(title_only|title_plus_narrative|title_plus_description)($|_) ]]; then
        query_type="${BASH_REMATCH[1]}"
    else
        # Skip files without recognized query type
        return
    fi
    
    # Convert query type to filename format (underscores to hyphens)
    local query_type_file=$(echo "$query_type" | tr '_' '-')

    # Parse run name to extract parameters and classify
    # Check for baseline: LMDirichlet-<mu>_<query_type> (exact match, nothing else after)
    if [[ "$run_name" =~ ^LMDirichlet-[0-9]+_(title_only|title_plus_narrative|title_plus_description)$ ]]; then
        # Baseline (no reranking, no PRF)
        echo -e "$run_name\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/baseline_${query_type_file}.tsv"
        
    elif [[ "$run_name" =~ .*rerank-mono[tT]5_topK-([0-9]+) ]]; then
        # MonoT5 reranker
        local depth="${BASH_REMATCH[1]}"
        echo -e "$run_name\t$depth\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/rerank_${query_type_file}.tsv"
        
    elif [[ "$run_name" =~ _rfStrategy-([^_]+)_rfModel.*_lambda-([0-9.]+)_e-([0-9]+) ]]; then
        # PRF with detected strategy
        local strategy="${BASH_REMATCH[1]}"
        local lambda="${BASH_REMATCH[2]}"
        local e="${BASH_REMATCH[3]}"
        
        # Extract depth (topK) if present, otherwise use "all" for ORACLE
        local depth="all"
        if [[ "$run_name" =~ topK-([0-9]+) ]]; then
            depth="${BASH_REMATCH[1]}"
        fi
        
        # Convert strategy name for filename
        local strategy_file=$(echo "$strategy" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
        echo -e "$run_name\t$depth\t$e\t$lambda\t$map\t$p10\t$ndcg100" >> "$TEMP_DIR/prf_${strategy_file}_${query_type_file}.tsv"
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
        parallel -j "$NUM_CORES" --progress --bar evaluate_run {} "$QRELS_PATH" "$TEMP_DIR" || true
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
shopt -s nullglob
perquery_files=("$TEMP_DIR"/perquery_*.txt)
if [ ${#perquery_files[@]} -gt 0 ]; then
    mv "$TEMP_DIR"/perquery_*.txt "$RESULTS_DIR/per_query/" 2>/dev/null || true
fi
shopt -u nullglob

# Consolidate baseline results (one file per query type)
for query_type in "${!BASELINE_QUERY_TYPES[@]}"; do
    query_type_file=$(echo "$query_type" | tr '_' '-')
    temp_file="$TEMP_DIR/baseline_${query_type_file}.tsv"
    if [ -f "$temp_file" ]; then
        summary_file="${SUMMARY_FILES["baseline::$query_type"]}"
        cat "$temp_file" >> "$summary_file"
    fi
done

# Consolidate rerank results (one file per query type)
for query_type in "${!RERANK_QUERY_TYPES[@]}"; do
    query_type_file=$(echo "$query_type" | tr '_' '-')
    temp_file="$TEMP_DIR/rerank_${query_type_file}.tsv"
    if [ -f "$temp_file" ]; then
        summary_file="${SUMMARY_FILES["rerank::$query_type"]}"
        cat "$temp_file" >> "$summary_file"
    fi
done

# Consolidate PRF strategy+query results
for combo in "${!STRATEGY_QUERY_COMBOS[@]}"; do
    # Split combo by :: using bash parameter expansion
    strategy="${combo%%::*}"     # Everything before ::
    query_type="${combo##*::}"   # Everything after ::
    
    strategy_file=$(echo "$strategy" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    query_type_file=$(echo "$query_type" | tr '_' '-')
    temp_file="$TEMP_DIR/prf_${strategy_file}_${query_type_file}.tsv"
    
    if [ -f "$temp_file" ]; then
        summary_file="${SUMMARY_FILES[$combo]}"
        cat "$temp_file" >> "$summary_file"
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

# Show best configurations for each query type

# Baseline (per query type)
for query_type in "${!BASELINE_QUERY_TYPES[@]}"; do
    summary_file="${SUMMARY_FILES["baseline::$query_type"]}"
    if [ -f "$summary_file" ] && [ $(wc -l < "$summary_file") -gt 1 ]; then
        query_display=$(echo "$query_type" | sed 's/_/ /g')
        echo -e "${YELLOW}Baseline (LM Dirichlet) - $query_display:${NC}"
        tail -n +2 "$summary_file" | awk -F'\t' '{printf "  MAP=%.4f, P@10=%.4f, ndcg@100=%.4f\n", $2, $3, $4}'
        echo ""
    fi
done

# MonoT5 Reranker (per query type)
for query_type in "${!RERANK_QUERY_TYPES[@]}"; do
    summary_file="${SUMMARY_FILES["rerank::$query_type"]}"
    if [ -f "$summary_file" ] && [ $(wc -l < "$summary_file") -gt 1 ]; then
        query_display=$(echo "$query_type" | sed 's/_/ /g')
        show_best "$summary_file" "MonoT5 Reranker - $query_display" "3" "MAP"
        echo ""
    fi
done

# PRF strategies (per strategy+query combination)
for combo in "${!STRATEGY_QUERY_COMBOS[@]}"; do
    # Split combo by :: using bash parameter expansion
    strategy="${combo%%::*}"      # Everything before ::
    query_type="${combo##*::}"    # Everything after ::
    summary_file="${SUMMARY_FILES[$combo]}"
    
    if [ -f "$summary_file" ] && [ $(wc -l < "$summary_file") -gt 1 ]; then
        # Create human-readable names
        strategy_display=$(echo "$strategy" | sed 's/-/ /g')
        query_display=$(echo "$query_type" | sed 's/_/ /g')
        show_best "$summary_file" "PRF + $strategy_display filter - $query_display" "5" "MAP"
        echo ""
    fi
done

echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary files created:"

# List all created summary files
for key in "${!SUMMARY_FILES[@]}"; do
    summary_file="${SUMMARY_FILES[$key]}"
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
