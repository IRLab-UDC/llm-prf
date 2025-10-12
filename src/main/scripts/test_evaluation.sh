#!/bin/bash

# test_evaluation.sh
# 
# Evaluates all PRF strategies on TEST data using the best parameters found during training.
# 
# Process:
# 1. Reads best parameters from grid_results/training
# 2. Runs each strategy with those parameters on TEST topics
# 3. Evaluates results with trec_eval
# 4. Calculates Robustness Index
# 5. Performs Wilcoxon signed-rank tests for significance
# 6. Generates comprehensive report
#
# Usage:
#   ./test_evaluation.sh <dataset>
# 
# Example:
#   ./test_evaluation.sh ap8889

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source dataset configuration
source "${SCRIPT_DIR}/dataset_config.sh"

# Check if dataset argument provided
if [ $# -gt 0 ]; then
    switch_dataset "$1"
fi

# Validate test topics and qrels are configured
if [ -z "$TOPICS_TEST" ] || [ -z "$QRELS_TEST" ]; then
    echo -e "${RED}Error: Test topics and qrels not configured for ${INDEX}${NC}"
    echo "Please add TOPICS_TEST and QRELS_TEST to dataset_config.sh"
    exit 1
fi

# Construct test paths
TOPICS_TEST_PATH="${FOLDER}/topics/${TOPICS_TEST}"
QRELS_TEST_PATH="${FOLDER}/topics/${QRELS_TEST}"
TEST_RUN_FOLDER="${FOLDER}/runs/${INDEX}_test"
TEST_RESULTS_DIR="${FOLDER}/test_results/${INDEX}"

# Validate test paths exist
if [ ! -f "$TOPICS_TEST_PATH" ]; then
    echo -e "${RED}Error: Test topics file not found: $TOPICS_TEST_PATH${NC}"
    exit 1
fi

if [ ! -f "$QRELS_TEST_PATH" ]; then
    echo -e "${RED}Error: Test qrels file not found: $QRELS_TEST_PATH${NC}"
    exit 1
fi

# Create output directories
mkdir -p "${TEST_RUN_FOLDER}"
mkdir -p "${TEST_RESULTS_DIR}"

# Python scripts
PYTHON_SCRIPT_DIR="${SCRIPT_DIR}/../python"

# JAR path (absolute)
JAR_PATH_ABS="${SCRIPT_DIR}/../../../target/prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar"

if [ ! -f "$JAR_PATH_ABS" ]; then
    echo -e "${RED}Error: JAR file not found: $JAR_PATH_ABS${NC}"
    echo "Please run 'mvn package' first"
    exit 1
fi

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}TEST EVALUATION - ${INDEX}${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Training Grid Results: ${RESULTS_DIR}"
echo "  Test Topics: ${TOPICS_TEST_PATH}"
echo "  Test Qrels: ${QRELS_TEST_PATH}"
echo "  Test Runs Output: ${TEST_RUN_FOLDER}"
echo "  Test Results Output: ${TEST_RESULTS_DIR}"
echo ""

# Function to extract best parameters from training results
extract_best_params() {
    local strategy="$1"
    
    # Map strategy name to summary file
    local summary_file=""
    case "$strategy" in
        "PRF")
            summary_file="${RESULTS_DIR}/summary_prf_prf.tsv"
            ;;
        "MONOT5")
            summary_file="${RESULTS_DIR}/summary_prf_monot5.tsv"
            ;;
        "MONOT5-PROB")
            summary_file="${RESULTS_DIR}/summary_prf_monot5_prob.tsv"
            ;;
        "MONOT5_RERANKER")
            summary_file="${RESULTS_DIR}/summary_monot5_rerank.tsv"
            ;;
        "VLLM")
            summary_file="${RESULTS_DIR}/summary_prf_vllm.tsv"
            ;;
        "VLLM-PROB")
            summary_file="${RESULTS_DIR}/summary_prf_vllm_prob.tsv"
            ;;
        "ORACLE")
            summary_file="${RESULTS_DIR}/summary_prf_oracle.tsv"
            ;;
        "ORACLE-K")
            summary_file="${RESULTS_DIR}/summary_prf_oracle_k.tsv"
            ;;
        *)
            echo -e "${YELLOW}Warning: Unknown strategy ${strategy}${NC}"
            return 1
            ;;
    esac
    
    if [ ! -f "$summary_file" ]; then
        echo -e "${YELLOW}Warning: Training results not found: $summary_file${NC}"
        return 1
    fi
    
    # Check if this is a reranker (different TSV format)
    if [[ "$strategy" == *"RERANKER"* ]]; then
        # Reranker format: run_name depth map P@10 ndcg@100
        # Skip header, sort by MAP (column 3), get best
        local best_row=$(tail -n +2 "$summary_file" | sort -t$'\t' -k3 -rn | head -1)
        
        if [ -z "$best_row" ]; then
            echo -e "${YELLOW}Warning: No results in ${summary_file}${NC}"
            return 1
        fi
        
        local depth=$(echo "$best_row" | awk '{print $2}')
        local train_map=$(echo "$best_row" | awk '{print $3}')
        
        echo "${depth},N/A,N/A,${train_map}"
    else
        # PRF format: run_name depth e lambda map P@10 ndcg@100
        # Skip header, sort by MAP (column 5), get best
        local best_row=$(tail -n +2 "$summary_file" | sort -t$'\t' -k5 -rn | head -1)
        
        if [ -z "$best_row" ]; then
            echo -e "${YELLOW}Warning: No results in ${summary_file}${NC}"
            return 1
        fi
        
        local depth=$(echo "$best_row" | awk '{print $2}')
        local e=$(echo "$best_row" | awk '{print $3}')
        local lambda=$(echo "$best_row" | awk '{print $4}')
        local train_map=$(echo "$best_row" | awk '{print $5}')
        
        echo "${depth},${e},${lambda},${train_map}"
    fi
}

# Function to run a single configuration on test data
run_test_configuration() {
    local strategy="$1"
    local depth="$2"
    local e="$3"
    local lambda="$4"
    # Check if this is a reranker strategy
    if [[ "$strategy" == *"RERANKER"* ]]; then
        # Reranker: only uses depth, no PRF parameters
        echo -e "${GREEN}  → Running ${strategy} (depth=${depth})${NC}" >&2
        
        # Build command for reranker (no PRF, just reranking)
        local cmd="java -cp \"${JAR_PATH_ABS}\" org.irlab.prfllm.searcher.TRECSearcherLucene"
        cmd+=" --index_path \"${INDEX_PATH}\""
        cmd+=" --topics_path \"${TOPICS_TEST_PATH}\""
        cmd+=" --trec_run_folder \"${TEST_RUN_FOLDER}\""
        cmd+=" --qrels_path \"${QRELS_TEST_PATH}\""
        cmd+=" --search_by title_only"
        cmd+=" --rerank_method monot5"
        cmd+=" --rerank_depth ${depth}"
        cmd+=" --cache_dir \"${CACHE_DIR}\""
        cmd+=" --mu ${MU}"
        
        # The Java program will create: LMDirichlet-{MU}_title_only_rerank-monoT5_topK-{depth}
        local output_file="${TEST_RUN_FOLDER}/LMDirichlet-${MU}_title_only_rerank-monoT5_topK-${depth}"
    else
        # PRF strategy: uses depth, e, lambda
        echo -e "${GREEN}  → Running ${strategy} (depth=${depth}, e=${e}, λ=${lambda})${NC}" >&2
        
        # Format lambda with 2 decimals (e.g., 0.20 -> 0.20)
        local lambda_formatted=$(printf "%.2f" "$lambda")
        
        # Build command for PRF strategy
        local cmd="java -cp \"${JAR_PATH_ABS}\" org.irlab.prfllm.searcher.TRECSearcherLucene"
        cmd+=" --index_path \"${INDEX_PATH}\""
        cmd+=" --topics_path \"${TOPICS_TEST_PATH}\""
        cmd+=" --trec_run_folder \"${TEST_RUN_FOLDER}\""
        cmd+=" --qrels_path \"${QRELS_TEST_PATH}\""
        cmd+=" --search_by title_only"
        cmd+=" --mu ${MU}"
        cmd+=" --rerank_method prf"
        cmd+=" --prf_strategy ${strategy}"
        cmd+=" --rf_model RM3"
        cmd+=" --rerank_depth ${depth}"
        cmd+=" -e ${e}"
        cmd+=" --lambda ${lambda}"
        cmd+=" --cache_dir \"${CACHE_DIR}\""
        
        # Add strategy-specific parameters for PRF
        case "$strategy" in
            MONOT5*|VLLM*|ORACLE*|PRF*)
                cmd+=" --prf_smoothing Additive"
                cmd+=" --prf_smoothing_param 0.1"
                ;;
        esac
        
        # The Java program will create: LMDirichlet-{MU}_title_only_prf-true_rfStrategy-{STRATEGY}_rfModel-RM3_prfSmoothing-Additive-0.1000_topK-{depth}_lambda-{lambda}_e-{e}
        local output_file="${TEST_RUN_FOLDER}/LMDirichlet-${MU}_title_only_prf-true_rfStrategy-${strategy}_rfModel-RM3_prfSmoothing-Additive-0.1000_topK-${depth}_lambda-${lambda_formatted}_e-${e}"
    fi
    
    # Execute
    if eval "$cmd" > "${output_file}.log" 2>&1; then
        echo "$output_file"
        return 0
    else
        echo -e "${RED}    ✗ Failed${NC}" >&2
        echo "    Last 10 lines of log:" >&2
        tail -10 "${output_file}.log" | sed 's/^/      /' >&2
        return 1
    fi
}

# Function to evaluate a run file
evaluate_run() {
    local run_file="$1"
    local output_file="$2"
    
    if [ ! -f "$run_file" ]; then
        echo -e "${RED}Error: Run file not found: $run_file${NC}" >&2
        return 1
    fi
    
    if ! rec_eval -m map -m P.10 -m ndcg_cut.100 "${QRELS_TEST_PATH}" "${run_file}" > "${output_file}" 2>&1; then
        echo -e "${RED}Error: rec_eval failed for $run_file${NC}" >&2
        return 1
    fi
}

# Function to evaluate per-query (for Robustness Index)
evaluate_per_query() {
    local run_file="$1"
    local output_file="$2"
    
    if [ ! -f "$run_file" ]; then
        echo -e "${RED}Error: Run file not found: $run_file${NC}" >&2
        return 1
    fi
    
    if ! rec_eval -q -m map "${QRELS_TEST_PATH}" "${run_file}" > "${output_file}" 2>&1; then
        echo -e "${RED}Error: rec_eval failed for $run_file${NC}" >&2
        return 1
    fi
}

# Function to calculate Robustness Index from per-query results
calculate_ri() {
    local test_file="$1"
    local baseline_file="$2"
    
    # Use external Python script
    python3 "${SCRIPT_DIR}/../python/calculate_robustness_index.py" \
        "$test_file" "$baseline_file" --format csv
}

# Main execution
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}STEP 1: Extract Best Parameters from Training${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Strategies to evaluate
STRATEGIES=("PRF" "MONOT5" "MONOT5-PROB" "MONOT5_RERANKER" "VLLM" "VLLM-PROB" "ORACLE" "ORACLE-K")

declare -A BEST_PARAMS
declare -A TRAIN_MAP

for strategy in "${STRATEGIES[@]}"; do
    echo -e "${GREEN}Extracting best parameters for ${strategy}...${NC}"
    
    params=$(extract_best_params "$strategy")
    if [ $? -eq 0 ]; then
        BEST_PARAMS[$strategy]="$params"
        
        depth=$(echo "$params" | cut -d',' -f1)
        e=$(echo "$params" | cut -d',' -f2)
        lambda=$(echo "$params" | cut -d',' -f3)
        train_map=$(echo "$params" | cut -d',' -f4)
        
        TRAIN_MAP[$strategy]="$train_map"
        
        # Display parameters based on strategy type
        if [[ "$strategy" == *"RERANKER"* ]]; then
            echo -e "  Best params: depth=${depth} (Training MAP=${train_map})"
        else
            echo -e "  Best params: depth=${depth}, e=${e}, λ=${lambda} (Training MAP=${train_map})"
        fi
    else
        echo -e "${YELLOW}  Skipping ${strategy} (no training results)${NC}"
    fi
done

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}STEP 2: Run Configurations on Test Data${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

declare -A TEST_RUN_FILES
declare -A TEST_EVAL_FILES
declare -A TEST_PERQ_FILES

for strategy in "${STRATEGIES[@]}"; do
    if [ -z "${BEST_PARAMS[$strategy]}" ]; then
        continue
    fi
    
    params="${BEST_PARAMS[$strategy]}"
    depth=$(echo "$params" | cut -d',' -f1)
    e=$(echo "$params" | cut -d',' -f2)
    lambda=$(echo "$params" | cut -d',' -f3)
    
    echo -e "${GREEN}Testing ${strategy}...${NC}"
    run_file=$(run_test_configuration "$strategy" "$depth" "$e" "$lambda")
    status=$?
    echo "Run file: $run_file"
    if [ $status -eq 0 ]; then
        TEST_RUN_FILES[$strategy]="$run_file"
        
        # Evaluate
        eval_file="${TEST_RESULTS_DIR}/${strategy}_eval.txt"
        perq_file="${TEST_RESULTS_DIR}/${strategy}_perquery.txt"
        
        evaluate_run "$run_file" "$eval_file"
        evaluate_per_query "$run_file" "$perq_file"
        
        TEST_EVAL_FILES[$strategy]="$eval_file"
        TEST_PERQ_FILES[$strategy]="$perq_file"
        
        echo -e "${GREEN}    ✓ Evaluation completed${NC}"
    fi
done

echo ""

# Run baseline (no PRF) for Robustness Index calculation and comparison
echo -e "${GREEN}Running baseline (LM Dirichlet μ=${MU}) for comparison...${NC}"
# The Java program will create: LMDirichlet-{MU}_title_only
baseline_run="${TEST_RUN_FOLDER}/LMDirichlet-${MU}_title_only"
java -cp "${JAR_PATH_ABS}" org.irlab.prfllm.searcher.TRECSearcherLucene \
    --index_path "${INDEX_PATH}" \
    --topics_path "${TOPICS_TEST_PATH}" \
    --trec_run_folder "${TEST_RUN_FOLDER}" \
    --qrels_path "${QRELS_TEST_PATH}" \
    --search_by title_only \
    --rerank_method none \
    --prf_strategy none \
    --mu ${MU} \
    > /dev/null 2>&1

# Evaluate baseline
baseline_eval="${TEST_RESULTS_DIR}/BASELINE_eval.txt"
baseline_perq="${TEST_RESULTS_DIR}/BASELINE_perquery.txt"
evaluate_run "$baseline_run" "$baseline_eval"
evaluate_per_query "$baseline_run" "$baseline_perq"

# Extract baseline metrics
baseline_map=$(grep "^map " "$baseline_eval" | awk '{print $3}')
baseline_p10=$(grep "^P_10 " "$baseline_eval" | awk '{print $3}')
baseline_ndcg100=$(grep "^ndcg_cut_100 " "$baseline_eval" | awk '{print $3}')

echo -e "${GREEN}  Baseline MAP: ${baseline_map}${NC}"
echo -e "${GREEN}  Baseline P@10: ${baseline_p10}${NC}"
echo -e "${GREEN}  Baseline NDCG@100: ${baseline_ndcg100}${NC}"

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}STEP 3: Extract Metrics${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Display baseline first
echo -e "${GREEN}BASELINE (LM Dirichlet μ=${MU}):${NC}"
echo "  MAP: ${baseline_map}"
echo "  P@10: ${baseline_p10}"
echo "  NDCG@100: ${baseline_ndcg100}"
echo ""

declare -A TEST_MAP
declare -A TEST_P10
declare -A TEST_NDCG100
declare -A TEST_RI

for strategy in "${STRATEGIES[@]}"; do
    if [ -z "${TEST_EVAL_FILES[$strategy]}" ]; then
        continue
    fi
    
    eval_file="${TEST_EVAL_FILES[$strategy]}"
    perq_file="${TEST_PERQ_FILES[$strategy]}"
    
    # Extract metrics
    map=$(grep "^map " "$eval_file" | awk '{print $3}')
    p10=$(grep "^P_10 " "$eval_file" | awk '{print $3}')
    ndcg100=$(grep "^ndcg_cut_100 " "$eval_file" | awk '{print $3}')
    
    TEST_MAP[$strategy]="$map"
    TEST_P10[$strategy]="$p10"
    TEST_NDCG100[$strategy]="$ndcg100"
    
    # Calculate RI
    ri_result=$(calculate_ri "$perq_file" "$baseline_perq")
    ri=$(echo "$ri_result" | cut -d',' -f1)
    improved=$(echo "$ri_result" | cut -d',' -f2)
    hurt=$(echo "$ri_result" | cut -d',' -f3)
    
    TEST_RI[$strategy]="${ri} (↑${improved}/↓${hurt})"
    
    echo -e "${GREEN}${strategy}:${NC}"
    echo "  MAP: ${map}"
    echo "  P@10: ${p10}"
    echo "  NDCG@100: ${ndcg100}"
    echo "  RI: ${ri} (↑${improved}/↓${hurt})"
done

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}STEP 4: Statistical Significance Tests (Wilcoxon)${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Perform pairwise Wilcoxon tests using external Python script
echo "Pairwise Wilcoxon Signed-Rank Tests:"
echo "======================================================================"
printf "%-30s %-12s %-15s\n" "Comparison" "p-value" "Significant"
echo "======================================================================"

# First, compare each strategy against baseline
echo "# Comparisons: Strategy > Baseline (expecting improvement):"
for strategy in "${STRATEGIES[@]}"; do
    if [ -n "${TEST_PERQ_FILES[$strategy]}" ] && [ -f "${TEST_PERQ_FILES[$strategy]}" ]; then
        file="${TEST_PERQ_FILES[$strategy]}"
        
        # Call wilcoxon_test.py (strategy first, baseline second)
        result=$(python3 "${SCRIPT_DIR}/../python/wilcoxon_test.py" \
            "$file" "$baseline_perq" --format csv 2>/dev/null || echo "error,error,error,0")
        
        # Parse result: statistic,p_value,significant,n_common
        IFS=',' read -r stat p_value significant n_common <<< "$result"
        
        if [ "$significant" = "yes" ]; then
            sig_text="✓ Yes (p<0.05)"
        elif [ "$significant" = "no" ]; then
            sig_text="No"
        else
            sig_text="Error"
        fi
        
        printf "%-30s %-12s %-15s\n" "$strategy > BASELINE" "$p_value" "$sig_text"
    fi
done

echo "# Strategy-to-Strategy Comparisons:"

# Get list of strategies that have results
AVAILABLE_STRATEGIES=()
for strategy in "${STRATEGIES[@]}"; do
    if [ -n "${TEST_PERQ_FILES[$strategy]}" ] && [ -f "${TEST_PERQ_FILES[$strategy]}" ]; then
        AVAILABLE_STRATEGIES+=("$strategy")
    fi
done

# Perform pairwise comparisons between strategies
for ((i=0; i<${#AVAILABLE_STRATEGIES[@]}; i++)); do
    for ((j=i+1; j<${#AVAILABLE_STRATEGIES[@]}; j++)); do
        strategy1="${AVAILABLE_STRATEGIES[$i]}"
        strategy2="${AVAILABLE_STRATEGIES[$j]}"
        
        file1="${TEST_PERQ_FILES[$strategy1]}"
        file2="${TEST_PERQ_FILES[$strategy2]}"
        
        # Call wilcoxon_test.py
        result=$(python3 "${SCRIPT_DIR}/../python/wilcoxon_test.py" \
            "$file1" "$file2" --format csv 2>/dev/null || echo "error,error,error,0")
        
        # Parse result: statistic,p_value,significant,n_common
        IFS=',' read -r stat p_value significant n_common <<< "$result"
        
        if [ "$significant" = "yes" ]; then
            sig_text="✓ Yes (p<0.05)"
        elif [ "$significant" = "no" ]; then
            sig_text="No"
        else
            sig_text="Error"
        fi
        
        printf "%-30s %-12s %-15s\n" "$strategy1 > $strategy2" "$p_value" "$sig_text"
    done
done

echo "======================================================================"

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}STEP 4.5: Multiple Comparisons with BH Correction${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Prepare arguments for multiple comparison script
MC_ARGS=""
MC_ARGS+="BASELINE:${baseline_perq} "
for strategy in "${STRATEGIES[@]}"; do
    if [ -n "${TEST_PERQ_FILES[$strategy]}" ] && [ -f "${TEST_PERQ_FILES[$strategy]}" ]; then
        MC_ARGS+="${strategy}:${TEST_PERQ_FILES[$strategy]} "
    fi
done

# Run multiple comparisons with BH correction
echo "Running Benjamini-Hochberg FDR correction for multiple comparisons..."
echo ""
python3 "${SCRIPT_DIR}/../python/wilcoxon_multiple_comparisons.py" \
    ${MC_ARGS} --format table --alpha 0.05

# Save CSV version for report
MC_CSV="${TEST_RESULTS_DIR}/multiple_comparisons_bh.csv"
python3 "${SCRIPT_DIR}/../python/wilcoxon_multiple_comparisons.py" \
    ${MC_ARGS} --format csv --alpha 0.05 > "${MC_CSV}"

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}STEP 5: Generate Report${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Generate Markdown report
REPORT_FILE="${TEST_RESULTS_DIR}/TEST_EVALUATION_REPORT.md"

cat > "$REPORT_FILE" <<'REPORT_HEADER'
# Test Evaluation Report

**Dataset**: ${INDEX}
**Test Topics**: ${TOPICS_TEST}
**Test Qrels**: ${QRELS_TEST}
**Generated**: $(date '+%Y-%m-%d %H:%M:%S')

---

## Overview

This report presents the evaluation results on **TEST data** using the best parameters identified during training.

---

## Best Parameters from Training

The following parameters were selected based on the highest MAP scores on training data:

| Strategy | Depth | E | Lambda | Training MAP |
|----------|-------|---|--------|--------------|
REPORT_HEADER

# Expand variables in report
eval "cat > \"$REPORT_FILE\" <<'REPORT_HEADER'
# Test Evaluation Report

**Dataset**: ${INDEX}  
**Test Topics**: ${TOPICS_TEST}  
**Test Qrels**: ${QRELS_TEST}  
**Generated**: $(date '+%Y-%m-%d %H:%M:%S')

---

## Overview

This report presents the evaluation results on **TEST data** using the best parameters identified during training.

---

## Best Parameters from Training

The following parameters were selected based on the highest MAP scores on training data:

| Strategy | Depth | E | Lambda | Training MAP |
|----------|-------|---|--------|--------------|
REPORT_HEADER
"

for strategy in "${STRATEGIES[@]}"; do
    if [ -n "${BEST_PARAMS[$strategy]}" ]; then
        params="${BEST_PARAMS[$strategy]}"
        depth=$(echo "$params" | cut -d',' -f1)
        e=$(echo "$params" | cut -d',' -f2)
        lambda=$(echo "$params" | cut -d',' -f3)
        train_map="${TRAIN_MAP[$strategy]}"
        
        # Format based on strategy type
        if [[ "$strategy" == *"RERANKER"* ]]; then
            echo "| ${strategy} | ${depth} | - | - | ${train_map} |" >> "$REPORT_FILE"
        else
            echo "| ${strategy} | ${depth} | ${e} | ${lambda} | ${train_map} |" >> "$REPORT_FILE"
        fi
    fi
done

cat >> "$REPORT_FILE" <<EOF

---

## Test Results

### Baseline Performance

**LM Dirichlet (μ=${MU})** - Initial ranking without PRF:

| Metric | Value |
|--------|-------|
| MAP | ${baseline_map} |
| P@10 | ${baseline_p10} |
| NDCG@100 | ${baseline_ndcg100} |

---

### PRF Strategies Performance

| Strategy | MAP | P@10 | NDCG@100 | Robustness Index |
|----------|-----|------|----------|------------------|
EOF

for strategy in "${STRATEGIES[@]}"; do
    if [ -n "${TEST_MAP[$strategy]}" ]; then
        echo "| ${strategy} | ${TEST_MAP[$strategy]} | ${TEST_P10[$strategy]} | ${TEST_NDCG100[$strategy]} | ${TEST_RI[$strategy]} |" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" <<EOF

### Metrics Explanation

- **MAP**: Mean Average Precision
- **P@10**: Precision at rank 10
- **NDCG@100**: Normalized Discounted Cumulative Gain at rank 100
- **Robustness Index (RI)**: (queries improved - queries hurt) / total queries
  - ↑ = number of queries improved vs baseline
  - ↓ = number of queries hurt vs baseline

---

## Statistical Significance Tests

### Wilcoxon Signed-Rank Test Results

###cat >> "$REPORT_FILE" <<EOF

#### Strategy > Baseline Comparisons

Each strategy should improve over baseline (LM Dirichlet):

| Strategy | p-value | Significant (α=0.05) |
|----------|---------|----------------------|
EOF

# Compare each strategy against baseline
for strategy in "${STRATEGIES[@]}"; do
    if [ -n "${TEST_PERQ_FILES[$strategy]}" ] && [ -f "${TEST_PERQ_FILES[$strategy]}" ]; then
        file="${TEST_PERQ_FILES[$strategy]}"
        
        # Call wilcoxon_test.py (strategy first, baseline second)
        result=$(python3 "${SCRIPT_DIR}/../python/wilcoxon_test.py" \
            "$file" "$baseline_perq" --format csv 2>/dev/null || echo "error,error,error,0")
        
        # Parse result: statistic,p_value,significant,n_common
        IFS=',' read -r stat p_value significant n_common <<< "$result"
        
        if [ "$significant" = "yes" ]; then
            sig_text="✓ Yes"
        elif [ "$significant" = "no" ]; then
            sig_text="No"
        else
            sig_text="Error"
            p_value="N/A"
        fi
        
        echo "| ${strategy} > BASELINE | $p_value | $sig_text |" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" <<EOF

#### Strategy-to-Strategy Comparisons

| Comparison | p-value | Significant (α=0.05) |
|------------|---------|----------------------|
EOF

# Run Wilcoxon tests using external Python script and append to report
for ((i=0; i<${#AVAILABLE_STRATEGIES[@]}; i++)); do
    for ((j=i+1; j<${#AVAILABLE_STRATEGIES[@]}; j++)); do
        strategy1="${AVAILABLE_STRATEGIES[$i]}"
        strategy2="${AVAILABLE_STRATEGIES[$j]}"
        
        file1="${TEST_PERQ_FILES[$strategy1]}"
        file2="${TEST_PERQ_FILES[$strategy2]}"
        
        # Call wilcoxon_test.py
        result=$(python3 "${SCRIPT_DIR}/../python/wilcoxon_test.py" \
            "$file1" "$file2" --format csv 2>/dev/null || echo "error,error,error,0")
        
        # Parse result: statistic,p_value,significant,n_common
        IFS=',' read -r stat p_value significant n_common <<< "$result"
        
        if [ "$significant" = "yes" ]; then
            sig_text="✓ Yes"
        elif [ "$significant" = "no" ]; then
            sig_text="No"
        else
            sig_text="Error"
            p_value="N/A"
        fi
        
        echo "| $strategy1 > $strategy2 | $p_value | $sig_text |" >> "$REPORT_FILE"
    done
done

cat >> "$REPORT_FILE" <<EOF

**Interpretation:**
- p-value < 0.05: Statistically significant difference
- p-value ≥ 0.05: No statistically significant difference

---

### Multiple Comparisons with Benjamini-Hochberg Correction

When performing multiple hypothesis tests, the probability of finding at least one significant result by chance increases (family-wise error rate). The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) to account for multiple comparisons.

| Comparison | p-value | p-adjusted (BH) | Uncorrected | BH Corrected |
|------------|---------|-----------------|-------------|--------------|
EOF

# Read BH correction results from CSV and add to report
if [ -f "${MC_CSV}" ]; then
    tail -n +2 "${MC_CSV}" | while IFS=',' read -r comparison p_value p_adjusted sig_uncorrected sig_corrected; do
        # Format significance indicators
        uncorr_text=$([ "$sig_uncorrected" = "yes" ] && echo "✓ Yes" || echo "No")
        corr_text=$([ "$sig_corrected" = "yes" ] && echo "✓ Yes" || echo "No")
        
        echo "| ${comparison} | ${p_value} | ${p_adjusted} | ${uncorr_text} | ${corr_text} |" >> "$REPORT_FILE"
    done
fi

cat >> "$REPORT_FILE" <<EOF

**Interpretation:**
- **p-value**: Raw p-value from Wilcoxon signed-rank test
- **p-adjusted (BH)**: p-value adjusted using Benjamini-Hochberg FDR correction
- **Uncorrected**: Significant at α=0.05 without correction
- **BH Corrected**: Significant at α=0.05 after FDR correction

The BH correction is more conservative and reduces false positives when making multiple comparisons. Results that remain significant after correction provide stronger evidence of true differences.

---

## Training vs Test Performance

Comparison of MAP scores between training and test sets:

| Strategy | Training MAP | Test MAP | Δ (Test - Train) |
|----------|--------------|----------|------------------|
EOF

for strategy in "${STRATEGIES[@]}"; do
    if [ -n "${TRAIN_MAP[$strategy]}" ] && [ -n "${TEST_MAP[$strategy]}" ]; then
        train="${TRAIN_MAP[$strategy]}"
        test="${TEST_MAP[$strategy]}"
        delta=$(python3 -c "print(f'{float('$test') - float('$train'):.4f}')")
        echo "| ${strategy} | ${train} | ${test} | ${delta} |" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" <<EOF

**Notes:**
- Positive Δ indicates better performance on test than training
- Negative Δ may indicate overfitting to training data

---

## Files Generated

- **Test Runs**: \`${TEST_RUN_FOLDER}/\`
- **Evaluation Results**: \`${TEST_RESULTS_DIR}/\`
- **Report**: \`${REPORT_FILE}\`

---

## Conclusion

This evaluation provides insights into the generalization performance of different PRF strategies on unseen test data.

EOF

echo -e "${GREEN}✓ Report generated: ${REPORT_FILE}${NC}"

# Generate HTML version using pandoc if available
if command -v pandoc &> /dev/null; then
    HTML_REPORT="${TEST_RESULTS_DIR}/TEST_EVALUATION_REPORT.html"
    pandoc "$REPORT_FILE" -o "$HTML_REPORT" \
        --standalone \
        --metadata title="Test Evaluation Report - ${INDEX}" \
        --css=https://cdn.jsdelivr.net/npm/github-markdown-css@5.2.0/github-markdown.min.css \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ HTML report generated: ${HTML_REPORT}${NC}"
    fi
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "${GREEN}Test evaluation completed successfully!${NC}"
echo ""
echo "Reports generated:"
echo "  - Markdown: ${REPORT_FILE}"
if [ -f "${TEST_RESULTS_DIR}/TEST_EVALUATION_REPORT.html" ]; then
    echo "  - HTML: ${TEST_RESULTS_DIR}/TEST_EVALUATION_REPORT.html"
fi
echo ""
echo "Test runs saved to: ${TEST_RUN_FOLDER}"
echo "Evaluation files saved to: ${TEST_RESULTS_DIR}"
echo ""
