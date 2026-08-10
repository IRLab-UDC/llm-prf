#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_config.sh"

if [ $# -ge 1 ]; then
    DATASET_ARG="$1"
    echo "Switching to dataset: $DATASET_ARG"
    if ! switch_dataset "$DATASET_ARG"; then
        exit 1
    fi
    echo ""
fi

if ! command -v trec_eval &> /dev/null; then
    echo -e "${RED}Error: trec_eval not found in PATH${NC}"
    echo "Please install trec_eval or ensure it's in your PATH"
    exit 1
fi

if ! validate_paths; then
    echo -e "${RED}Configuration validation failed.${NC}"
    show_config
    exit 1
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/per_query"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Grid Search Results Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

show_config

echo "Evaluating runs..."
echo ""

TEMP_DIR="$RESULTS_DIR/temp_eval_$$"
mkdir -p "$TEMP_DIR"

BASELINE_SUMMARY="baseline.tsv"
MONOT5_SUMMARY="monot5.tsv"
PRF_SUMMARY="prf.tsv"
echo -e "run_name\tbaseline\tmap\tP@10\tndcg@100" > "$TEMP_DIR/$BASELINE_SUMMARY"
echo -e "run_name\tbaseline\tdepth\tmap\tP@10\tndcg@100" > "$TEMP_DIR/$MONOT5_SUMMARY"
echo -e "run_name\tbaseline\tstrategy\trf_model\tdepth\te\tlambda\tterm_filter\tmap\tP@10\tndcg@100" > "$TEMP_DIR/$PRF_SUMMARY"

echo "Evaluating runs..."
echo ""

TOTAL_RUNS=$(find "$RUN_FOLDER" -type f | grep -E "(LMDirichlet|BM25)" | wc -l)
echo "Total runs to evaluate: $TOTAL_RUNS"
echo ""

eval_run() {
    local run_file="$1"
    local run_name
    run_name=$(basename "$run_file")

    if [ ! -f "$run_file" ]; then return; fi

    local eval_output
    eval_output=$(trec_eval -m map -m P.10 -m ndcg_cut.100 "$QRELS_PATH" "$run_file" 2>/dev/null)
    local map p10 ndcg100
    map=$(echo "$eval_output"     | grep "^map "          | awk '{print $3}')
    p10=$(echo "$eval_output"     | grep "^P_10 "         | awk '{print $3}')
    ndcg100=$(echo "$eval_output" | grep "^ndcg_cut_100 " | awk '{print $3}')

    if [ -z "$map" ] || [ -z "$p10" ] || [ -z "$ndcg100" ]; then return; fi

    trec_eval -q -m map "$QRELS_PATH" "$run_file" 2>/dev/null \
        | grep -v "^map[[:space:]]*all" > "$TEMP_DIR/perquery_${run_name}.txt"

    local baseline=""
    if   [[ "$run_name" =~ LMDirichlet ]]; then baseline="LMDirichlet"
    elif [[ "$run_name" =~ BM25 ]];        then baseline="BM25"
    else return; fi

    if [[ "$run_name" =~ ^(LMDirichlet-2000|BM25)_content$ ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$run_name" "$baseline" "$map" "$p10" "$ndcg100" \
            > "$TEMP_DIR/row_baseline_${run_name}.tsv"

    elif [[ "$run_name" =~ _rerank-monoT5_topK-([0-9]+)$ ]]; then
        local depth="${BASH_REMATCH[1]}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run_name" "$baseline" "$depth" "$map" "$p10" "$ndcg100" \
            > "$TEMP_DIR/row_monot5_${run_name}.tsv"

    elif [[ "$run_name" =~ _prf-true_rfStrategy-([^_]+)_rfModel-([^_]+)_.*_lambda-([0-9.]+)_e-([0-9]+)(_termFilter-([^_]+))?$ ]]; then
        local strategy="${BASH_REMATCH[1]}"
        local rf_model="${BASH_REMATCH[2]}"
        local lambda="${BASH_REMATCH[3]}"
        local e="${BASH_REMATCH[4]}"
        local term_filter="${BASH_REMATCH[6]:-none}"
        local depth="ALL"
        if [[ "$run_name" =~ _topK-([0-9]+)_ ]]; then depth="${BASH_REMATCH[1]}"; fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_name" "$baseline" "$strategy" "$rf_model" "$depth" "$e" "$lambda" "$term_filter" "$map" "$p10" "$ndcg100" \
            > "$TEMP_DIR/row_prf_${run_name}.tsv"
    fi
}

_prf_test_file=$(find "$RUN_FOLDER" -type f -name '*prf-true*' | head -1)
if [ -n "$_prf_test_file" ]; then
    eval_run "$_prf_test_file"
    _prf_test_name=$(basename "$_prf_test_file")
    if [ -f "$TEMP_DIR/row_prf_${_prf_test_name}.tsv" ]; then
        echo -e "${GREEN}✓ eval_run sanity check passed${NC}"
        /bin/rm -f "$TEMP_DIR/row_prf_${_prf_test_name}.tsv"
    else
        echo -e "${RED}✗ eval_run sanity check FAILED for: $_prf_test_file${NC}"
        echo "  TEMP_DIR=$TEMP_DIR"
        echo "  QRELS_PATH=$QRELS_PATH"
        _out=$(trec_eval -m map -m P.10 -m ndcg_cut.100 "$QRELS_PATH" "$_prf_test_file" 2>&1)
        echo "  trec_eval output: $_out"
        exit 1
    fi
fi

NUM_CORES=$(nproc 2>/dev/null || echo 4)
echo "Using $NUM_CORES CPU cores for parallel evaluation..."

RUNNING=0
while IFS= read -r run_file; do
    ( set +e; eval_run "$run_file" ) &
    RUNNING=$((RUNNING + 1))
    if (( RUNNING >= NUM_CORES )); then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done < <(find "$RUN_FOLDER" -type f | grep -E "(LMDirichlet|BM25)")
wait

echo "  row_prf files: $(find "$TEMP_DIR" -name 'row_prf_*.tsv' | wc -l)"

echo ""
echo "Consolidating results..."

find "$TEMP_DIR" -name 'row_baseline_*.tsv' -exec cat {} + >> "$TEMP_DIR/$BASELINE_SUMMARY"
find "$TEMP_DIR" -name 'row_monot5_*.tsv'  -exec cat {} + >> "$TEMP_DIR/$MONOT5_SUMMARY"
find "$TEMP_DIR" -name 'row_prf_*.tsv'     -exec cat {} + >> "$TEMP_DIR/$PRF_SUMMARY"

cp "$TEMP_DIR/$BASELINE_SUMMARY" "$RESULTS_DIR/$BASELINE_SUMMARY"
cp "$TEMP_DIR/$MONOT5_SUMMARY"   "$RESULTS_DIR/$MONOT5_SUMMARY"
cp "$TEMP_DIR/$PRF_SUMMARY"      "$RESULTS_DIR/$PRF_SUMMARY"

find "$TEMP_DIR" -name 'perquery_*.txt' -exec mv {} "$RESULTS_DIR/per_query/" \;

/bin/rm -rf "$TEMP_DIR"

echo -e "${GREEN}✓ Evaluation completed${NC}"
echo ""

echo -e "${BLUE}=== Best Configurations ===${NC}"
echo ""

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
            if (NF == 5)  printf "  %s (%s): MAP=%.4f, P@10=%.4f, nDCG@100=%.4f\n", $1, $2, $3, $4, $5
            else if (NF == 6)  printf "  %s (%s, depth=%s): MAP=%.4f, P@10=%.4f, nDCG@100=%.4f\n", $1, $2, $3, $4, $5, $6
            else if (NF == 11) printf "  %s+%s (%s, d=%s, e=%s, λ=%s, tf=%s): MAP=%.4f, P@10=%.4f, nDCG@100=%.4f\n", $3, $4, $2, $5, $6, $7, $8, $9, $10, $11
        }'
    done
    echo ""
}

show_best "$RESULTS_DIR/$BASELINE_SUMMARY" "Best Baseline Models" "3"
show_best "$RESULTS_DIR/$MONOT5_SUMMARY" "Best MonoT5 Reranking" "4"
show_best "$RESULTS_DIR/$PRF_SUMMARY" "Best PRF Configurations" "9"

echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary files created:"
echo "  - $RESULTS_DIR/$BASELINE_SUMMARY"
echo "  - $RESULTS_DIR/$MONOT5_SUMMARY"
echo "  - $RESULTS_DIR/$PRF_SUMMARY"
echo "  - Per-query results: $RESULTS_DIR/per_query/"
echo ""

echo -e "${GREEN}✓ Analysis completed!${NC}"
