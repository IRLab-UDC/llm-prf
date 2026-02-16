#!/bin/bash

# Grid search script for PRF with LLM strategies
# This script runs experiments with:
# 1. Baseline LMDirichlet (mu=2000)
# 2. MonoT5 Reranker (optional, can be skipped)
# 3. PRF with different combinations of:
#    - depth: 5, 10, 25, 50, 75, 100
#    - e (expansion terms): 5, 10, 15, 20, 25, 30
#    - lambda (interpolation): 0.1 to 0.9 in steps of 0.1
#    - RF strategies: RAW RM3, VLLM, VLLM-PROB, MONOT5, MONOT5-PROB

set -e

# Source shared dataset configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_config.sh"

# Parse command line arguments
SKIP_RERANK=false

for arg in "$@"; do
    case $arg in
        --skip-rerank)
            SKIP_RERANK=true
            ;;
        ap8889|robust04|dl19|wt10g)
            echo "Switching to dataset: $arg"
            if ! switch_dataset "$arg"; then
                exit 1
            fi
            echo ""
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [dataset] [--skip-rerank]"
            echo "Available datasets: ap8889, robust04, wt10g, dl19"
            exit 1
            ;;
    esac
done

# Show current configuration
show_config

# Validate paths
if ! validate_paths; then
    echo -e "${RED}Configuration validation failed. Please check paths.${NC}"
    exit 1
fi

# Check if JAR exists
if [ ! -f "$JAR_PATH" ]; then
    echo -e "${RED}Error: JAR file not found at $JAR_PATH${NC}"
    echo "Please run: mvn package"
    exit 1
fi

# Check if MonoT5 service is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: MonoT5 service might not be running at http://127.0.0.1:5000${NC}"
    echo "Start it with: python src/main/python/mono_t5.py"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Calculate total experiments
# Now PRF uses grid search mode, so it's just one Java call per RF strategy
TOTAL_PRF=${#RF_STRATEGY_VALUES[@]}  # One grid search per strategy
TOTAL_BASELINE=1  # LMDirichlet baseline
if [ "$SKIP_RERANK" = true ]; then
    TOTAL_RERANK=0
    TOTAL_EXPERIMENTS=$((TOTAL_BASELINE + TOTAL_PRF))
else
    TOTAL_RERANK=${#DEPTHS[@]}
    TOTAL_EXPERIMENTS=$((TOTAL_BASELINE + TOTAL_PRF + TOTAL_RERANK))
fi

# Calculate total grid configurations for informational purposes
GRID_CONFIGS=$((${#DEPTHS[@]} * ${#E_VALUES[@]} * ${#LAMBDA_VALUES[@]}))
TOTAL_PRF_CONFIGS=$((GRID_CONFIGS * ${#RF_STRATEGY_VALUES[@]}))

echo -e "Grid Search Mode: ${GREEN}ENABLED${NC} (opening index once per strategy)"
echo -e "Total Java invocations: ${GREEN}$TOTAL_EXPERIMENTS${NC}"
echo "  - Baseline: $TOTAL_BASELINE"
echo "  - PRF Grid Searches: $TOTAL_PRF (covering $TOTAL_PRF_CONFIGS configurations)"
if [ "$SKIP_RERANK" = true ]; then
    echo -e "  - MonoT5 Reranker: ${YELLOW}SKIPPED${NC}"
else
    echo "  - MonoT5 Reranker: $TOTAL_RERANK"
fi
echo ""

# read -p "Press Enter to start..."

# Counter for progress
COUNTER=0
START_TIME=$(date +%s)

# Run baseline LMDirichlet experiment
echo ""
echo -e "${BLUE}=== Part 1: Baseline LMDirichlet (mu=$MU) ===${NC}"
echo ""

COUNTER=$((COUNTER + 1))
echo -e "${GREEN}[$COUNTER/$TOTAL_EXPERIMENTS]${NC} Running baseline LMDirichlet with mu=$MU"

java -cp "$JAR_PATH" org.irlab.ecir26.searcher.TRECSearcherLucene \
    --index "$INDEX_PATH" \
    --topics "$TOPICS_PATH" \
    --trec_run_folder "$RUN_FOLDER" \
    --qrels "$QRELS_PATH" \
    --search_by "$SEARCH_BY" \
    --rerank_method none \
    --prf_strategy none \
    --mu $MU

echo -e "${GREEN}✓${NC} Completed baseline LMDirichlet"
echo ""

# Run MonoT5 Reranker experiments (only if not skipped)
if [ "$SKIP_RERANK" = false ]; then
    echo ""
    echo -e "${BLUE}=== Part 2: MonoT5 Reranker (no PRF) ===${NC}"
    echo ""

    # Run MonoT5 Reranker experiments (no PRF, just reranking)
    for depth in "${DEPTHS[@]}"; do
        COUNTER=$((COUNTER + 1))
        echo -e "${GREEN}[$COUNTER/$TOTAL_EXPERIMENTS]${NC} Running MonoT5 Reranker with depth=$depth"
        
        java -cp "$JAR_PATH" org.irlab.ecir26.searcher.TRECSearcherLucene \
            --index "$INDEX_PATH" \
            --topics "$TOPICS_PATH" \
            --qrels "$QRELS_PATH" \
            --trec_run_folder "$RUN_FOLDER" \
            --search_by "$SEARCH_BY" \
            --rerank_method monot5 \
            --rerank_depth $depth \
            --cache_dir "$CACHE_DIR" \
            --mu $MU

        echo -e "${GREEN}✓${NC} Completed MonoT5 Reranker depth=$depth"
        echo ""
    done
else
    echo ""
    echo -e "${YELLOW}=== Skipping MonoT5 Reranker experiments ===${NC}"
    echo ""
fi

echo ""
echo -e "${BLUE}=== Part 3: PRF with LLM Strategies (Grid Search Mode) ===${NC}"
echo ""

# Build parameter strings for grid search
DEPTHS_STR=$(IFS=,; echo "${DEPTHS[*]}")
E_VALUES_STR=$(IFS=,; echo "${E_VALUES[*]}")
LAMBDA_VALUES_STR=$(IFS=,; echo "${LAMBDA_VALUES[*]}")

# Run PRF experiments with internal grid search (one Java invocation per strategy)
for RF_STRATEGY in "${RF_STRATEGY_VALUES[@]}"; do
    COUNTER=$((COUNTER + 1))
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  ${RF_STRATEGY} Grid Search Experiment${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  Index: $INDEX_PATH"
    echo "  Topics: $TOPICS_PATH"
    echo "  Qrels: $QRELS_PATH"
    echo "  Output: $RUN_FOLDER"
    echo "  Mu: $MU"
    echo "  RF Strategy: $RF_STRATEGY"
    echo ""
    echo "Grid parameters:"
    echo "  Depths: $DEPTHS_STR"
    echo "  E values: $E_VALUES_STR"
    echo "  Lambda values: $LAMBDA_VALUES_STR"
    echo ""
    
    echo -e "${GREEN}[$COUNTER/$((TOTAL_BASELINE + 1 + ${#RF_STRATEGY_VALUES[@]}))]${NC} Running ${RF_STRATEGY} PRF grid search..."
    
    java -cp "$JAR_PATH" org.irlab.ecir26.searcher.TRECSearcherLucene \
        --index "$INDEX_PATH" \
        --topics "$TOPICS_PATH" \
        --qrels "$QRELS_PATH" \
        --cache_dir "$CACHE_DIR" \
        --trec_run_folder "$RUN_FOLDER" \
        --search_by "$SEARCH_BY" \
        --mu $MU \
        --rerank_method prf \
        --prf_strategy "$RF_STRATEGY" \
        --rf_model "$RF_MODEL" \
        --prf_smoothing_model "$PRF_SMOOTHING" \
        --prf_smoothing_parameter $PRF_SMOOTHING_PARAM \
        --grid_search \
        --depths "$DEPTHS_STR" \
        --e_values "$E_VALUES_STR" \
        --lambdas "$LAMBDA_VALUES_STR"
    
    echo -e "${GREEN}✓${NC} Completed ${RF_STRATEGY} grid search"
    echo ""
done

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Grid Search Completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
echo "  Total Java invocations: $TOTAL_EXPERIMENTS"
echo "  Total configurations: $((TOTAL_BASELINE + TOTAL_RERANK + TOTAL_PRF_CONFIGS))"
if [ "$SKIP_RERANK" = true ]; then
    echo "  (MonoT5 Reranker experiments were skipped)"
fi
echo "  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Average time per invocation: $((TOTAL_TIME / TOTAL_EXPERIMENTS))s"
echo ""
echo "Results saved in: $RUN_FOLDER"
echo ""
echo "Next steps:"
echo "  1. Evaluate runs with trec_eval"
echo "  2. Analyze results to find best parameters"
echo ""
echo -e "${GREEN}Done!${NC}"
