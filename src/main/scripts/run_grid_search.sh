#!/bin/bash

# Grid search script for PRF with LLM strategies
# This script runs experiments with:
# 1. Baselines: LMDirichlet (mu=2000) and BM25 (k1=1.2, b=0.75)
# 2. MonoT5 Reranker 
# 3. PRF with different combinations of:
#    - depth: 5, 10, 25, 50, 75, 100
#    - e (expansion terms): 5, 10, 15, 20, 25, 30
#    - lambda (interpolation): 0.1 to 0.9 in steps of 0.1
#    - RF models: RM3, DMM, MEDMM
#    - RF strategies: PRF, VLLM, VLLM-PROB, MONOT5, MONOT5-PROB, ORACLE, ORACLE-K

set -e

# Source shared dataset configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_config.sh"

# Parse command line arguments
SKIP_RERANK=false
BASELINE_MODELS=("LMDirichlet" "BM25")  # Default: run both baselines

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
            echo "Options:"
            echo "  --skip-rerank       Skip MonoT5 reranker experiments"
            exit 1
            ;;
    esac
done

mkdir -p "$RUN_FOLDER"

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

# Check if MonoT5 service is running (only if not skipping rerank)
if [ "$SKIP_RERANK" = false ]; then
    if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: MonoT5 service might not be running at http://127.0.0.1:5000${NC}"
        echo "Start it with: python src/main/python/mono_t5.py"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Calculate total experiments
TOTAL_BASELINES=${#BASELINE_MODELS[@]}
TOTAL_PRF=$((${#RF_STRATEGY_VALUES[@]} * ${#RF_MODEL_VALUES[@]}))  # Strategy x RF Model combinations
if [ "$SKIP_RERANK" = true ]; then
    TOTAL_RERANK=0
    TOTAL_EXPERIMENTS=$((TOTAL_BASELINES + TOTAL_PRF))
else
    TOTAL_RERANK=$((${#DEPTHS[@]} * ${#BASELINE_MODELS[@]}))  # Rerank for each baseline
    TOTAL_EXPERIMENTS=$((TOTAL_BASELINES + TOTAL_PRF + TOTAL_RERANK))
fi

# Calculate total grid configurations for informational purposes
GRID_CONFIGS=$((${#DEPTHS[@]} * ${#E_VALUES[@]} * ${#LAMBDA_VALUES[@]}))
TOTAL_PRF_CONFIGS=$((GRID_CONFIGS * TOTAL_PRF))

echo -e "Grid Search Mode: ${GREEN}ENABLED${NC} (opening index once per strategy/RF model combo)"
echo -e "Total Java invocations: ${GREEN}$TOTAL_EXPERIMENTS${NC}"
echo "  - Baseline models: $TOTAL_BASELINES (${BASELINE_MODELS[*]})"
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

# Run baseline experiments (LMDirichlet and/or BM25)
echo ""
echo -e "${BLUE}=== Part 1: Baseline Models ===${NC}"
echo ""

for baseline_model in "${BASELINE_MODELS[@]}"; do
    COUNTER=$((COUNTER + 1))
    echo -e "${GREEN}[$COUNTER/$TOTAL_EXPERIMENTS]${NC} Running baseline $baseline_model"
    
    java -cp "$JAR_PATH" org.irlab.llmprf.searcher.TRECSearcherLucene search \
        --index "$INDEX_PATH" \
        --topics "$TOPICS_PATH" \
        --runsOutputFolder "$RUN_FOLDER" \
        --baseline_model "$baseline_model" \
        --rerank_method none \
        --prf_strategy none

    echo -e "${GREEN}✓${NC} Completed baseline $baseline_model"
    echo ""
done

# Run MonoT5 Reranker experiments (only if not skipped)
if [ "$SKIP_RERANK" = false ]; then
    echo ""
    echo -e "${BLUE}=== Part 2: MonoT5 Reranker (no PRF) ===${NC}"
    echo ""

    # Run MonoT5 Reranker experiments for each baseline and depth
    for baseline_model in "${BASELINE_MODELS[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            COUNTER=$((COUNTER + 1))
            echo -e "${GREEN}[$COUNTER/$TOTAL_EXPERIMENTS]${NC} Running MonoT5 Reranker with $baseline_model baseline, depth=$depth"
            
            java -cp "$JAR_PATH" org.irlab.llmprf.searcher.TRECSearcherLucene search \
                --index "$INDEX_PATH" \
                --topics "$TOPICS_PATH" \
                --runsOutputFolder "$RUN_FOLDER" \
                --baseline_model "$baseline_model" \
                --rerank_method monot5 \
                --prf_strategy none \
                --rerank_depth $depth \
                --cache_dir "$CACHE_DIR"

            echo -e "${GREEN}✓${NC} Completed MonoT5 Reranker $baseline_model depth=$depth"
            echo ""
        done
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

# Run PRF experiments with internal grid search (one Java invocation per strategy x RF model)
for RF_STRATEGY in "${RF_STRATEGY_VALUES[@]}"; do
    for RF_MODEL in "${RF_MODEL_VALUES[@]}"; do
        COUNTER=$((COUNTER + 1))
        
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}  ${RF_STRATEGY} + ${RF_MODEL} Grid Search Experiment${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        echo "Configuration:"
        echo "  Index: $INDEX_PATH"
        echo "  Topics: $TOPICS_PATH"
        echo "  Output: $RUN_FOLDER"
        echo "  Baseline: $DEFAULT_BASELINE_MODEL (for PRF experiments)"
        echo "  RF Strategy: $RF_STRATEGY"
        echo "  RF Model: $RF_MODEL"
        echo ""
        echo "Grid parameters:"
        echo "  Depths: $DEPTHS_STR"
        echo "  E values: $E_VALUES_STR"
        echo "  Lambda values: $LAMBDA_VALUES_STR"
        echo ""
        
        echo -e "${GREEN}[$COUNTER/$TOTAL_EXPERIMENTS]${NC} Running ${RF_STRATEGY} + ${RF_MODEL} PRF grid search..."
        
        # Prepare additional parameters based on RF strategy
        ADDITIONAL_PARAMS=""
        case "$RF_STRATEGY" in
            "ORACLE"|"ORACLE-K")
                ADDITIONAL_PARAMS="--qrels $QRELS_PATH"
                ;;
            "MONOT5"|"MONOT5-PROB"|"VLLM"|"VLLM-PROB")
                ADDITIONAL_PARAMS="--cache_dir $CACHE_DIR"
                ;;
        esac
        
        java -cp "$JAR_PATH" org.irlab.llmprf.searcher.TRECSearcherLucene search \
            --index "$INDEX_PATH" \
            --topics "$TOPICS_PATH" \
            --runsOutputFolder "$RUN_FOLDER" \
            --baseline_model "$DEFAULT_BASELINE_MODEL" \
            --rerank_method prf \
            --prf_strategy "$RF_STRATEGY" \
            --prf_model "$RF_MODEL" \
            --grid_search \
            --depths "$DEPTHS_STR" \
            --e_values "$E_VALUES_STR" \
            --lambdas "$LAMBDA_VALUES_STR" \
            $ADDITIONAL_PARAMS \
        
        echo -e "${GREEN}✓${NC} Completed ${RF_STRATEGY} + ${RF_MODEL} grid search"
        echo ""
    done
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
echo "  Baseline models tested: ${BASELINE_MODELS[*]}"
echo "  RF models tested: ${RF_MODEL_VALUES[*]}"
echo "  RF strategies tested: ${RF_STRATEGY_VALUES[*]}"
echo "  Total configurations: $((TOTAL_BASELINES + TOTAL_RERANK + TOTAL_PRF_CONFIGS))"
if [ "$SKIP_RERANK" = true ]; then
    echo "  (MonoT5 Reranker experiments were skipped)"
fi
echo "  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Average time per invocation: $((TOTAL_TIME / TOTAL_EXPERIMENTS))s"
echo ""
echo "Results saved in: $RUN_FOLDER"
echo ""
echo -e "${GREEN}Done!${NC}"
