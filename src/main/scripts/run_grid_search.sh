#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_config.sh"

SKIP_RERANK=false
BASELINE_MODELS=("LMDirichlet" "BM25")

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

show_config

if ! validate_paths; then
    echo -e "${RED}Configuration validation failed. Please check paths.${NC}"
    exit 1
fi

if [ ! -f "$JAR_PATH" ]; then
    echo -e "${RED}Error: JAR file not found at $JAR_PATH${NC}"
    echo "Please run: mvn package"
    exit 1
fi

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

NEEDS_VLLM=false
for combo in "${RF_STRATEGY_TERMFILTER_COMBOS[@]}"; do
    strategy="${combo%%:*}"
    tf="${combo##*:}"
    if [[ "$strategy" == VLLM* || "$tf" == vllm* ]]; then NEEDS_VLLM=true; break; fi
done

if [ "$NEEDS_VLLM" = true ]; then
    VLLM_HOST="${VLLM_HOST:-localhost:8080}"
    if ! curl -s "http://${VLLM_HOST}/health" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: VLLM service might not be running at http://${VLLM_HOST}${NC}"
        echo "Start it with: python src/main/python/serve_vllm.py"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

TOTAL_BASELINES=${#BASELINE_MODELS[@]}

PRF_RUNS_PER_BASELINE=0
for combo in "${RF_STRATEGY_TERMFILTER_COMBOS[@]}"; do
    combo_strategy="${combo%%:*}"
    combo_tf="${combo##*:}"
    if [[ "$combo_strategy" == "VLLM-NARR" || "$combo_strategy" == "VLLM-NARR-PROB" ]] && [ "$SUPPORTS_NARRATIVE" != "true" ]; then
        continue
    fi
    if [ "$combo_tf" = "vllmspans2" ] && [ "$SUPPORTS_NARRATIVE" != "true" ]; then
        continue
    fi
    for rf_model in "${RF_MODEL_VALUES[@]}"; do
        if [[ "$combo_strategy" == *-PROB ]] && [ "$rf_model" != "RM3" ]; then
            continue
        fi
        PRF_RUNS_PER_BASELINE=$((PRF_RUNS_PER_BASELINE + 1))
    done
done
TOTAL_PRF=$((PRF_RUNS_PER_BASELINE * TOTAL_BASELINES))
if [ "$SKIP_RERANK" = true ]; then
    TOTAL_RERANK=0
    TOTAL_EXPERIMENTS=$((TOTAL_BASELINES + TOTAL_PRF))
else
    TOTAL_RERANK=$((${#DEPTHS[@]} * ${#BASELINE_MODELS[@]}))
    TOTAL_EXPERIMENTS=$((TOTAL_BASELINES + TOTAL_PRF + TOTAL_RERANK))
fi

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

COUNTER=0
START_TIME=$(date +%s)

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

if [ "$SKIP_RERANK" = false ]; then
    echo ""
    echo -e "${BLUE}=== Part 2: MonoT5 Reranker (no PRF) ===${NC}"
    echo ""

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

DEPTHS_STR=$(IFS=,; echo "${DEPTHS[*]}")
E_VALUES_STR=$(IFS=,; echo "${E_VALUES[*]}")
LAMBDA_VALUES_STR=$(IFS=,; echo "${LAMBDA_VALUES[*]}")

for BASELINE_MODEL in "${BASELINE_MODELS[@]}"; do
    for combo in "${RF_STRATEGY_TERMFILTER_COMBOS[@]}"; do
        RF_STRATEGY="${combo%%:*}"
        TERM_FILTER="${combo##*:}"

        if [[ "$RF_STRATEGY" == "VLLM-NARR" || "$RF_STRATEGY" == "VLLM-NARR-PROB" || "$RF_STRATEGY" == "VLLM-NARR-JUDGESPANS" ]] && [ "$SUPPORTS_NARRATIVE" != "true" ]; then
            echo -e "${YELLOW}Skipping $RF_STRATEGY (dataset '$INDEX' has no narratives)${NC}"
            continue
        fi
        if [ "$TERM_FILTER" = "vllmspans2" ] && [ "$SUPPORTS_NARRATIVE" != "true" ]; then
            echo -e "${YELLOW}Skipping ${RF_STRATEGY}:${TERM_FILTER} (dataset '$INDEX' has no narratives)${NC}"
            continue
        fi

        for RF_MODEL in "${RF_MODEL_VALUES[@]}"; do
            if [[ "$RF_STRATEGY" == *-PROB ]] && [ "$RF_MODEL" != "RM3" ]; then
                echo -e "${YELLOW}Skipping ${RF_STRATEGY}:${TERM_FILTER} + ${RF_MODEL} (logit only applies to RM3)${NC}"
                continue
            fi

            COUNTER=$((COUNTER + 1))

            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}  ${BASELINE_MODEL} + ${RF_STRATEGY} + ${RF_MODEL} + termFilter=${TERM_FILTER}${NC}"
            echo -e "${BLUE}========================================${NC}"

            echo -e "${GREEN}[$COUNTER/$TOTAL_EXPERIMENTS]${NC} Running..."

            ADDITIONAL_PARAMS="--term_filter $TERM_FILTER"
            case "$RF_STRATEGY" in
                "ORACLE-K")
                    ADDITIONAL_PARAMS="$ADDITIONAL_PARAMS --qrels $QRELS_PATH"
                    ;;
                "MONOT5"|"MONOT5-PROB"|"VLLM"|"VLLM-PROB"|"VLLM-NARR"|"VLLM-NARR-PROB"|"VLLM-JUDGESPANS"|"VLLM-NARR-JUDGESPANS")
                    ADDITIONAL_PARAMS="$ADDITIONAL_PARAMS --cache_dir $CACHE_DIR"
                    ;;
            esac

            DEPTHS_ARG="$DEPTHS_STR"
            if [[ "$TERM_FILTER" == vllm* ]]; then
                if [[ "$RF_STRATEGY" != "MONOT5"* && "$RF_STRATEGY" != "VLLM"* ]]; then
                    ADDITIONAL_PARAMS="$ADDITIONAL_PARAMS --cache_dir $CACHE_DIR"
                fi
            fi

            java -cp "$JAR_PATH" org.irlab.llmprf.searcher.TRECSearcherLucene search \
                --index "$INDEX_PATH" \
                --topics "$TOPICS_PATH" \
                --runsOutputFolder "$RUN_FOLDER" \
                --baseline_model "$BASELINE_MODEL" \
                --rerank_method prf \
                --prf_strategy "$RF_STRATEGY" \
                --prf_model "$RF_MODEL" \
                --grid_search \
                --depths "$DEPTHS_ARG" \
                --e_values "$E_VALUES_STR" \
                --lambdas "$LAMBDA_VALUES_STR" \
                $ADDITIONAL_PARAMS

            echo -e "${GREEN}✓${NC} Completed"
            echo ""
        done
    done
done

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
echo "  RF strategy:term_filter combos tested: ${RF_STRATEGY_TERMFILTER_COMBOS[*]}"
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
