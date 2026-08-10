#!/bin/bash

FOLDER="/home/david/llm-prf"

declare -a AP8889=("ap8889" "topics.51-100" "qrels.AP8889.51-100" "topics.101-200" "qrels_ap8889_101_200.txt" "true")
declare -a ROBUST04=("robust04" "topics.301-350.trec.txt" "qrels.robust04.300-450.601-700.trec.txt" "topics.351-400.trec.txt" "qrels.robust04.300-450.601-700.trec.txt" "true")
declare -a DL19=("msmarco" "topics.dl-19.trec" "qrels.dl19-passage.nist.trec.txt" "topics.dl-20.trec" "qrels.dl20-passage.nist.trec.txt" "false")
declare -a WT10G=("wt10g" "topics.451-500.trec.txt" "qrels.trec9.main_web.451-500" "topics.501-550" "qrels.wt10g.501-550" "true")
DATASET=("${AP8889[@]}")

INDEX="${DATASET[0]}"
TOPICS="${DATASET[1]}"
QRELS="${DATASET[2]}"
TOPICS_TEST="${DATASET[3]}"
QRELS_TEST="${DATASET[4]}"
SUPPORTS_NARRATIVE="${DATASET[5]}"

INDEX_PATH="${FOLDER}/indices/${INDEX}"
TOPICS_PATH="${FOLDER}/topics/${TOPICS}"
QRELS_PATH="${FOLDER}/qrels/${QRELS}"
RUN_FOLDER="${FOLDER}/runs/${INDEX}"
RESULTS_DIR="${FOLDER}/grid_results/${INDEX}"
CACHE_DIR="${FOLDER}/cache/${INDEX}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

DEPTHS=(100 5 10 25 50 75)
E_VALUES=(5 10 15 20 25 30 35 40 45 50 75 100 110 125 150)
RF_MODEL_VALUES=("RM3" "DMM" "MEDMM")
LAMBDA_VALUES=(0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

RF_STRATEGY_TERMFILTER_COMBOS=(
    "PRF:none"
    "MONOT5:none"
    "MONOT5-PROB:none"
    "VLLM:none"
    "VLLM-PROB:none"
    "VLLM-NARR:none"
    "VLLM-NARR-PROB:none"
    "ORACLE-K:none"
    "PRF:vllmspans2-nonarr"
    "PRF:vllmspans2"
    "MONOT5:vllmspans2-nonarr"
    "MONOT5:vllmspans2"
    "VLLM:vllmspans2-nonarr"
    "VLLM:vllmspans2"
    "VLLM-NARR:vllmspans2-nonarr"
    "VLLM-NARR:vllmspans2"
    "MONOT5-PROB:vllmspans2-nonarr"
    "MONOT5-PROB:vllmspans2"
    "VLLM-PROB:vllmspans2-nonarr"
    "VLLM-PROB:vllmspans2"
    "VLLM-NARR-PROB:vllmspans2-nonarr"
    "VLLM-NARR-PROB:vllmspans2"
    "VLLM-JUDGESPANS:vllmjudgespans"
    "VLLM-NARR-JUDGESPANS:vllmjudgespans"
)

DEFAULT_BASELINE_MODEL="LMDirichlet"
SEARCH_BY="content"

JAR_PATH="../llmprf-1.0-jar-with-dependencies.jar"

switch_dataset() {
    local dataset_name="$1"
    case "$dataset_name" in
        "ap8889"|"AP8889")
            DATASET=("${AP8889[@]}")
            ;;
        "robust04"|"ROBUST04")
            DATASET=("${ROBUST04[@]}")
            ;;
        "wt10g"|"WT10G")
            DATASET=("${WT10G[@]}")
            ;;
        "dl19"|"DL19")
          DATASET=("${DL19[@]}")
            ;;
        *)
            echo -e "${RED}Error: Unknown dataset '$dataset_name'. Available: ap8889, dl19, wt10g, robust04${NC}"
            return 1
            ;;
    esac
    
    INDEX="${DATASET[0]}"
    TOPICS="${DATASET[1]}"
    QRELS="${DATASET[2]}"
    TOPICS_TEST="${DATASET[3]}"
    QRELS_TEST="${DATASET[4]}"
    INDEX_PATH="${FOLDER}/indices/${INDEX}"
    TOPICS_PATH="${FOLDER}/topics/${TOPICS}"
    QRELS_PATH="${FOLDER}/qrels/${QRELS}"
    TOPICS_TEST_PATH="${FOLDER}/topics/${TOPICS_TEST}"
    QRELS_TEST_PATH="${FOLDER}/qrels/${QRELS_TEST}"
    SUPPORTS_NARRATIVE="${DATASET[5]}"
    RUN_FOLDER="${FOLDER}/runs/${INDEX}"
    RESULTS_DIR="${FOLDER}/grid_results/${INDEX}"
    CACHE_DIR="${FOLDER}/cache/${INDEX}"
    
    echo -e "${GREEN}Switched to dataset: $dataset_name${NC}"
    echo "  Index: $INDEX"
    echo "  Topics: $TOPICS"
    echo "  Qrels: $QRELS"
}

validate_paths() {
    local errors=0
    
    if [ ! -d "$INDEX_PATH" ]; then
        echo -e "${RED}Error: Index path not found: $INDEX_PATH${NC}"
        errors=$((errors + 1))
    fi
    
    if [ ! -f "$TOPICS_PATH" ]; then
        echo -e "${RED}Error: Topics file not found: $TOPICS_PATH${NC}"
        errors=$((errors + 1))
    fi
    
    if [ ! -f "$QRELS_PATH" ]; then
        echo -e "${RED}Error: Qrels file not found: $QRELS_PATH${NC}"
        errors=$((errors + 1))
    fi
    
    return $errors
}

show_config() {
    echo -e "${BLUE}Current Dataset Configuration:${NC}"
    echo "  Dataset: ${INDEX}"
    echo "  Index Path: ${INDEX_PATH}"
    echo "  Topics Path: ${TOPICS_PATH}"
    echo "  Qrels Path: ${QRELS_PATH}"
    echo "  Run Folder: ${RUN_FOLDER}"
    echo "  Results Dir: ${RESULTS_DIR}"
    echo "  Cache Dir: ${CACHE_DIR}"
    echo ""
}
