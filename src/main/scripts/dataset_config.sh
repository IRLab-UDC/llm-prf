#!/bin/bash

# Shared dataset configuration for grid search and analysis scripts
# This file contains dataset paths and configuration that can be sourced by other scripts

# Base folder configuration
FOLDER="/home/javier/data/"

# Dataset definitions (array format: index_name topics_file qrels_file)
declare -a AP8889=("ap8889_index" "topics.101-200" "qrels_ap8889_101_200.txt")
declare -a ROBUST04=("robust04_index" "topics.301-350.trec.txt" "qrels.robust04.300-450.601-700.trec.txt")
declare -a DL19=("msmarco_index" "topics.dl-19.trec" "qrels.dl19-passage.nist.trec.txt")

# Select active dataset (change this to switch datasets)
# Options: DATASET=("${AP8889[@]}") or DATASET=("${ROBUST04[@]}")
DATASET=("${AP8889[@]}")  # Currently set to AP8889

# Extract dataset components
INDEX="${DATASET[0]}"
TOPICS="${DATASET[1]}"
QRELS="${DATASET[2]}"

# Construct full paths
INDEX_PATH="${FOLDER}/indices/${INDEX}"
TOPICS_PATH="${FOLDER}/topics/${TOPICS}"
QRELS_PATH="${FOLDER}/topics/${QRELS}"
RUN_FOLDER="${FOLDER}/runs/${INDEX}"
RESULTS_DIR="${FOLDER}/grid_results/${INDEX}"
CACHE_DIR="${FOLDER}/cache/${INDEX}"  # Collection-specific cache directory

# Colors for consistent output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Grid search parameters (used by run_grid_search.sh)
DEPTHS=(100 5 10 25 50 75)
E_VALUES=(5 10 15 20 25 30)
RF_STRATEGY_VALUES=("VLLM" "VLLM-PROB" "ORACLE" "ORACLE-K" "MONOT5" "MONOT5-PROB" "PRF")
LAMBDA_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Model parameters
MU=2000
SEARCH_BY="title_only"
RF_MODEL="RM3"
PRF_SMOOTHING="Additive"
PRF_SMOOTHING_PARAM=0.1

# JAR path (relative to scripts directory)
JAR_PATH="../prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar"

# Function to switch dataset
switch_dataset() {
    local dataset_name="$1"
    case "$dataset_name" in
        "ap8889"|"AP8889")
            DATASET=("${AP8889[@]}")
            ;;
        "robust04"|"ROBUST04")
            DATASET=("${ROBUST04[@]}")
            ;;
        "dl19"|"DL19")
          DATASET=("${DL19[@]}")
            ;;
        *)
            echo -e "${RED}Error: Unknown dataset '$dataset_name'. Available: ap8889, robust04${NC}"
            return 1
            ;;
    esac
    
    # Update derived variables
    INDEX="${DATASET[0]}"
    TOPICS="${DATASET[1]}"
    QRELS="${DATASET[2]}"
    INDEX_PATH="${FOLDER}/indices/${INDEX}"
    TOPICS_PATH="${FOLDER}/topics/${TOPICS}"
    QRELS_PATH="${FOLDER}/topics/${QRELS}"
    RUN_FOLDER="${FOLDER}/runs/${INDEX}"
    RESULTS_DIR="${FOLDER}/grid_results/${INDEX}"
    CACHE_DIR="${FOLDER}/cache/${INDEX}"
    
    echo -e "${GREEN}Switched to dataset: $dataset_name${NC}"
    echo "  Index: $INDEX"
    echo "  Topics: $TOPICS"
    echo "  Qrels: $QRELS"
}

# Function to validate paths
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

# Function to display current configuration
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