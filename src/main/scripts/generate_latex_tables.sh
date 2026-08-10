#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_config.sh"

PYTHON_SCRIPT="${SCRIPT_DIR}/../python/test_evaluation.py"
OUTPUT_DIR="${FOLDER}/test_results"

MODE="${1:-all}"

VENV_PATH="${SCRIPT_DIR}/../../../../.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

TERMS_EXAMPLES=(
    "ap8889:148:12:table_terms_148:RM3:PRF:none:RM3:100:125:0.10;+MonoT5F + RM3:MONOT5:none:RM3:100:150:0.10;+LLMF w/narr + SpanSelection w/narr + RM3:VLLM-NARR:vllmspans2:RM3:100:150:0.05"
)

run_aggregate_tables() {
    echo -e "${BLUE}=== Generating aggregate LaTeX tables (effectiveness, tuned parameters, Robustness Index) ===${NC}"
    python3 "$PYTHON_SCRIPT" --latex "$FOLDER"
    echo -e "${GREEN}✓ Aggregate tables written to $OUTPUT_DIR${NC}"
    echo ""
}

run_terms_tables() {
    echo -e "${BLUE}=== Generating expanded-terms case-study tables ===${NC}"

    if [ ! -f "$JAR_PATH" ]; then
        echo -e "${RED}Error: JAR file not found at $JAR_PATH${NC}"
        echo "Please run: mvn package"
        return 1
    fi

    for entry in "${TERMS_EXAMPLES[@]}"; do
        IFS=':' read -r ds topic_id top_n out_name compare_spec <<< "$entry"

        echo -e "${GREEN}-> topic $topic_id on $ds -> ${out_name}.tex${NC}"
        switch_dataset "$ds" > /dev/null

        raw_output=$(java -cp "$JAR_PATH" org.irlab.llmprf.searcher.TRECSearcherLucene search \
            --index "$INDEX_PATH" \
            --topics "$TOPICS_TEST_PATH" \
            --runsOutputFolder /tmp \
            --baseline_model BM25 \
            --rerank_method dump_terms \
            --cache_dir "$CACHE_DIR" \
            --topic_id "$topic_id" \
            --top_n "$top_n" \
            --compare "$compare_spec" 2>/dev/null)

        echo "$raw_output" | sed -n '/\\begin{table\*}/,/\\end{table\*}/p' > "$OUTPUT_DIR/${out_name}.tex"

        if [ ! -s "$OUTPUT_DIR/${out_name}.tex" ]; then
            echo -e "${RED}  Warning: no table found in output -- check the raw java output above${NC}"
            echo "$raw_output"
        else
            echo -e "${GREEN}  ✓ Written: $OUTPUT_DIR/${out_name}.tex${NC}"
        fi
    done
    echo ""
}

mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    --aggregate-only)
        run_aggregate_tables
        ;;
    --terms-only)
        run_terms_tables
        ;;
    all)
        run_aggregate_tables
        run_terms_tables
        ;;
    *)
        echo -e "${RED}Unknown option: $MODE${NC}"
        echo "Usage: $0 [--aggregate-only|--terms-only]"
        exit 1
        ;;
esac

echo -e "${GREEN}Done. LaTeX tables in: $OUTPUT_DIR${NC}"
