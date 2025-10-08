#!/bin/bash

# Master script - runs complete grid search pipeline
# 1. Runs grid search experiments
# 2. Evaluates with rec_eval
# 3. Generates visualizations
# 4. Creates markdown report

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════╗"
echo "║   MonoT5 Grid Search - Complete Pipeline          ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
echo "Checking prerequisites..."

# 1. JAR file
if [ ! -f "target/prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar" ]; then
    echo -e "${RED}✗ Fat JAR not found${NC}"
    echo "Building project..."
    mvn clean package -DskipTests
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ JAR exists${NC}"

# 2. MonoT5 service
if ! curl -s http://127.0.0.1:5000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ MonoT5 service not detected${NC}"
    echo "Please start MonoT5 in another terminal:"
    echo "  python src/main/python/mono_t5.py"
    read -p "Press Enter when ready (or Ctrl+C to cancel)..."
fi
echo -e "${GREEN}✓ MonoT5 service running${NC}"

# 3. Python packages for visualization
if ! python3 -c "import pandas, matplotlib, seaborn" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Python packages for visualization not found${NC}"
    echo "Installing: pandas, matplotlib, seaborn"
    pip install pandas matplotlib seaborn
fi
echo -e "${GREEN}✓ Python packages available${NC}"

# 4. rec_eval
if ! command -v rec_eval &> /dev/null; then
    echo -e "${YELLOW}⚠ rec_eval not found${NC}"
    echo "Evaluation and analysis will be skipped"
    SKIP_EVAL=true
else
    echo -e "${GREEN}✓ rec_eval available${NC}"
    SKIP_EVAL=false
fi

echo ""

# Ask for confirmation
echo -e "${BLUE}Pipeline steps:${NC}"
echo "  1. Run grid search"
echo "  2. Evaluate with rec_eval (if available)"
echo "  3. Generate visualizations"
echo "  4. Create markdown report"
echo ""

read -p "Continue with full pipeline? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Step 1: Run grid search
echo -e "${BLUE}═══ Step 1/4: Running Grid Search ═══${NC}"
echo ""
START_TOTAL=$(date +%s)

./run_grid_search.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}Grid search failed!${NC}"
    exit 1
fi

echo ""

# Step 2: Evaluate (if rec_eval available)
if [ "$SKIP_EVAL" = false ]; then
    echo -e "${BLUE}═══ Step 2/4: Evaluating Results ═══${NC}"
    echo ""
    ./analyze_grid_results.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Evaluation failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}═══ Step 2/4: Skipped (no rec_eval) ═══${NC}"
fi

echo ""

# Step 3: Visualizations (if evaluation was done)
if [ "$SKIP_EVAL" = false ]; then
    echo -e "${BLUE}═══ Step 3/4: Generating Visualizations ═══${NC}"
    echo ""
    python3 src/main/python/visualize_grid_results.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Visualization failed${NC}"
    fi
else
    echo -e "${YELLOW}═══ Step 3/4: Skipped (no evaluation data) ═══${NC}"
fi

echo ""

# Step 4: Report (if evaluation was done)
if [ "$SKIP_EVAL" = false ]; then
    echo -e "${BLUE}═══ Step 4/4: Generating Report ═══${NC}"
    echo ""
    python3 src/main/python/generate_report.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Report generation failed${NC}"
    fi
else
    echo -e "${YELLOW}═══ Step 4/4: Skipped (no evaluation data) ═══${NC}"
fi

# Summary
END_TOTAL=$(date +%s)
TOTAL_TIME=$((END_TOTAL - START_TOTAL))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════╗"
echo "║            Pipeline Complete! 🎉                   ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""

if [ "$SKIP_EVAL" = false ]; then
    echo "Generated files:"
    echo "  📁 Runs: /home/javier/data/runs/"
    echo "  📊 Summaries: grid_results/summary_*.tsv"
    echo "  📈 Plots: grid_results/plots/"
    echo "  📝 Report: grid_results/GRID_SEARCH_REPORT.md"
    echo ""
    echo "Next steps:"
    echo "  1. Review report: cat grid_results/GRID_SEARCH_REPORT.md"
    echo "  2. Check plots: ls grid_results/plots/"
    echo "  3. Analyze best configuration"
    echo "  4. Run statistical significance tests"
else
    echo "Generated files:"
    echo "  📁 Runs: /home/javier/data/runs/"
    echo ""
    echo "To complete analysis, install rec_eval and run:"
    echo "  src/main/scripts/analyze_grid_results.sh"
    echo "  python3 src/main/python/visualize_grid_results.py"
    echo "  python3 src/main/python/generate_report.py"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
