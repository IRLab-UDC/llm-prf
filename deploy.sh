#!/bin/bash

# Deploy script for PRF-LLM project
# Copies JAR, Python scripts, and shell scripts to remote server (aule)
# Usage: ./deploy.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REMOTE_USER="javier"
REMOTE_HOST="aule"
REMOTE_BASE_DIR="prf-llm"
LOCAL_JAR="target/prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar"
REMOTE_JAR_NAME="prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar"
mvn clean install
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PRF-LLM Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if JAR exists
if [ ! -f "$LOCAL_JAR" ]; then
    echo -e "${RED}Error: JAR file not found at $LOCAL_JAR${NC}"
    echo -e "${YELLOW}Please run: mvn clean package${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found JAR file: $LOCAL_JAR${NC}"

# Create remote directory structure
echo -e "${BLUE}Creating remote directory structure...${NC}"
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_BASE_DIR}/python ${REMOTE_BASE_DIR}/scripts"
echo -e "${GREEN}✓ Remote directories created${NC}"

# Copy JAR file
echo -e "${BLUE}Copying JAR file to ${REMOTE_HOST}:${REMOTE_BASE_DIR}/...${NC}"
scp "$LOCAL_JAR" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/${REMOTE_JAR_NAME}
echo -e "${GREEN}✓ JAR file copied${NC}"

# Copy Python scripts
echo -e "${BLUE}Copying Python scripts to ${REMOTE_HOST}:${REMOTE_BASE_DIR}/python/...${NC}"
scp -r src/main/python/* ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/python/
echo -e "${GREEN}✓ Python scripts copied${NC}"

# Copy shell scripts
echo -e "${BLUE}Copying shell scripts to ${REMOTE_HOST}:${REMOTE_BASE_DIR}/scripts/...${NC}"
scp -r src/main/scripts/* ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/scripts/
echo -e "${GREEN}✓ Shell scripts copied${NC}"

# Update JAR path in run_grid_search.sh on remote server
echo -e "${BLUE}Updating JAR path in dataset_config.sh on remote server...${NC}"
ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
cd prf-llm/scripts
# Update JAR_PATH to use the JAR in the parent directory
sed -i 's|^JAR_PATH=.*|JAR_PATH="../prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar"|' dataset_config.sh
echo "Updated JAR_PATH in dataset_config.sh"
EOF
echo -e "${GREEN}✓ JAR path updated in dataset_config.sh${NC}"

# Make scripts executable on remote server
echo -e "${BLUE}Making scripts executable on remote server...${NC}"
ssh ${REMOTE_USER}@${REMOTE_HOST} "chmod +x ${REMOTE_BASE_DIR}/scripts/*.sh"
echo -e "${GREEN}✓ Scripts made executable${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Remote structure:${NC}"
echo "  ${REMOTE_HOST}:${REMOTE_BASE_DIR}/"
echo "  ├── ${REMOTE_JAR_NAME}"
echo "  ├── python/"
echo "  │   ├── generate_report.py"
echo "  │   ├── visualize_grid_results.py"
echo "  │   └── mono_t5.py"
echo "  └── scripts/"
echo "      ├── run_grid_search.sh"
echo "      ├── analyze_grid_results.sh"
echo "      ├── dataset_config.sh"
echo "      └── ..."
echo ""
echo -e "${YELLOW}To run on remote server:${NC}"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_BASE_DIR}/scripts"
echo "  ./run_grid_search.sh"
echo ""
