#!/bin/bash

set -e

# Configuration
REMOTE_USER="root"
REMOTE_HOST="aule"
REMOTE_BASE_DIR="/home/javier/prf-llm"
REMOTE_SCRIPTS_DIR="${REMOTE_BASE_DIR}/scripts"
REMOTE_PYTHON_DIR="${REMOTE_BASE_DIR}/python"

# Get the project root directory (3 levels up from scripts folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="${PROJECT_ROOT}/src/main/scripts"

echo "==> Starting deployment to ${REMOTE_USER}@${REMOTE_HOST}"
echo ""

# Step 1: Copy scripts to remote server
echo "==> Copying scripts to ${REMOTE_SCRIPTS_DIR}"
scp "${SCRIPTS_DIR}/run_grid_search.sh" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SCRIPTS_DIR}/"
scp "${SCRIPTS_DIR}/dataset_config.sh" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SCRIPTS_DIR}/"
scp "${SCRIPTS_DIR}/analyze_grid_results.sh" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SCRIPTS_DIR}/"
echo "✓ Scripts copied successfully"

# Step 1b: Copy visualize_grid_results.py to remote python directory
echo "==> Copying visualize_grid_results.py to ${REMOTE_PYTHON_DIR}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p \"${REMOTE_PYTHON_DIR}\""
scp "${PROJECT_ROOT}/src/main/python/visualize_grid_results.py" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PYTHON_DIR}/"
echo "✓ Python visualization script copied successfully"
echo ""
echo ""

# Step 2: Build the project with Maven
echo "==> Building project with Maven (package)"
cd "${PROJECT_ROOT}"
mvn clean package
echo "✓ Maven build completed successfully"
echo ""

# Step 3: Copy JAR with dependencies to remote server
JAR_FILE="${PROJECT_ROOT}/target/prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar"
if [ ! -f "${JAR_FILE}" ]; then
    echo "ERROR: JAR file not found at ${JAR_FILE}"
    exit 1
fi

echo "==> Copying JAR to ${REMOTE_BASE_DIR}"
scp "${JAR_FILE}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/"
echo "✓ JAR copied successfully"
echo ""

echo "==> Deployment completed successfully!"
echo "Remote locations:"
echo "  - Scripts: ${REMOTE_SCRIPTS_DIR}/"
echo "  - JAR: ${REMOTE_BASE_DIR}/prf-llm-0.0.1-SNAPSHOT-jar-with-dependencies.jar"
