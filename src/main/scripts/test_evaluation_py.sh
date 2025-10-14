#!/bin/bash
# Wrapper script to run the Python version of test_evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."
PYTHON_SCRIPT="${SCRIPT_DIR}/../python/test_evaluation.py"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Run Python script with all arguments
python3 "$PYTHON_SCRIPT" "$@"
