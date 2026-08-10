#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/../python/test_evaluation.py"

JAR_ARG=""
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jar)
            JAR_ARG="--jar $2"
            shift 2
            ;;
        --jar=*)
            JAR_ARG="--jar ${1#*=}"
            shift
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

VENV_PATH="${SCRIPT_DIR}/../../../../.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

python3 "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}" $JAR_ARG
