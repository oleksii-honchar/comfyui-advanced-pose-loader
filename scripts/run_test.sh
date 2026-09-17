#!/bin/bash
set -e

# Run AdvancedOpenposeLoader unit tests
# Usage: ./scripts/run_test.sh [path-to-node-root]
# If no path is given, defaults to the parent of this script's directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_ROOT="${1:-$(dirname "$SCRIPT_DIR")}"

echo "Running AdvancedOpenposeLoader unit tests..."
echo "Node root: $NODE_ROOT"

# Use venv python if it exists, otherwise system python
if [ -x "$NODE_ROOT/.venv/bin/python" ]; then
    PYTHON="$NODE_ROOT/.venv/bin/python"
    echo "Using venv python: $PYTHON"
else
    PYTHON=python3
    echo "Using system python: $PYTHON"
fi

cd "$NODE_ROOT/src/tests"
PYTHONPATH="$NODE_ROOT" "$PYTHON" -m pytest -v "$@"