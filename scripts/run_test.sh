#!/usr/bin/env bash
# run_test.sh - Test Advanced OpenPose Loader
# Assumes dependencies are already installed (run install_deps.sh first).
# Sets up Python environment and runs the test suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="${REPO_DIR}/deps"
VENV_DIR="${REPO_DIR}/.venv"

echo "=== Advanced OpenPose Loader Test Suite ==="

# Check that dependencies are installed
if [ ! -d "${DEPS_DIR}/comfyui/.git" ]; then
    echo "ERROR: Dependencies not installed. Run ${SCRIPT_DIR}/install_deps.sh first."
    exit 1
fi

# Check that virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Virtual environment not found. Run ${SCRIPT_DIR}/install_deps.sh first."
    exit 1
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Set up PYTHONPATH to include cloned deps
export PYTHONPATH="${DEPS_DIR}/comfyui:${DEPS_DIR}/ComfyUI-Flux2Klein-Enhancer:${DEPS_DIR}/comfyui-flux2fun-controlnet:${DEPS_DIR}/ComfyUI-Multi-Folder-Loader:${REPO_DIR}"

echo ""
echo "=== Test 1: Import Advanced OpenPose Loader ==="
python3 -c "
import sys
print('Python version:', sys.version)
print('')

# Test imports
try:
    import comfy.utils
    print('✓ comfy.utils imported')
except ImportError as e:
    print('✗ comfy.utils import failed:', e)
    sys.exit(1)

try:
    import folder_paths
    print('✓ folder_paths imported')
except ImportError as e:
    print('✗ folder_paths import failed:', e)
    sys.exit(1)

try:
    from advanced_openpose_loader import AdvancedOpenposeLoader
    print('✓ AdvancedOpenposeLoader imported')
except ImportError as e:
    print('✗ AdvancedOpenposeLoader import failed:', e)
    sys.exit(1)

print('')
print('All imports successful!')
"

echo ""
echo "=== Test 2: Instantiate Advanced OpenPose Loader ==="
python3 -c "
from advanced_openpose_loader import AdvancedOpenposeLoader

loader = AdvancedOpenposeLoader()
print('✓ AdvancedOpenposeLoader instantiated')
print('  RETURN_TYPES:', loader.RETURN_TYPES)
print('  FUNCTION:', loader.FUNCTION)
print('  CATEGORY:', loader.CATEGORY)
"

echo ""
echo "=== Test 3: Check node schema ==="
python3 -c "
from advanced_openpose_loader import AdvancedOpenposeLoader

inputs = AdvancedOpenposeLoader.INPUT_TYPES()
print('Required inputs:')
for k in inputs.get('required', {}):
    print(f'  - {k}')
print('')
print('Optional inputs:')
for k in inputs.get('optional', {}):
    print(f'  - {k}')
"

echo ""
echo "=== All tests passed! ==="