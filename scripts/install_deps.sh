#!/usr/bin/env bash
# install_deps.sh - Install dependencies for Advanced OpenPose Loader
# Clones required repos, creates Python virtual environment, and installs packages.
# Run this before running tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="${REPO_DIR}/deps"
VENV_DIR="${REPO_DIR}/.venv"

echo "=== Advanced OpenPose Loader - Install Dependencies ==="

# Clone dependencies if not already present
clone_if_missing() {
    local name="$1"
    local url="$2"
    local dest="${DEPS_DIR}/${name}"
    if [ ! -d "${dest}/.git" ]; then
        echo "Cloning ${name}..."
        git clone --depth 1 "${url}" "${dest}"
        echo "✓ ${name} cloned"
    else
        echo "✓ ${name} already cloned"
    fi
}

echo ""
echo "Setting up dependencies..."
mkdir -p "${DEPS_DIR}"

# Clone ComfyUI core
clone_if_missing "comfyui" "https://github.com/comfyanonymous/ComfyUI.git"

# Clone required custom node repos
clone_if_missing "ComfyUI-Flux2Klein-Enhancer" "https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer.git"
clone_if_missing "comfyui-flux2fun-controlnet" "https://github.com/bryanmcguire/comfyui-flux2fun-controlnet.git"
clone_if_missing "ComfyUI-Multi-Folder-Loader" "https://github.com/Jiqize/ComfyUI-Multi-Folder-Loader.git"

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "✓ Virtual environment created at ${VENV_DIR}"
else
    echo "✓ Virtual environment already exists at ${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

echo ""
echo "Installing Advanced OpenPose Loader dependencies..."
pip install -r "${REPO_DIR}/requirements.txt" --quiet
pip install pytest --quiet
echo "✓ Advanced OpenPose Loader dependencies installed"

echo ""
echo "Installing ComfyUI dependencies..."
pip install -r "${DEPS_DIR}/comfyui/requirements.txt" --quiet
echo "✓ ComfyUI dependencies installed"

echo ""
echo "=== Dependencies installed successfully! ==="
echo "Run tests with: ${SCRIPT_DIR}/run_test.sh"