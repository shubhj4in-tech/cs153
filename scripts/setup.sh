#!/usr/bin/env bash
# ============================================================
# setup.sh — One-shot environment setup for 4drecon
#
# Usage:
#   bash scripts/setup.sh
#
# What it does:
#   1. Creates conda env "4drecon" (Python 3.10)
#   2. Installs CUDA-based PyTorch first, then all requirements
#   3. Clones all required repos as git submodules
#   4. Builds the 3DGS CUDA extensions
#   5. Runs download_models.sh to pull model weights
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"; NC="\033[0m"
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
err()  { echo -e "${RED}[setup]${NC} $*" >&2; }

# ── 1. Conda environment ──────────────────────────────────────────────────────
ENV_NAME="4drecon"

if conda env list | grep -q "^${ENV_NAME}"; then
    warn "Conda env '${ENV_NAME}' already exists — skipping creation."
else
    log "Creating conda env '${ENV_NAME}' (Python 3.10) ..."
    conda create -y -n "$ENV_NAME" python=3.10
    log "Conda env created."
fi

# Activate the env for subsequent commands
# shellcheck disable=SC1090
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
log "Activated conda env: $(python --version)"

# ── 2. Install PyTorch with CUDA 12.1 first ───────────────────────────────────
log "Installing PyTorch (CUDA 12.1) ..."
pip install --quiet \
    torch==2.3.0+cu121 \
    torchvision==0.18.0+cu121 \
    torchaudio==2.3.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── 3. Install Python requirements ───────────────────────────────────────────
log "Installing Python requirements ..."
pip install --quiet -r requirements.txt

# ── 4. Clone git submodules ───────────────────────────────────────────────────
log "Cloning required repositories ..."

clone_if_absent() {
    local url="$1"
    local dest="$2"
    shift 2
    if [ -d "$dest/.git" ]; then
        warn "  $dest already exists — skipping."
    else
        log "  Cloning $url → $dest ..."
        git clone "$@" "$url" "$dest"
    fi
}

# gaussian-splatting with its submodules (diff-gaussian-rasterization, simple-knn)
clone_if_absent \
    https://github.com/graphdeco-inria/gaussian-splatting \
    gaussian-splatting \
    --recursive

# Deformable-3D-Gaussians
clone_if_absent \
    https://github.com/ingra14m/Deformable-3D-Gaussians \
    Deformable-3D-Gaussians

# DUSt3R
clone_if_absent \
    https://github.com/naver/dust3r \
    dust3r

# SAM2
clone_if_absent \
    https://github.com/facebookresearch/sam2 \
    sam2

# GaussianSplats3D (JavaScript viewer library — not installed as Python pkg)
clone_if_absent \
    https://github.com/mkkellogg/GaussianSplats3D \
    GaussianSplats3D

# ── 5. Install Python packages from cloned repos ─────────────────────────────
log "Installing DUSt3R ..."
pip install --quiet -r dust3r/requirements.txt
pip install --quiet -e dust3r/

log "Installing SAM2 ..."
pip install --quiet -e sam2/

log "Installing Deformable-3D-Gaussians requirements ..."
if [ -f "Deformable-3D-Gaussians/requirements.txt" ]; then
    pip install --quiet -r Deformable-3D-Gaussians/requirements.txt || true
fi

# ── 6. Build 3DGS CUDA extensions ─────────────────────────────────────────────
log "Building 3DGS CUDA extensions ..."

if [ ! -d "gaussian-splatting/submodules/diff-gaussian-rasterization" ]; then
    err "diff-gaussian-rasterization submodule missing. Run:"
    err "  cd gaussian-splatting && git submodule update --init --recursive"
    exit 1
fi

(
    cd gaussian-splatting
    log "  Building diff-gaussian-rasterization ..."
    pip install --quiet ./submodules/diff-gaussian-rasterization
    log "  Building simple-knn ..."
    pip install --quiet ./submodules/simple-knn
)

# ── 7. Install viewer Node.js deps ───────────────────────────────────────────
if command -v npm &>/dev/null; then
    log "Installing viewer Node.js dependencies ..."
    (cd viewer && npm install --silent)
else
    warn "npm not found — skipping viewer install. Install Node.js >= 18, then run:"
    warn "  cd viewer && npm install"
fi

# ── 8. Download model weights ─────────────────────────────────────────────────
log "Downloading model weights ..."
bash scripts/download_models.sh

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
log "Setup complete!"
echo ""
echo "  To activate the environment in a new shell:"
echo "    conda activate 4drecon"
echo ""
echo "  Quick start:"
echo "    python pipeline/run_pipeline.py --input myvideo.mp4 --output ./scene/"
echo ""
